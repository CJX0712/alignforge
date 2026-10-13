# AlignForge · 对齐后训练三段管线：SFT → DPO → GRPO
#   SFT  : 监督微调（Ouyang et al. 2022, InstructGPT 阶段一）
#   DPO  : 直接偏好优化（Rafailov et al. 2023），隐式奖励 β·Δlogπ，免奖励模型
#   GRPO : 组相对策略优化（Shao et al. 2024, DeepSeekMath），组内零均值优势 + k3 KL 惩罚
# 参考（reference）模型通过 state 冻结；k3 KL 代理 exp(d)−d−1 ≥ 0（Schulman 2020）。
# 确定性可复现：所有随机性来自种子化的 default_rng。
# 作者：晨星
from __future__ import annotations
import numpy as np
from . import autodiff as ad
from .autodiff import const
from .model import TinyLM


# ---------------- Adam（Kingma & Ba 2015，含偏置校正） ----------------
class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
        self.ps = params
        self.m = [np.zeros_like(p.v) for p in params]
        self.v = [np.zeros_like(p.v) for p in params]
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, b1, b2, eps, clip
        self.t = 0

    def step(self):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        # 全局范数梯度裁剪（Pascanu et al. 2013），鲁棒性金标准
        if self.clip and self.clip > 0:
            gn = np.sqrt(sum(np.sum(p.g * p.g) for p in self.ps if p.g is not None))
            scale = self.clip / max(self.clip, gn)
        else:
            scale = 1.0
        for i, p in enumerate(self.ps):
            g = p.g
            if g is None:
                continue
            g = g * scale
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            p.v -= self.lr * (self.m[i] / c1) / (np.sqrt(self.v[i] / c2) + self.eps)
            p.g = None


# ---------------- logp 工具 ----------------

def per_seq_logp(model, ids, comp_start, use_lora=True):
    """补全区段逐序列 logπ(y|x)之和，返回 Node (B,)。"""
    logits = model.forward(ids, use_lora=use_lora)      # (B,T,V)
    lse = ad.logsumexp_rows(logits)                     # (B,T,1)
    B, T = ids.shape
    oh = np.zeros(logits.v.shape)
    oh[np.arange(B)[:, None], np.arange(T)[None, :], ids] = 1.0
    oh[:, :comp_start, :] = 0.0
    picked = ad.mul(logits - lse, const(oh))
    flat = ad.reshape(picked, (B, -1))
    return ad.sum_axis(flat, axis=-1)                   # (B,)


# ---------------- 阶段一：SFT ----------------

def sft(model, prompts, comps, epochs=60, lr=3e-3, bs=32, seed=0, verbose=False):
    """监督微调：补全区段教师强制交叉熵。训练全部基座参数（LoRA 保持零）。"""
    rng = np.random.default_rng(seed)
    seqs = np.concatenate([prompts, comps], axis=1)
    cs = prompts.shape[1]
    n = len(seqs)
    opt = Adam(model.trainable(), lr=lr)
    losses = []
    for ep in range(epochs):
        idx = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            batch = seqs[idx[i:i + bs]]
            B, T = batch.shape
            logits = model.forward(batch)
            lse = ad.logsumexp_rows(logits)
            oh = np.zeros((B, T, model.cfg.vocab))
            oh[np.arange(B)[:, None], np.arange(T)[None, :], batch] = 1.0
            oh[:, :cs, :] = 0.0
            ntok = oh.sum()
            loss = ad.scale(ad.sum_all(ad.mul(logits - lse, const(oh))), -1.0 / ntok)
            loss.backward()
            opt.step()
            tot += loss.v.item() * ntok
        losses.append(tot / n)
        if verbose and (ep + 1) % 20 == 0:
            print(f"  [SFT] epoch {ep+1:3d}  CE {losses[-1]:.4f}")
    return losses


# ---------------- 阶段二：DPO ----------------

def dpo(model, prompts, good, bad, steps=200, lr=2e-3, beta=0.2,
        bs=32, seed=0, verbose=False):
    """直接偏好优化。基座冻结，仅 LoRA 可训练；ref = use_lora=False 前向（= SFT 基座）。
    margin = β[(logπ_w−logπ_ref,w) − (logπ_l−logπ_ref,l)]；loss = −log σ(margin)。"""
    model.zero_lora()
    opt = Adam(model.trainable(lora_only=True), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(prompts)
    gs_all = np.concatenate([prompts, good], axis=1)
    bs_all = np.concatenate([prompts, bad], axis=1)
    cs = prompts.shape[1]
    margins = []
    for st in range(steps):
        idx = rng.permutation(n)[:bs]
        gs, bl = gs_all[idx], bs_all[idx]
        rg = const(per_seq_logp(model, gs, cs, use_lora=False).v)   # ref（冻结）
        rb = const(per_seq_logp(model, bl, cs, use_lora=False).v)
        pg = per_seq_logp(model, gs, cs, use_lora=True)             # policy
        pb = per_seq_logp(model, bl, cs, use_lora=True)
        margin = ad.scale(ad.sub(ad.sub(pg, rg), ad.sub(pb, rb)), beta)
        loss = ad.neg(ad.log_sigmoid(margin))
        loss.backward()
        opt.step()
        if (st + 1) % 20 == 0:
            margins.append(margin.v.mean())
            if verbose:
                print(f"  [DPO] step {st+1:3d}  margin {margins[-1]:.4f}")
    return margins


# ---------------- 阶段三：GRPO ----------------

def reward_fn(prompts, comps):
    """奖励：末 token 命中可计算目标 +1；中性 token 每个 +0.05。"""
    from .data import target_of
    tgt = target_of(prompts)
    r = (comps[:, -1] == tgt).astype(np.float64)
    neutral = ((comps[:, :-1] >= 20) & (comps[:, :-1] < 28)).sum(axis=1) * 0.05
    return r + neutral


def sample_completions(model, prompts, G, comp_len, rng, temp=1.0, prefix=None):
    """温度采样补全 (G*Bp, comp_len)。可选中性前缀 prefix（消除自回归暴露偏差）。"""
    full = np.tile(prompts, (G, 1))
    comps = np.zeros((full.shape[0], comp_len), dtype=int)
    if prefix is not None:
        cur = np.concatenate([full, np.tile(prefix, (full.shape[0], 1))], axis=1)
    else:
        cur = full
    for t in range(comp_len):
        lg = model.logits_np(cur)[:, -1, :] / temp
        z = lg - lg.max(axis=-1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=-1, keepdims=True)
        nxt = np.array([rng.choice(model.cfg.vocab, p=pp) for pp in p])
        comps[:, t] = nxt
        if t + 1 < comp_len:
            cur = np.concatenate([cur, nxt[:, None]], axis=1)
    return full, comps


def grpo(model, prompts, steps=100, G=8, comp_len=1, prefix=None, lr=3e-4,
         beta_kl=0.05, temp=1.0, clip_eps=0.2, inner_epochs=4, seed=0, verbose=False):
    """组相对策略优化（DeepSeekMath 式）。
    采样：旧策略温度采样 G 个补全/组（comp_len 通常=1，配合中性前缀消除暴露偏差），
          优势 A = (r−μ_group)/σ_group（组内零均值）。
    更新：inner_epochs 个内层步，PPO 式裁剪代理
        L = −mean(min(r·A, clip(r,1±ε)·A)) + β·KL_k3(π_ref‖π)，k3 裁剪防溢出。
    KL 用 k3 估计 exp(d)−d−1 ≥ 0（Schulman 2020）；ref = 进入 GRPO 时的冻结全量状态。"""
    rng = np.random.default_rng(seed)
    ref_state = model.state()
    opt = Adam(model.trainable(), lr=lr)
    rewards_hist, kl_hist = [], []
    n = len(prompts)
    for st in range(steps):
        P = prompts[rng.permutation(n)[:16]]
        Lp = P.shape[1]
        full, comps = sample_completions(model, P, G, comp_len, rng, temp, prefix=prefix)
        seqs = (np.concatenate([full, np.tile(prefix, (full.shape[0], 1)), comps], axis=1)
                if prefix is not None else np.concatenate([full, comps], axis=1))
        comp_start = Lp + (len(prefix) if prefix is not None else 0)
        R = reward_fn(full, comps).reshape(G, -1)
        adv = ((R - R.mean(axis=0, keepdims=True)) /
               (R.std(axis=0, keepdims=True) + 1e-6)).reshape(-1)
        old_c = const(per_seq_logp(model, seqs, comp_start, use_lora=True).v)
        adv_c = const(adv)
        cur_state = model.state()
        model.load_state(ref_state)
        ref_c = const(per_seq_logp(model, seqs, comp_start, use_lora=True).v)
        model.load_state(cur_state)
        for e in range(inner_epochs):
            lp = per_seq_logp(model, seqs, comp_start, use_lora=True)
            ratio = ad.exp(ad.sub(lp, old_c))
            un = ad.mul(ratio, adv_c)
            clipped = ad.mul(ad.minimum(ad.maximum(ratio, const(1 - clip_eps)),
                                        const(1 + clip_eps)), adv_c)
            pg = ad.neg(ad.sum_all(ad.minimum(un, clipped)))
            d = ad.clip(ad.sub(ref_c, lp), -10.0, 10.0)             # 防 exp 溢出
            kl_seq = ad.add(ad.sub(ad.exp(d), d), const(-1.0))      # exp(d)−d−1 ≥ 0
            loss = ad.add(pg, ad.scale(kl_seq, beta_kl))
            loss.backward()
            opt.step()
        rewards_hist.append(R.mean())
        kl_hist.append(kl_seq.v / len(seqs))
        if verbose and (st + 1) % 20 == 0:
            print(f"  [GRPO] step {st+1:3d}  reward {rewards_hist[-1]:.4f}  KL {kl_hist[-1]:.5f}")
    return rewards_hist, kl_hist
