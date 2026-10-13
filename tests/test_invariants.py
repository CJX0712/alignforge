#!/usr/bin/env python
# AlignForge · 不变量自检套件（可独立运行：python tests/test_invariants.py）
# 每条不变量 = 一条可被交叉验证的硬金标准。全绿才算交付。
# 作者：晨星
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from alignforge import ad, Config, TinyLM, sft, dpo, grpo, reward_fn
from alignforge.autodiff import const
from alignforge.data import make_dataset, target_of, sequences
from alignforge.align import per_seq_logp

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"  {'✅' if ok else '❌'} {name:52s} {detail}")


def small_model(**kw):
    with_lora = kw.pop("with_lora", True)
    d = dict(vocab=16, d_model=12, n_heads=3, n_layers=1, d_ff=24, max_seq=12,
             lora_rank=3, seed=11)
    d.update(kw)
    return TinyLM(Config(**d), with_lora=with_lora)


def grad_check(loss_fn, params, n_check=8, tol=2e-5, eps=1e-6, seed=0):
    """中心差分梯度检验：返回最大相对误差。"""
    rng = np.random.default_rng(seed)
    # 干扰梯度注入，确保每层均有随机扰动（避免对称位）
    noise = [p.v * 0 + rng.normal(0, 0.05, p.v.shape) for p in params]
    loss0 = loss_fn()
    loss0.backward()
    ana = [p.g.copy() if p.g is not None else np.zeros_like(p.v) for p in params]
    max_rel = 0.0
    picks = []
    for i, p in enumerate(params):
        flat = p.v.size
        idxs = rng.choice(flat, size=min(n_check, flat), replace=False)
        picks += [(i, j) for j in idxs]
    for i, j in picks:
        old = params[i].v.flat[j]
        params[i].v.flat[j] = old + eps
        lp = loss_fn().v.item()
        params[i].v.flat[j] = old - eps
        lm = loss_fn().v.item()
        params[i].v.flat[j] = old
        num = (lp - lm) / (2 * eps)
        a = ana[i].flat[j]
        rel = abs(num - a) / max(1e-8, abs(num) + abs(a))
        max_rel = max(max_rel, rel)
    return max_rel


# ---------------- T1/T2 自动微分算子梯度检验 ----------------
def t1_ad_ops():
    rng = np.random.default_rng(3)
    W1 = ad.param(rng.normal(size=(5, 7)) * 0.4, "W1")
    W2 = ad.param(rng.normal(size=(7, 5)) * 0.4, "W2")
    W3 = ad.param(rng.normal(size=(5, 5)) * 0.4, "W3")
    x = rng.normal(size=(4, 5))

    def loss():
        h = ad.rmsnorm(const(x), ad.const(np.abs(np.ones(5)) + 0.5))
        a = ad.silu(ad.matmul(h, W1))
        s = ad.softmax_rows(ad.matmul(a, W2))
        e = ad.exp(ad.scale(ad.matmul(s, W3), 0.3))
        d = ad.sub(e, ad.scale(s, 0.1))
        m = ad.minimum(ad.maximum(d, const(-0.5)), const(0.8))
        return ad.sum_all(ad.mul(m, m)) + ad.neg(ad.sum_all(ad.log_sigmoid(ad.sum_axis(m))))

    loss().backward()
    params = [W1, W2, W3]
    max_rel = grad_check(lambda: loss(), params, n_check=10, seed=5)
    check("T1 自动微分算子组梯度检验 (中心差分)", max_rel < 2e-5, f"maxRelErr={max_rel:.2e}")


def t2_model_gradcheck():
    m = small_model(with_lora=True)
    rng = np.random.default_rng(1)
    ids = rng.integers(2, m.cfg.vocab, size=(3, 6))
    cs = 3

    def loss():
        lp = per_seq_logp(m, ids, cs, use_lora=True)
        return ad.neg(ad.sum_all(lp))

    loss().backward()
    ps = m.trainable()
    max_rel = grad_check(loss, ps, n_check=6, seed=9)
    check("T2 微型 LM 端到端梯度检验 (含 LoRA/注意力/RMSNorm/SwiGLU)", max_rel < 2e-5,
          f"maxRelErr={max_rel:.2e}")


def t3_causal_mask():
    m = small_model(with_lora=False)
    rng = np.random.default_rng(2)
    ids = rng.integers(2, m.cfg.vocab, size=(2, 6))
    l1 = m.logits_np(ids)[:, :4, :]
    ids2 = ids.copy(); ids2[:, 4:] = (ids2[:, 4:] + 5) % m.cfg.vocab
    l2 = m.logits_np(ids2)[:, :4, :]
    ok = np.allclose(l1, l2, atol=1e-12)
    check("T3 因果性：未来 token 扰动不影响过去 logits", ok,
          f"maxDiff={np.abs(l1-l2).max():.2e}")


def t4_lora_rank_and_frozen():
    m = small_model(with_lora=True)
    rng = np.random.default_rng(4)
    for A, B in m.lora.values():
        A.v[...] = rng.normal(size=A.v.shape)
        B.v[...] = rng.normal(size=B.v.shape)
    ok_rank = all(np.linalg.matrix_rank(m.merge_lora()[k]) <= m.cfg.lora_rank
                  for k in m.merge_lora())
    check("T4a LoRA 低秩约束 rank(ΔW) ≤ r", ok_rank, f"r={m.cfg.lora_rank}")
    # B=0 零初始化（Hu et al. 2021）⇒ ΔW=0 ⇒ LoRA 路径与基座输出逐位一致
    m2 = small_model(with_lora=True)
    ids = rng.integers(2, m2.cfg.vocab, size=(2, 6))
    out_lora = m2.logits_np(ids, use_lora=True)
    out_base = m2.logits_np(ids, use_lora=False)
    ok_zero_init = np.array_equal(out_lora, out_base)
    check("T4b LoRA 零初始化：B=0 ⇒ 起点严格等价基座", ok_zero_init,
          f"bitwise_equal={ok_zero_init}")


def t5_reference_frozen():
    m = small_model(with_lora=True, vocab=48)
    prompts = np.array([[2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
    good = np.array([[20, 21, 22, 11], [21, 22, 23, 12]])
    bad = np.array([[20, 21, 22, 13], [21, 22, 23, 14]])
    before = m.state()
    dpo(m, prompts, good, bad, steps=5, lr=5e-3, verbose=False)
    after = m.state()
    ok = all(np.array_equal(before["base"][k], after["base"][k]) for k in before["base"])
    lora_moved = any(not np.array_equal(before["lora"][k][0], after["lora"][k][0])
                     for k in before["lora"])
    check("T5 DPO 中参考模型（基座）严格冻结", ok and lora_moved,
          f"base_frozen={ok}, lora_updated={lora_moved}")


def t6_dpo_gradcheck():
    m = small_model(with_lora=True)
    prompts = np.array([[2, 3, 4], [4, 5, 6]])
    good = np.array([[8, 9, 10], [9, 10, 11]])
    bad = np.array([[10, 11, 12], [11, 12, 13]])
    beta = 0.2
    gs = np.concatenate([prompts, good], 1)
    bl = np.concatenate([prompts, bad], 1)
    cs = prompts.shape[1]

    def loss():
        rg = const(per_seq_logp(m, gs, cs, use_lora=False).v)
        rb = const(per_seq_logp(m, bl, cs, use_lora=False).v)
        pg = per_seq_logp(m, gs, cs, use_lora=True)
        pb = per_seq_logp(m, bl, cs, use_lora=True)
        margin = ad.scale(ad.sub(ad.sub(pg, rg), ad.sub(pb, rb)), beta)
        return ad.neg(ad.sum_all(ad.log_sigmoid(margin)))

    m.zero_lora()
    loss().backward()
    max_rel = grad_check(loss, m.trainable(lora_only=True), n_check=6, seed=13)
    check("T6 DPO 损失梯度检验（隐式奖励路径）", max_rel < 2e-5, f"maxRelErr={max_rel:.2e}")


def t7_grpo_advantage_zero_mean():
    rng = np.random.default_rng(7)
    G, Bp = 8, 16
    prompts = rng.integers(2, 10, size=(Bp, 5))
    comps = rng.integers(10, 28, size=(G * Bp, 4))
    R = reward_fn(np.tile(prompts, (G, 1)), comps).reshape(G, Bp)
    adv = (R - R.mean(0, keepdims=True)) / (R.std(0, keepdims=True) + 1e-6)
    ok = np.allclose(adv.mean(axis=0), 0, atol=1e-12)
    check("T7 GRPO 组内优势严格零均值", ok, f"max|mean|={np.abs(adv.mean(0)).max():.2e}")


def t8_kl_nonnegative():
    rng = np.random.default_rng(8)
    m = small_model(with_lora=True)
    for A, B in m.lora.values():
        A.v[...] = rng.normal(size=A.v.shape) * 0.3
        B.v[...] = rng.normal(size=B.v.shape) * 0.3
    ids = rng.integers(2, m.cfg.vocab, size=(4, 6))
    cs = 3
    ref_c = const(per_seq_logp(m, ids, cs, use_lora=False).v)
    lp = per_seq_logp(m, ids, cs, use_lora=True)
    d = ad.sub(ref_c, lp)
    kl = ad.sum_all(ad.add(ad.sub(ad.exp(d), d), const(-1.0)))
    check("T8 k3-KL 代理 exp(d)−d−1 ≥ 0（全样本）", (kl.v.item() > 0) and np.all(
        (np.exp(d.v) - d.v - 1) > 0), f"KL={kl.v.item():.4f}")


def t9_softmax_rows_sum_one():
    rng = np.random.default_rng(9)
    x = const(rng.normal(size=(3, 4, 16)) * 5)
    s = ad.softmax_rows(x)
    ok = np.allclose(s.v.sum(-1), 1.0, atol=1e-12)
    check("T9 softmax 行概率质量 = 1", ok, f"maxDev={np.abs(s.v.sum(-1)-1).max():.2e}")


def t10_determinism():
    def run():
        m = small_model(with_lora=True, vocab=32, seed=42)
        prompts, good, bad = make_dataset(64, seed=99)
        sft(m, prompts, good, epochs=8, seed=0)
        dpo(m, prompts, good, bad, steps=8, seed=0)
        return m.state()
    s1, s2 = run(), run()
    ok = all(np.array_equal(s1["base"][k], s2["base"][k]) for k in s1["base"])
    ok = ok and all(np.array_equal(s1["lora"][k][0], s2["lora"][k][0]) for k in s1["lora"])
    check("T10 端到端逐位确定性（两次独立运行一致）", ok)


def t11_sft_learns():
    m = small_model(vocab=32, seed=21)
    prompts, good, bad = make_dataset(128, seed=5)
    losses = sft(m, prompts, good, epochs=80, lr=5e-3, seed=0)
    ok = losses[-1] < losses[0] * 0.5
    check("T11 SFT 损失显著下降", ok, f"{losses[0]:.3f} → {losses[-1]:.3f}")


def t12_dpo_margin_up():
    m = small_model(vocab=32, seed=33)
    prompts, good, bad = make_dataset(128, seed=5)
    sft(m, prompts, good, epochs=80, lr=5e-3, seed=0)
    margins = dpo(m, prompts, good, bad, steps=150, lr=1e-3, beta=0.2, seed=0)
    ok = margins[-1] > margins[0]
    check("T12 DPO 隐式奖励 margin 单调上升", ok,
          f"{margins[0]:.4f} → {margins[-1]:.4f}")


def t13_grpo_reward_up():
    m = small_model(vocab=32, seed=55)
    prompts, good, bad = make_dataset(128, seed=5)
    sft(m, prompts, good, epochs=80, lr=5e-3, seed=0)
    dpo(m, prompts, good, bad, steps=100, lr=1e-3, seed=0)
    rs, _ = grpo(m, prompts, steps=40, G=8, comp_len=1, lr=3e-4, seed=0, inner_epochs=4)
    ok = rs[-1] > rs[0]
    check("T13 GRPO 平均奖励上升", ok, f"{rs[0]:.4f} → {rs[-1]:.4f}")


def t14_merge_equivalence():
    """merge 前后（以数学等价方式叠加 ΔW）输出一致：LoRA 数学正确性金标准。"""
    m = small_model(with_lora=True)
    rng = np.random.default_rng(14)
    for A, B in m.lora.values():
        A.v[...] = rng.normal(size=A.v.shape) * 0.2
        B.v[...] = rng.normal(size=B.v.shape) * 0.2
    ids = rng.integers(2, m.cfg.vocab, size=(2, 6))
    out_lora = m.logits_np(ids, use_lora=True)
    scale = m.cfg.lora_alpha / m.cfg.lora_rank
    for k, (A, B) in m.lora.items():
        pre, nm = k.split(".")
        m.params[f"{pre}.{nm}"].v += (A.v @ B.v) * scale      # 数学合并 W += A·B·s
    out_merged = m.logits_np(ids, use_lora=False)
    ok = np.allclose(out_lora, out_merged, atol=1e-10)
    check("T14 LoRA 合并等价：W+BA·s 输出与低秩路径逐元素一致", ok,
          f"maxDiff={np.abs(out_lora-out_merged).max():.2e}")


if __name__ == "__main__":
    print("=" * 78)
    print("AlignForge 不变量自检套件  ·  作者：晨星")
    print("=" * 78)
    tests = [t1_ad_ops, t2_model_gradcheck, t3_causal_mask, t4_lora_rank_and_frozen,
             t5_reference_frozen, t6_dpo_gradcheck, t7_grpo_advantage_zero_mean,
             t8_kl_nonnegative, t9_softmax_rows_sum_one, t10_determinism,
             t11_sft_learns, t12_dpo_margin_up, t13_grpo_reward_up, t14_merge_equivalence]
    for t in tests:
        t()
    print("-" * 78)
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"结果：{npass}/{len(RESULTS)} 全绿" + ("  ✦" if npass == len(RESULTS) else "  ⚠ 有失败项"))
    sys.exit(0 if npass == len(RESULTS) else 1)
