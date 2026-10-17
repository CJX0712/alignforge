# AlignForge · 微型因果 Transformer-LM + LoRA
# 架构站在现代 SOTA Transformer 原语之上：Pre-Norm RMSNorm · RoPE-free 学习位置嵌入 ·
# 多头因果自注意力 · SwiGLU FFN · 绑定嵌入输出头。LoRA 遵循 Hu et al. 2021，
# 作用于每层 Wq/Wv（ΔW = B·A·scale，基座冻结）。
# 作者：晨星
from __future__ import annotations

import numpy as np

from . import autodiff as ad
from .autodiff import const, param


class Config:
    def __init__(self, vocab=48, d_model=32, n_heads=4, n_layers=2, d_ff=88,
                 max_seq=16, lora_rank=4, lora_alpha=1.0, seed=7):
        self.vocab, self.d_model, self.n_heads = vocab, d_model, n_heads
        self.n_layers, self.d_ff, self.max_seq = n_layers, d_ff, max_seq
        self.lora_rank, self.lora_alpha = lora_rank, lora_alpha
        self.seed = seed
        self.d_head = d_model // n_heads


def _xavier(rng, fan_in, fan_out):
    bound = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-bound, bound, size=(fan_in, fan_out))


class TinyLM:
    """纯 NumPy 微型因果语言模型。params 全部为 ad.param 叶子。"""

    def __init__(self, cfg: Config, with_lora=True):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.rng = rng
        d, V = cfg.d_model, cfg.vocab
        p = {}
        p["E"] = param(_xavier(rng, V, d) * 0.5, "E")          # token 嵌入（绑定输出头）
        p["pos"] = param(_xavier(rng, cfg.max_seq, d) * 0.3, "pos")
        for l in range(cfg.n_layers):
            pre = f"L{l}."
            p[pre + "g1"] = param(np.ones(d), pre + "g1")
            p[pre + "Wq"] = param(_xavier(rng, d, d), pre + "Wq")
            p[pre + "Wk"] = param(_xavier(rng, d, d), pre + "Wk")
            p[pre + "Wv"] = param(_xavier(rng, d, d), pre + "Wv")
            p[pre + "Wo"] = param(_xavier(rng, d, d), pre + "Wo")
            p[pre + "g2"] = param(np.ones(d), pre + "g2")
            p[pre + "W1"] = param(_xavier(rng, d, cfg.d_ff), pre + "W1")
            p[pre + "W3"] = param(_xavier(rng, d, cfg.d_ff), pre + "W3")
            p[pre + "W2"] = param(_xavier(rng, cfg.d_ff, d), pre + "W2")
        p["gF"] = param(np.ones(d), "gF")
        self.params = p

        self.lora = {}
        if with_lora:
            for l in range(cfg.n_layers):
                for nm in ("Wq", "Wv"):
                    A = param(rng.normal(0, 0.02, size=(d, cfg.lora_rank)), f"lora{l}.{nm}.A")
                    B = param(np.zeros((cfg.lora_rank, d)), f"lora{l}.{nm}.B")
                    self.lora[f"L{l}.{nm}"] = (A, B)
        # 因果掩码（常数，不可训练）
        m = np.full((cfg.max_seq, cfg.max_seq), -1e9)
        m = np.triu(m, k=1)
        self.causal = const(m[None, None, :, :], "causal_mask")
        self._causal_np = m[None, None, :, :]

    # ---- 前向（返回 logits Node，缓存各中间节点供复用/检查）----
    def forward(self, ids: np.ndarray, use_lora=True):
        """ids: (B,T) int。返回 logits Node (B,T,V)。"""
        cfg = self.cfg
        B, T = ids.shape
        oh = np.zeros((B, T, cfg.vocab))
        oh[np.arange(B)[:, None], np.arange(T)[None, :], ids] = 1.0
        h = ad.matmul(const(oh), self.params["E"])          # (B,T,d)
        h = ad.add(h, ad.reshape(ad.slice_node(self.params["pos"], (slice(T),)),
                                 (1, T, cfg.d_model)))
        scale_lora = cfg.lora_alpha / cfg.lora_rank
        for l in range(cfg.n_layers):
            pre = f"L{l}."
            x = ad.rmsnorm(h, self.params[pre + "g1"])
            q = ad.matmul(x, self.params[pre + "Wq"])
            k = ad.matmul(x, self.params[pre + "Wk"])
            v = ad.matmul(x, self.params[pre + "Wv"])
            if use_lora and f"L{l}.Wq" in self.lora:
                A, Bn = self.lora[f"L{l}.Wq"]
                q = ad.add(q, ad.scale(ad.matmul(ad.matmul(x, A), Bn), scale_lora))
            if use_lora and f"L{l}.Wv" in self.lora:
                A, Bn = self.lora[f"L{l}.Wv"]
                v = ad.add(v, ad.scale(ad.matmul(ad.matmul(x, A), Bn), scale_lora))
            # 分头：(B,T,d) -> (B,H,T,dh)
            def heads(t):
                t = ad.reshape(t, (B, T, cfg.n_heads, cfg.d_head))
                return ad.transpose(t, (0, 2, 1, 3))
            qh, kh, vh = heads(q), heads(k), heads(v)
            att = ad.matmul(qh, ad.transpose(kh, (0, 1, 3, 2)))     # (B,H,T,T)
            att = ad.scale(att, cfg.d_head ** -0.5)
            mask = np.triu(np.full((T, T), -1e9), k=1)[None, None, :, :]
            att = ad.add(att, const(mask))
            att = ad.softmax_rows(att)
            ctx = ad.matmul(att, vh)                                # (B,H,T,dh)
            ctx = ad.transpose(ctx, (0, 2, 1, 3))
            ctx = ad.reshape(ctx, (B, T, cfg.d_model))
            h = ad.add(h, ad.matmul(ctx, self.params[pre + "Wo"]))
            x2 = ad.rmsnorm(h, self.params[pre + "g2"])
            ffn = ad.matmul(ad.mul(ad.silu(ad.matmul(x2, self.params[pre + "W1"])),
                                   ad.matmul(x2, self.params[pre + "W3"])),
                            self.params[pre + "W2"])
            h = ad.add(h, ffn)
        h = ad.rmsnorm(h, self.params["gF"])
        logits = ad.matmul(h, ad.transpose(self.params["E"], (1, 0)))  # 绑定头 (B,T,V)
        self.last_nodes = {"h": h, "logits": logits}
        return logits

    # ---- 纯 numpy 推理（无 tape，供采样/评估）----
    def logits_np(self, ids, use_lora=True):
        return self.forward(ids, use_lora=use_lora).v

    def logprobs_np(self, ids, use_lora=True):
        lg = self.logits_np(ids, use_lora)
        z = lg - lg.max(axis=-1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

    # ---- 可训练参数清单 ----
    def trainable(self, lora_only=False):
        if lora_only:
            return [p for pair in self.lora.values() for p in pair]
        return list(self.params.values()) + [p for pair in self.lora.values() for p in pair]

    def snapshot(self):
        return {k: v.v.copy() for k, v in self.params.items()}

    def restore(self, snap):
        for k, v in self.params.items():
            v.v[...] = snap[k]
            v.g = None

    def state(self):
        """全量状态（基座 + LoRA），用于冻结参考模型。"""
        return {"base": self.snapshot(),
                "lora": {k: (A.v.copy(), B.v.copy()) for k, (A, B) in self.lora.items()}}

    def load_state(self, st):
        self.restore(st["base"])
        for k, (A, B) in self.lora.items():
            A.v[...] = st["lora"][k][0]
            B.v[...] = st["lora"][k][1]
            A.g = B.g = None

    def merge_lora(self):
        """返回 LoRA 合并后的 ΔW 字典（用于秩不变量与 merge-等价检验）。
        前向为 q += scale·(x@A)@B，等效权重增量 ΔW = A@B（使 x@ΔW = (x@A)@B）。"""
        cfg = self.cfg
        s = cfg.lora_alpha / cfg.lora_rank
        return {k: (A.v @ B.v) * s for k, (A, B) in self.lora.items()}

    def zero_lora(self):
        for A, B in self.lora.values():
            B.v[...] = 0.0
            A.g = B.g = None
