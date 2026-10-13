#!/usr/bin/env python
# AlignForge · 端到端演示 / 基准（确定性可复现）
# 运行：python run.py
# 管线：SFT → DPO → GRPO，每段以可验证指标与硬不变量度量。作者：晨星
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from alignforge import Config, TinyLM, sft, dpo, grpo
from alignforge.data import make_dataset, target_of
from alignforge.align import reward_fn


def topk(model, prompt, k=5):
    """给定 prompt，返回补全首 token 的 top-k 预测。"""
    lp = model.logprobs_np(prompt[None, :])[0, -1]
    idx = np.argsort(-lp)[:k]
    return [(int(i), float(lp[i])) for i in idx]


def main():
    print("=" * 72)
    print("✦ AlignForge · 世界顶级 LLM 对齐后训练系统  作者：晨星")
    print("=" * 72)
    cfg = Config(vocab=32, d_model=32, n_heads=4, n_layers=2, d_ff=88,
                 max_seq=12, lora_rank=4, lora_alpha=1.0, seed=33)
    m = TinyLM(cfg, with_lora=True)
    prompts, good, bad = make_dataset(256, seed=5)
    print(f"语料：train={len(prompts)}  prompt=[a,b] 补全=单token 目标=复刻 a")

    # ---- SFT ----
    print("\n[阶段一] SFT 监督微调（Adam, 80 epoch）")
    losses = sft(m, prompts, good, epochs=80, lr=5e-3, seed=0)
    print(f"  CE: {losses[0]:.4f} → {losses[-1]:.4f}")

    # ---- DPO ----
    print("[阶段二] DPO 直接偏好优化（LoRA, β=0.2）")
    margins = dpo(m, prompts, good, bad, steps=150, lr=1e-3, beta=0.2, seed=0)
    print(f"  margin(β·Δlogπ): {margins[0]:.4f} → {margins[-1]:.4f}")

    # ---- GRPO ----
    print("[阶段三] GRPO 组相对策略优化（PPO 裁剪 + k3-KL）")
    rs, kl = grpo(m, prompts, steps=60, G=8, comp_len=1, lr=3e-4, seed=0, inner_epochs=4)
    print(f"  reward: {rs[0]:.4f} → {rs[-1]:.4f}   avg KL: {np.mean(kl):.5f}")

    # ---- 演示：对齐前（ref=SFT 基座） vs 对齐后 ----
    print("\n[演示] prompt→补全 top-3 预测（对齐后策略）")
    m_ref = TinyLM(cfg, with_lora=False); m_ref.restore(m.snapshot())  # 仅取基座即 SFT 后
    for a, b in [(2, 5), (4, 7), (9, 3)]:
        p = np.array([a, b])
        tgt = target_of(p[None, :])[0]
        tk = topk(m, p, 3)
        hit = any(i == tgt for i, _ in tk)
        print(f"  prompt={p.tolist()}  目标={int(tgt)}  预测top3={tk}  "
              f"{'✅命中' if hit else '⚠未命中'}")

    print("\n对齐完成。不变量自检见：python tests/test_invariants.py")


if __name__ == "__main__":
    main()
