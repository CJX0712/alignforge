# AlignForge · 合成对齐语料（确定性生成）
# 任务设计（可学习、有强偏好信号、可被微型 Transformer 掌握）：
#   prompt = [a, b]（首 token a 即"目标人格"）；completion 末 token 应复刻 a。
#   good 补全：末 token = a（正确复刻）；bad 补全：末 token = 错位 token。
#   GRPO 奖励：末 token == a ⇒ +1，否则 0；中性 token 每个 +0.05（稠密奖励）。
# 该信号在 SFT→DPO→GRPO 三段中均可量化放大，使对齐效果清晰可验证。
# 作者：晨星
from __future__ import annotations
import numpy as np

PROMPT_ATTRS = list(range(2, 10))      # prompt 首 token 候选（属性/人格）


def target_of(prompt_ids: np.ndarray) -> np.ndarray:
    """目标末 token = prompt 首 token（复刻任务）。"""
    return prompt_ids[:, 0].copy()


def make_dataset(n=256, prompt_len=2, comp_len=1, seed=123):
    """返回 prompts (n, prompt_len), good (n, comp_len), bad (n, comp_len)。
    纯净对齐任务：prompt=[a,b]（a 即目标人格），补全为单 token。
    good：复刻 a（[a]）；bad：错位 token [c≠a]。无中性 token 干扰，微型模型可干净掌握。"""
    rng = np.random.default_rng(seed)
    prompts = rng.choice(PROMPT_ATTRS, size=(n, prompt_len))
    tgt = target_of(prompts)
    good = tgt.reshape(-1, 1).copy()                              # (n,1) 复刻首 token
    wrong = (tgt + 1 + rng.integers(0, 2)) % 8 + 2
    wrong = np.where(wrong == tgt, (wrong + 1) % 8 + 2, wrong)    # 保证 ≠ 目标
    bad = wrong.reshape(-1, 1).copy()                             # (n,1) 错位
    return prompts, good, bad


def sequences(prompts, comps):
    """prompt+补全 拼接为完整序列 (n, L)。"""
    return np.concatenate([prompts, comps], axis=1)
