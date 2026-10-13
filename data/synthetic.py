# -*- coding: utf-8 -*-
"""合成 Bradley-Terry 偏好数据生成（DGP）。Author: 晨星 (CJX0712)

设计原则（踩坑库 B 节：甜点难度 + 防泄漏）：
- rewards 仅由 DGP 持有，仅供评测；fit 接口只接收成对索引。
- 偏好对按 Bradley-Terry 从 rewards 采样，并支持 label_noise 翻转（噪声鲁棒性）。
- reward_scheme 控制难度：
    sparse_peak : 线性趋势 + 单强峰 + 噪声（默认甜点）
    graded      : 纯线性可分 + 噪声（对照：point-wise 也易）
    hard        : 近常数 reward + 大噪声（失败案例：所有方法都难）
"""

from __future__ import annotations

import numpy as np

from core.config import Config
from core.errors import DGPError
from core.seed import get_rng
from core.types import PreferenceDataset, PromptData


def list_schemes() -> list:
    return ["sparse_peak", "graded", "hard"]


def _make_rewards(rng: np.random.Generator, K: int, scheme: str) -> np.ndarray:
    if scheme == "graded":
        # 线性趋势 + 轻噪声（point-wise 也易排）
        trend = np.linspace(-0.5, 0.5, K)
        return trend + 0.06 * rng.standard_normal(K)
    if scheme == "hard":
        # 近常数 + 强噪声（区分度极低）
        return 0.05 * rng.standard_normal(K)
    # sparse_peak（默认）
    norm = (np.arange(K) - (K - 1) / 2.0) / max(K - 1, 1)
    r = 0.45 * norm + 0.12 * rng.standard_normal(K)
    peak = int(rng.integers(0, K))
    r[peak] += 0.9 + 0.25 * abs(rng.standard_normal())
    return r


def make_synthetic(cfg: Config) -> PreferenceDataset:
    """生成合成偏好数据集。rewards 不进入 fit，仅评测使用。"""
    cfg.validate()
    if cfg.k_candidates < 2:
        raise DGPError(f"k_candidates 必须 >= 2，得到 {cfg.k_candidates}")
    rng = get_rng(cfg.seed)
    K = cfg.k_candidates
    prompts: list[PromptData] = []

    for pid in range(cfg.n_prompts):
        r = _make_rewards(rng, K, cfg.reward_scheme)
        pairs: list[tuple] = []
        for _ in range(cfg.n_pairs):
            i, j = rng.choice(K, size=2, replace=False)
            p_win = 1.0 / (1.0 + np.exp(-(r[i] - r[j]) / max(cfg.bt_temp, 1e-6)))
            winner = i if rng.random() < p_win else j
            loser = j if winner == i else i
            if rng.random() < cfg.label_noise:
                winner, loser = loser, winner  # 标签翻转
            pairs.append((int(winner), int(loser)))
        prompts.append(PromptData(prompt_id=pid, rewards=r, pairs=pairs))

    meta = {
        "scheme": cfg.reward_scheme,
        "bt_temp": cfg.bt_temp,
        "label_noise": cfg.label_noise,
        "n_pairs": cfg.n_pairs,
        "k": K,
        "n_prompts": cfg.n_prompts,
        "data_source": "synthetic_bradley_terry",
    }
    return PreferenceDataset(prompts=prompts, k_candidates=K, seed=cfg.seed, meta=meta)
