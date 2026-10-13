# -*- coding: utf-8 -*-
"""领域数据类型。Author: 晨星 (CJX0712)"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class PreferencePair:
    """一条成对偏好：在 prompt_id 下，winner 胜于 loser（候选索引）。"""

    prompt_id: int
    winner: int
    loser: int


@dataclass
class PromptData:
    """单个 prompt 的全部信息。rewards 仅用于评测，绝不进入 fit。"""

    prompt_id: int
    rewards: np.ndarray  # (K,) ground-truth reward，评测用
    pairs: List[tuple] = field(default_factory=list)  # [(winner, loser), ...]

    def __post_init__(self) -> None:
        self.rewards = np.asarray(self.rewards, dtype=np.float64)


@dataclass
class PreferenceDataset:
    """偏好数据集。prompts 之间独立；rewards 仅评测用。"""

    prompts: List[PromptData]
    k_candidates: int
    seed: int = 42
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for p in self.prompts:
            if p.rewards.shape[0] != self.k_candidates:
                raise ValueError(
                    f"prompt {p.prompt_id} rewards 维度 {p.rewards.shape[0]} "
                    f"!= k_candidates {self.k_candidates}"
                )

    def all_pairs(self) -> List[PreferencePair]:
        out: List[PreferencePair] = []
        for p in self.prompts:
            for w, l in p.pairs:
                out.append(PreferencePair(p.prompt_id, w, l))
        return out

    def single_labels(self, min_margin: float = 0.0) -> List[tuple]:
        """为 KTO 等单样本方法提供 (prompt_id, candidate_idx, desirable)。

        严格只从「已观测的成对反馈」导出，绝不读取 rewards（防泄漏）：
        赢过的候选 = desirable，输过的候选 = undesirable，按净胜场数加权，
        净胜场为 0 的候选（无证据）不产出标签。
        """
        out: List[tuple] = []
        for p in self.prompts:
            K = self.k_candidates
            score = np.zeros(K, dtype=np.float64)
            seen = np.zeros(K, dtype=bool)
            for w, l in p.pairs:
                score[w] += 1.0
                score[l] -= 1.0
                seen[w] = True
                seen[l] = True
            for idx in range(K):
                if not seen[idx]:
                    continue
                if abs(score[idx]) < max(min_margin, 1e-9):
                    continue  # 净胜场为 0，无证据，不产出标签
                out.append((p.prompt_id, idx, bool(score[idx] > 0.0)))
        return out


@dataclass
class MetricBundle:
    """一个方法在单个 seed 上的指标集合。"""

    method: str
    seed: int
    expected_reward: float  # E_{y~pi}[r] 原始
    norm_reward: float  # 相对 uniform→optimal 归一化 [0,1]
    kendall_tau: float  # 与真实 reward 排序一致性
    spearman_rho: float
    kl_ref: float  # KL(pi || pi_ref) 平均
    win_rate_vs_ref: float  # 期望 reward 优于参考策略比例
    elapsed_sec: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class AlignResult:
    """一个方法的完整结果。"""

    method: str
    seed: int
    logits: Dict[int, np.ndarray]  # prompt_id -> (K,) logits
    metrics: MetricBundle
    meta: dict = field(default_factory=dict)
