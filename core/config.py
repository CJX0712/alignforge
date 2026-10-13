# -*- coding: utf-8 -*-
"""全局配置：环境变量覆盖 + schema 校验。Author: 晨星 (CJX0712)"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    """单一配置来源。环境变量 ALIGNFORGE_* 覆盖默认值。"""

    n_prompts: int = 64
    k_candidates: int = 10
    n_pairs: int = 24
    bt_temp: float = 0.25
    label_noise: float = 0.10
    reward_scheme: str = "sparse_peak"
    seed: int = 42
    n_seeds: int = 3

    @classmethod
    def from_env(cls) -> "Config":
        def _int(k: str, d: int) -> int:
            v = os.environ.get(f"ALIGNFORGE_{k}")
            return int(v) if v is not None else d

        def _float(k: str, d: float) -> float:
            v = os.environ.get(f"ALIGNFORGE_{k}")
            return float(v) if v is not None else d

        def _str(k: str, d: str) -> str:
            v = os.environ.get(f"ALIGNFORGE_{k}")
            return v if v is not None else d

        cfg = cls(
            n_prompts=_int("N_PROMPTS", cls.n_prompts),
            k_candidates=_int("K_CANDIDATES", cls.k_candidates),
            n_pairs=_int("N_PAIRS", cls.n_pairs),
            bt_temp=_float("BT_TEMP", cls.bt_temp),
            label_noise=_float("LABEL_NOISE", cls.label_noise),
            reward_scheme=_str("REWARD_SCHEME", cls.reward_scheme),
            seed=_int("SEED", cls.seed),
            n_seeds=_int("N_SEEDS", cls.n_seeds),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.n_prompts <= 0:
            raise ValueError(f"[E300] n_prompts 必须 > 0，得到 {self.n_prompts}")
        if self.k_candidates < 2:
            raise ValueError(f"[E300] k_candidates 必须 >= 2，得到 {self.k_candidates}")
        if self.n_pairs < 1:
            raise ValueError(f"[E300] n_pairs 必须 >= 1，得到 {self.n_pairs}")
        if self.bt_temp <= 0:
            raise ValueError(f"[E300] bt_temp 必须 > 0，得到 {self.bt_temp}")
        if not (0.0 <= self.label_noise <= 1.0):
            raise ValueError(f"[E300] label_noise 必须 ∈ [0,1]，得到 {self.label_noise}")
        if self.reward_scheme not in {"sparse_peak", "graded", "hard"}:
            raise ValueError(f"[E300] 未知 reward_scheme: {self.reward_scheme}")

    def to_dict(self) -> dict:
        return asdict(self)


# 训练超参（不进 env，集中管理，供各 aligner 引用）
@dataclass(frozen=True)
class TrainConfig:
    lr: float = 0.5
    n_epochs: int = 200
    beta_dpo: float = 2.0
    beta_ipo: float = 6.0
    beta_kto: float = 1.0
    margin_kto: float = 0.5
    lambda_desirable: float = 1.0
    lambda_undesirable: float = 1.0
    # ---- AlignFuse 旗舰组件超参 ----
    fuse_gamma: float = 3.0  # C3 边际融合锐度（|a| 越大越偏 DPO）
    fuse_tau0: float = 0.5  # C3 边际阈值：|a|=tau0 时 α=0.5
    fuse_beta_min: float = 0.5  # C1 β 退火起点
    fuse_beta_max: float = 3.0  # C1 β 退火终点（= 最终有效温度）
    fuse_anneal_power: float = 2.0  # C1 退火曲线指数 p
    fuse_kappa: float = 1.0  # C2 一致性门控锐度
    fuse_consistency_bias: float = 2.0  # C2 门控偏置（防止证据少时过度裁剪）
    fuse_kl_budget: float = 1.3  # C4 平均 KL(π‖π_ref) 上限
    grad_clip: float = 50.0  # 每轮 logit 增量上限（梯度裁剪，防 IPO 过冲发散）
