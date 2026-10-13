# -*- coding: utf-8 -*-
"""对齐评测指标。Author: 晨星 (CJX0712)

所有方法统一在「期望 reward / 排序一致性 / KL 约束」三套口径下比较。
norm_reward 把 E[r] 归一到 [uniform=0, optimal=1]，跨 DGP 可比。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr

from core.types import MetricBundle


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def compute_metrics(
    method: str,
    logits: dict,
    dataset,
    seed: int,
    pi_ref: np.ndarray | None = None,
) -> MetricBundle:
    K = dataset.k_candidates
    if pi_ref is None:
        pi_ref = np.full(K, 1.0 / K)
    log_ref = np.log(pi_ref + 1e-12)

    Er_p = 0.0
    Er_u = 0.0
    Er_o = 0.0
    kl_sum = 0.0
    win = 0
    L: list = []
    R: list = []
    for p in dataset.prompts:
        r = p.rewards
        lg = np.asarray(logits[p.prompt_id], dtype=np.float64)
        pi = softmax(lg)
        pol = float(pi @ r)
        Er_p += pol
        Er_u += float(r.mean())
        Er_o += float(r.max())
        kl_sum += float(np.sum(pi * (np.log(pi + 1e-12) - log_ref)))
        if pol > float(r.mean()):
            win += 1
        L.extend(lg.tolist())
        R.extend(r.tolist())

    n = max(len(dataset.prompts), 1)
    Er_p /= n
    Er_u /= n
    Er_o /= n
    kl_sum /= n

    denom = Er_o - Er_u
    norm = (Er_p - Er_u) / denom if abs(denom) > 1e-9 else 0.0
    if not np.isfinite(norm):
        norm = 0.0
    norm = float(min(max(norm, 0.0), 1.0))

    try:
        tau, _ = kendalltau(R, L)
    except Exception:
        tau = 0.0
    try:
        rho, _ = spearmanr(R, L)
    except Exception:
        rho = 0.0
    if not np.isfinite(tau):
        tau = 0.0
    if not np.isfinite(rho):
        rho = 0.0

    return MetricBundle(
        method=method,
        seed=seed,
        expected_reward=Er_p,
        norm_reward=norm,
        kendall_tau=float(tau),
        spearman_rho=float(rho),
        kl_ref=kl_sum,
        win_rate_vs_ref=win / n,
    )
