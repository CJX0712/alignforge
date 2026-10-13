# -*- coding: utf-8 -*-
"""DPO (Direct Preference Optimization, Rafailov et al. 2023, NeurIPS)。Author: 晨星 (CJX0712)

强基线 / SOTA 对标：直接优化策略，无需显式奖励模型。
参考策略取 uniform，故 log-ratio 退化为 logit 差；梯度为完整 softmax 雅可比。
"""

from __future__ import annotations

import numpy as np

from core.config import TrainConfig

from .base import BaseAligner


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class DPOAligner(BaseAligner):
    name = "DPO"

    def __init__(self, tc: TrainConfig | None = None) -> None:
        super().__init__(tc)

    def fit(self, dataset):
        self._init_logits(dataset)
        lr = self.tc.lr
        beta = self.tc.beta_dpo
        K = dataset.k_candidates
        eye = np.eye(K)
        for _ in range(self.tc.n_epochs):
            for p in dataset.prompts:
                lg = self.logits[p.prompt_id]
                grad = np.zeros(K)
                for w, l in p.pairs:
                    a = lg[w] - lg[l]
                    delta = _sigmoid(beta * a)
                    coef = (1.0 - delta) * beta
                    # dm/dθ = e_w - e_l（log-softmax 的 LSE 在两项间抵消）
                    grad += coef * (eye[w] - eye[l])
                self.logits[p.prompt_id] = lg + np.clip(
                    lr * grad, -self.tc.grad_clip, self.tc.grad_clip
                )
        self._fitted = True
        return self
