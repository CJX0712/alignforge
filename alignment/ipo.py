# -*- coding: utf-8 -*-
"""IPO (Identity Preference Optimization, Azar et al. 2023)。Author: 晨星 (CJX0712)

对偏好噪声更鲁棒：用 (margin - ½β)² 替代 DPO 的对数似然，削弱对 β 的敏感性。
"""

from __future__ import annotations

import numpy as np

from core.config import TrainConfig

from .base import BaseAligner


class IPOAligner(BaseAligner):
    name = "IPO"

    def __init__(self, tc: TrainConfig | None = None) -> None:
        super().__init__(tc)

    def fit(self, dataset):
        self._init_logits(dataset)
        lr = self.tc.lr
        beta = self.tc.beta_ipo
        K = dataset.k_candidates
        eye = np.eye(K)
        for _ in range(self.tc.n_epochs):
            for p in dataset.prompts:
                lg = self.logits[p.prompt_id]
                grad = np.zeros(K)
                for w, l in p.pairs:
                    m = lg[w] - lg[l]
                    # 步长按 β 归一，固定点不变（m=½β）；否则 β 同时放大步长致发散
                    coef = -2.0 * (m - 0.5 * beta) / max(abs(beta), 1e-6)
                    # dm/dθ = e_w - e_l
                    grad += coef * (eye[w] - eye[l])
                self.logits[p.prompt_id] = lg + np.clip(
                    lr * grad, -self.tc.grad_clip, self.tc.grad_clip
                )
        self._fitted = True
        return self
