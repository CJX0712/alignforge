# -*- coding: utf-8 -*-
"""BC / SFT 弱基线：监督拟合 winner 边际，丢弃 pairwise 结构。Author: 晨星 (CJX0712)"""

from __future__ import annotations

import numpy as np

from core.config import TrainConfig

from .base import BaseAligner


class BCAligner(BaseAligner):
    name = "BC"

    def __init__(self, tc: TrainConfig | None = None) -> None:
        super().__init__(tc)

    def fit(self, dataset):
        self._init_logits(dataset)
        lr = self.tc.lr
        for _ in range(self.tc.n_epochs):
            for p in dataset.prompts:
                lg = self.logits[p.prompt_id]
                pi = self.softmax(lg)
                grad = np.zeros_like(lg)
                for w, _l in p.pairs:
                    grad[w] += 1.0 - pi[w]  # 仅拟合 winner，结构被丢弃
                self.logits[p.prompt_id] = lg + lr * grad
        self._fitted = True
        return self
