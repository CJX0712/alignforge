# -*- coding: utf-8 -*-
"""对齐器基类与公共工具。Author: 晨星 (CJX0712)"""

from __future__ import annotations

import numpy as np

from core.config import TrainConfig
from core.errors import NotFittedError

from .metrics import compute_metrics, softmax


class BaseAligner:
    name = "base"

    def __init__(self, tc: TrainConfig | None = None) -> None:
        self.tc = tc or TrainConfig()
        self.logits: dict = {}
        self._fitted = False

    def _init_logits(self, dataset) -> None:
        K = dataset.k_candidates
        self.logits = {p.prompt_id: np_zeros(K) for p in dataset.prompts}

    def fit(self, dataset):
        raise NotImplementedError

    def predict_logits(self, prompt_id: int) -> "np.ndarray":
        if not self._fitted:
            raise NotFittedError(f"{self.name} 未 fit 即被调用")
        return self.logits[prompt_id]

    def available(self) -> bool:
        return True

    def evaluate(self, dataset):
        if not self._fitted:
            raise NotFittedError(self.name)
        return compute_metrics(self.name, self.logits, dataset, dataset.seed)

    @staticmethod
    def softmax(x) -> "np.ndarray":
        return softmax(x)


def np_zeros(n: int) -> "np.ndarray":
    return np.zeros(n, dtype=np.float64)
