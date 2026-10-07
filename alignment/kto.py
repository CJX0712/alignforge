# -*- coding: utf-8 -*-
"""KTO (Kahneman-Tversky Optimization, Ethayarajh et al. 2024)。Author: 晨星 (CJX0712)

无需成对偏好，只用单样本的「可取 / 不可取」信号（reward 中位数二分）。
用 wellness 函数 σ(β·z − ½) 推高可取样本、压低不可取样本。
"""

from __future__ import annotations

import numpy as np

from core.config import TrainConfig

from .base import BaseAligner


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _logsumexp(x: np.ndarray) -> float:
    xm = x.max()
    return float(xm + np.log(np.exp(x - xm).sum()))


class KTOAligner(BaseAligner):
    name = "KTO"

    def __init__(self, tc: TrainConfig | None = None) -> None:
        super().__init__(tc)

    def fit(self, dataset):
        self._init_logits(dataset)
        lr = self.tc.lr
        beta = self.tc.beta_kto
        margin = self.tc.margin_kto
        lam_d = self.tc.lambda_desirable
        lam_u = self.tc.lambda_undesirable
        K = dataset.k_candidates
        eye = np.eye(K)
        labels = dataset.single_labels()

        for _ in range(self.tc.n_epochs):
            for pid, cand, desirable in labels:
                lg = self.logits[pid]
                pi = self.softmax(lg)
                lse = _logsumexp(lg)
                z = lg[cand] - lse + np.log(K)  # log(π_θ/π_ref), ref 为 uniform
                if desirable:
                    s = _sigmoid(beta * (z - margin))
                    coef = lam_d * beta * s * (1.0 - s)
                    grad = coef * (eye[cand] - pi)
                else:
                    s = _sigmoid(beta * (-z - margin))
                    coef = -lam_u * beta * s * (1.0 - s)
                    grad = coef * (eye[cand] - pi)
                self.logits[pid] = lg + lr * grad
        self._fitted = True
        return self
