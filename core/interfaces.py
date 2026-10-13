# -*- coding: utf-8 -*-
"""对齐器接口契约。Author: 晨星 (CJX0712)"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import MetricBundle, PreferenceDataset


@runtime_checkable
class Aligner(Protocol):
    """所有对齐方法的统一契约：fit -> predict_logits。"""

    name: str

    def fit(self, dataset: PreferenceDataset) -> "Aligner":
        """在偏好数据上训练；rewards 仅评测用，绝不可被方法读取。"""
        ...

    def predict_logits(self, prompt_id: int) -> np.ndarray:
        """返回 (K,) logits，softmax 得策略。"""
        ...

    def available(self) -> bool:
        """依赖是否就绪（纯 numpy 默认 True）。"""
        ...

    def evaluate(self, dataset: PreferenceDataset) -> MetricBundle:
        """在评测集上计算指标（只读 rewards）。"""
        ...
