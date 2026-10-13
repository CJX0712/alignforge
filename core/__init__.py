# -*- coding: utf-8 -*-
"""AlignForge core 层：类型 / 错误 / 配置 / 种子 / 接口契约。

Author: 晨星 (CJX0712)
"""

from .config import Config
from .errors import (
    AlignerError,
    AlignForgeError,
    ConfigError,
    DataLeakError,
    DGPError,
    NotFittedError,
    NumericsError,
)
from .interfaces import Aligner
from .seed import get_rng, set_all
from .types import AlignResult, MetricBundle, PreferenceDataset, PreferencePair, PromptData

__all__ = [
    "PreferencePair",
    "PromptData",
    "PreferenceDataset",
    "AlignResult",
    "MetricBundle",
    "AlignForgeError",
    "DGPError",
    "AlignerError",
    "NotFittedError",
    "ConfigError",
    "DataLeakError",
    "NumericsError",
    "Config",
    "set_all",
    "get_rng",
    "Aligner",
]
