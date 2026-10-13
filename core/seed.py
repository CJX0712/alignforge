# -*- coding: utf-8 -*-
"""全局确定性入口。Author: 晨星 (CJX0712)

所有随机性必须经过 set_all() 一处设定；numpy / random 一次性设齐。
demo 两次运行同 seed 必须逐位一致（max|Δ|=0）。
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

_SEED: dict = {"seed": 42}


def set_all(seed: int) -> int:
    """设定全局确定性种子。返回实际生效的 seed（便于断言）。"""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"[E300] seed 必须非负整数，得到 {seed!r}")
    seed = int(seed) & 0x7FFFFFFF
    np.random.seed(seed)
    random.seed(seed)
    _SEED["seed"] = seed
    return seed


def get_seed() -> int:
    return _SEED["seed"]


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """返回独立 Generator（不污染全局状态）。"""
    s = seed if seed is not None else _SEED["seed"]
    return np.random.default_rng(s)
