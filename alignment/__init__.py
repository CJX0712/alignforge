# -*- coding: utf-8 -*-
"""alignment 层：对齐方法集合。Author: 晨星 (CJX0712)"""

from .alignfuse import AlignFuseAligner
from .base import BaseAligner
from .bc import BCAligner
from .dpo import DPOAligner
from .ipo import IPOAligner
from .kto import KTOAligner

__all__ = [
    "BaseAligner",
    "BCAligner",
    "DPOAligner",
    "IPOAligner",
    "KTOAligner",
    "AlignFuseAligner",
    "build_registry",
]


def build_registry() -> dict:
    """方法注册表。AlignFuse 为旗舰，置于最后。"""
    return {
        "BC": BCAligner,
        "DPO": DPOAligner,
        "IPO": IPOAligner,
        "KTO": KTOAligner,
        "AlignFuse": AlignFuseAligner,
    }
