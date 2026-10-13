# AlignForge · 世界顶级 LLM 对齐后训练系统（SFT→DPO→GRPO + LoRA）
# 纯 NumPy 零依赖 · 确定性可复现 · 不变量可验证 · 作者：晨星
from . import autodiff as ad
from .model import Config, TinyLM
from .align import Adam, sft, dpo, grpo, reward_fn
from . import data

__version__ = "1.0.0"
__author__ = "晨星 (MorningStar)"
