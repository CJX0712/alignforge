# -*- coding: utf-8 -*-
"""错误码体系 E100~E500。Author: 晨星 (CJX0712)"""

from __future__ import annotations


class AlignForgeError(Exception):
    """基类。所有 AlignForge 异常携带稳定 code（用于 CI 门禁与日志路由）。"""

    code = "E000"

    def __init__(self, message: str = "", code: str | None = None) -> None:
        self.code = code or self.__class__.code
        super().__init__(f"[{self.code}] {message}")


class DGPError(AlignForgeError):
    """数据生成 / DGP 构造错误。"""

    code = "E100"


class AlignerError(AlignForgeError):
    """对齐器内部错误。"""

    code = "E200"


class NotFittedError(AlignerError):
    """对齐器未 fit 即被调用。"""

    code = "E210"


class ConfigError(AlignForgeError):
    """配置校验失败。"""

    code = "E300"


class DataLeakError(AlignForgeError):
    """检测到标签/奖励泄漏进训练。"""

    code = "E400"


class NumericsError(AlignForgeError):
    """数值不稳定 / 不收敛。"""

    code = "E500"
