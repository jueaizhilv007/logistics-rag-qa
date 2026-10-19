"""统一日志配置。

内核本身不主动调用本文件：是否输出日志由调用方决定。
作为独立应用运行时由入口脚本调用 setup_logging，
作为库被嵌入宿主项目时保持沉默，不污染宿主日志配置。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO, log_dir: Path | None = None) -> None:
    """配置根 logger。

    控制台输出始终启用；传入 log_dir 时额外输出到文件。
    force=True 保证重复调用时重建 handler，不会重复打印。
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "rag_core.log", encoding="utf-8"))

    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。保留统一入口便于后续替换为结构化日志实现。"""
    return logging.getLogger(name)
