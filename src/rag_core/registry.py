"""实现注册表。

内核本身不依赖本模块：只有在配置驱动的装配路径（RagCore.from_config）
才需要用字符串选取实现。走显式注入路径时完全不触碰这里，
因此宿主项目即使自带一套依赖注入体系也不会与本模块冲突。
"""

from __future__ import annotations

from typing import Literal

# 注册类别。reranker 与 retriever 是新增项：
# 原实现没有独立的检索层与重排层，
# 而它们正是对照实验 E3 / E4 要验证的变量，必须留出身位。
Kind = Literal[
    "embedding",
    "llm",
    "vector_store",
    "splitter",
    "loader",
    "retriever",
    "reranker",
]


class RegistryError(Exception):
    """注册表所有错误的基类，调用方可统一拦截。"""


class UnknownKindError(RegistryError):
    """kind 不在注册表中。"""


class UnknownImplError(RegistryError):
    """impl 不在对应 kind 下。"""


class DuplicateRegistrationError(RegistryError):
    """同名实现重复注册。"""


# 结构：{kind: {impl_name: cls}}，首次 register 时由 setdefault 创建分组
_REGISTRY: dict[str, dict[str, type]] = {}


def register(kind: Kind, impl: str):
    """装饰器：把实现类注册到指定类别下，并返回原类，不影响类本身的定义。"""
    def deco(cls):
        bucket = _REGISTRY.setdefault(kind, {})
        if impl in bucket:
            raise DuplicateRegistrationError(
                f"重复注册 {kind}/{impl}："
                f"已由 {bucket[impl].__module__}.{bucket[impl].__name__} 占用，"
                f"与 {cls.__module__}.{cls.__name__} 冲突"
            )
        bucket[impl] = cls
        return cls
    return deco


def create(kind: Kind, impl: str, **kwargs):
    """按 kind 加 impl 构造实例，参数透传给实现类的构造函数。"""
    bucket = _REGISTRY.get(kind)
    if bucket is None:
        raise UnknownKindError(
            f"未注册的类别 `{kind}`，已注册类别：{sorted(_REGISTRY)}。"
            f"若这是自定义实现，请确认已 import 该实现模块以触发注册"
        )
    if impl not in bucket:
        raise UnknownImplError(
            f"未注册的 {kind} 实现 `{impl}`，可用实现：{sorted(bucket)}"
        )
    return bucket[impl](**kwargs)


def available(kind: Kind | None = None) -> list[str] | dict[str, list[str]]:
    """列出可用实现。不传 kind 时返回全部类别的清单。"""
    if kind is not None:
        return sorted(_REGISTRY.get(kind, {}))
    return {k: sorted(v) for k, v in sorted(_REGISTRY.items())}


def clear() -> None:
    """清空注册表。供宿主项目在多套配置间切换或做环境隔离时使用。"""
    _REGISTRY.clear()
