"""内置实现的插件入口。

内核装配采用注册表，而注册发生在「实现模块被 import 时」。
因此配置驱动路径必须先 import 全部内置实现，否则会报 UnknownKindError。

单独收敛到一个函数的原因：入口可能有多个（CLI、HTTP 服务、评测脚本），
各自写一遍 import 迟早会漏，漏了的表现是运行时报错而非启动报错。

本模块不在 rag_core/__init__.py 中导入：
显式注入路径不需要任何内置实现，提前导入会把 lancedb、pymupdf 等
依赖强加给那些只想用自己的 embedding 与存储的宿主项目。
"""

from __future__ import annotations


def load_builtin_plugins() -> None:
    """导入全部内置实现模块，触发注册。重复调用无副作用。"""
    import rag_core.embeddings  # noqa: F401
    import rag_core.ingest.loaders  # noqa: F401
    import rag_core.ingest.splitters  # noqa: F401
    import rag_core.llms  # noqa: F401
    import rag_core.stores  # noqa: F401
    # rerankers.py 由 S3 提供，届时在此补一行 import 即可，
    # 未实现期间不导入，否则本函数会抛 ImportError。
