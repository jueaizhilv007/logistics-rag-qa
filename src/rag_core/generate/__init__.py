"""生成层：提示词构造与问答流水线。

与检索层不同，本层没有对应的 Protocol 与注册表项：
生成流程只有「强制引用 + 未覆盖拒答」这一套，
把它做成可替换接口反而会让「引用是硬约束」这件事变成可选。
真要换的是 prompt，而 prompt 已经是模块里的模板常量。

引用编号的解析函数刻意不导出：它只做正则提取，不做越界收敛，
单独使用会拿到模型编造的编号。要解析就走 `GeneratePipeline`，
那条路径一定会拿实际给出的资料编号做收敛。
"""

from rag_core.generate.pipeline import (
    GeneratePipeline,
    count_tokens,
    select_within_budget,
)
from rag_core.generate.prompt import build_context, build_messages

__all__ = [
    "GeneratePipeline",
    "build_context",
    "build_messages",
    "count_tokens",
    "select_within_budget",
]
