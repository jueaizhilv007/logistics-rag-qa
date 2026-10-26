"""生成层：提示词构造与问答流水线。

与检索层不同，本层没有对应的 Protocol 与注册表项：
生成流程只有「强制引用 + 未覆盖拒答」这一套，
把它做成可替换接口反而会让「引用是硬约束」这件事变成可选。
真要换的是 prompt，而 prompt 已经是模块里的模板常量。
"""

from rag_core.generate.pipeline import (
    GeneratePipeline,
    count_tokens,
    parse_cited_indices,
    select_within_budget,
)
from rag_core.generate.prompt import build_context, build_messages

__all__ = [
    "GeneratePipeline",
    "build_context",
    "build_messages",
    "count_tokens",
    "parse_cited_indices",
    "select_within_budget",
]
