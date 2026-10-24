"""行为契约：定义各层的可替换接口。

这里使用 Protocol（结构化子类型）而不是 ABC：
实现类不需要继承本文件的任何基类，只要方法签名匹配即可被接受。
第三方可以在完全不 import rag_core 的前提下提供自己的实现，
这是「哪里需要接哪里」这一目标的落点。

from __future__ import annotations：让整个文件的所有类型注解“只写不跑”：类型检查器看得懂、运行时零成本、前向引用随便写、新语法在旧 Python 上不炸。
对契约层这种“类型定义密集”的文件来说，是标配的第一行。Python 3.14 开始支持。

@runtime_checkable：确保在运行时检查协议是否被正确实现。
只检查“方法名存在”， 不检查参数签名、返回类型
这在类型检查器中是透明的，不会影响代码的执行。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from rag_core.core.contracts import (
    Chunk,
    LoadedDoc,
    RetrievedChunk,
    TextRecord,
    VectorHit,
)


@runtime_checkable
class BaseLoader(Protocol):
    """文档解析：一个实现对应一类文件格式。"""
    extensions: set[str]

    def load(self, path: Path) -> LoadedDoc: ...


@runtime_checkable
class BaseSplitter(Protocol):
    """文本切分：输入完整文档，输出 Chunk 列表。"""
    def split(self, doc: LoadedDoc) -> list[Chunk]: ...


@runtime_checkable
class BaseEmbedding(Protocol):
    """文本向量化。"""
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    @property
    def dimension(self) -> int: ...


@runtime_checkable
class BaseVectorStore(Protocol):
    """向量存储。"""
    def ensure_collection(self, dimension: int) -> None:
        """按维度建立集合，已存在则复用。延迟到首次写入时调用，
        因此建表所需的真实维度由首个 batch 决定，无需预先配置。
        """
        ...

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def has_doc(self, doc_id: str) -> bool: ...

    def delete_doc(self, doc_id: str) -> None: ...

    def search(self, query_vector: list[float], k: int) -> list[VectorHit]: ...

    def iter_text_records(self) -> Iterator[TextRecord]:
        """流式产出全部文本记录。用迭代器而非列表返回，
        让实现方能分批抓取，避免一次性把整表读进内存。
        """
        ...

    def count(self) -> int: ...

    def optimize(self) -> None: ...

    def clear(self) -> None:
        """清空整个 collection。

        对照实验切换切片策略后必须重建索引（E1→E2），逐条删除代价高。
        drop_table 是合理语义。
        """
        ...


@runtime_checkable
class BaseRetriever(Protocol):
    """检索器。"""
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]: ...


@runtime_checkable
class BaseReranker(Protocol):
    """重排器：对召回候选做精排，按新顺序返回前 top_k 个。"""
    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class BaseLLM(Protocol):
    """LLM 客户端。"""
    def chat(self, messages: list[dict], **kwargs) -> str: ...

    def stream_chat(self, messages: list[dict], **kwargs) -> Iterator[str]: ...

    @property
    def model_name(self) -> str: ...
