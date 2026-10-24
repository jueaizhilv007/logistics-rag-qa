"""入库流水线：解析 → 判重 → 切分 → 向量化 → 写入。

幂等基于 doc_id（内容哈希），重复灌入会被 has_doc 拦下。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from rag_core.core.contracts import Chunk, IngestResult, LoadedDoc
from rag_core.core.logging import get_logger
from rag_core.core.protocols import (
    BaseEmbedding,
    BaseLoader,
    BaseSplitter,
    BaseVectorStore,
)


class IngestPipeline:
    """单文件入库流水线。

    装载一批 loader（按扩展名路由），串联 splitter、embedding、store。
    不主动 optimize：optimize 在大量写入后才有效，单文档后调用是浪费。
    一次多文件灌入后由调用方决定何时 optimize。
    """

    def __init__(
        self,
        loaders: Iterable[BaseLoader],
        splitter: BaseSplitter,
        embedding: BaseEmbedding,
        store: BaseVectorStore,
    ):
        self._loaders: dict[str, BaseLoader] = {
            ext: loader for loader in loaders for ext in loader.extensions
        }
        self.splitter = splitter
        self.embedding = embedding
        self.store = store
        self.log = get_logger("rag_core.ingest")

    def supported_extensions(self) -> set[str]:
        return set(self._loaders)

    def prepare(self, path: Path) -> tuple[IngestResult, list[Chunk], LoadedDoc | None]:
        """解析 + 判重 + 切分，但不向量化、不写入。

        dry-run 场景需要观察 chunk 内容与 heading_path 注入情况时使用。
        返回 (IngestResult, chunks, doc)：doc 为 None 表示准备阶段就出错。
        """
        loader = self._loaders.get(path.suffix.lower())
        if loader is None:
            return (
                # 不支持的文件类型 → 跳过
                IngestResult(source=str(path), status="skipped", reason=f"unsupported {path.suffix}"),
                [],
                None,
            )
        try:
            doc = loader.load(path)
        except Exception as exc:  # noqa: BLE001
            return (
                # 解析抛异常 → 失败
                IngestResult(source=str(path), status="error", reason=f"parse: {exc}"),
                [],
                None,
            )

        if self.store.has_doc(doc.doc_id):
            return (
                # 已索引 → 跳过（幂等去重）
                IngestResult(source=str(path), status="skipped", doc_id=doc.doc_id, reason="already indexed"),
                [],
                doc,
            )

        # 切分 + 过滤空块【c.text.strip()】
        chunks = [c for c in self.splitter.split(doc) if c.text.strip()]
        if not chunks:
            return (
                # 切分后为空（空文档） → 跳过
                IngestResult(source=str(path), status="skipped", doc_id=doc.doc_id, reason="empty after split"),
                [],
                doc,
            )
        return (
            # 切分后非空 → 入库
            IngestResult(source=str(path), status="indexed", doc_id=doc.doc_id, chunks=len(chunks)),
            chunks,
            doc,
        )

    def run(self, path: Path) -> IngestResult:
        result, chunks, _doc = self.prepare(path)
        if result.status != "indexed":
            return result

        try:
            # 向量化
            vectors = self.embedding.embed_documents([c.embed_text for c in chunks])
            # 写入向量存储
            self.store.add(chunks, vectors)
        except Exception as exc:  # noqa: BLE001
            return IngestResult(
                source=str(path),
                status="error",
                doc_id=result.doc_id,
                chunks=result.chunks,
                reason=f"embed/write: {exc}",
            )
        return result

