"""文档解析：一个实现类对应一类文件格式。

Loader 只负责「把文件变成文本」，不做标题层级识别：
层级解析属于切片策略的差异（fixed 不需要、hierarchical 需要），
放到 splitter 里才能让两组实验的差异收敛在单一变量上。

文件格式本身不参与这一步：doc_id 由内容哈希得到，
同一份内容换目录或改名不会重复入库，内容变化才视为新文档。
"""

from __future__ import annotations

import hashlib
import re
from io import BytesIO
from pathlib import Path

import pymupdf
from charset_normalizer import from_bytes
from docx import Document

from rag_core.core.contracts import LoadedDoc, LoadedPage
from rag_core.core.logging import get_logger
from rag_core.registry import register

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def content_doc_id(raw: bytes) -> str:
    """按文件内容生成文档 ID，取前 16 位十六进制。

    用内容而非路径：同一份内容换目录或改名不会重复入库，
    内容不变（如复制到别处）时不会被当成新文档，内容变化才视为新文档。
    """
    return hashlib.sha256(raw).hexdigest()[:16]


def decode_bytes(raw: bytes) -> str:
    """自动识别编码，兜底中文常见的 GBK / GB18030。"""
    best = from_bytes(raw).best()
    if best is not None:
        return str(best)
    return raw.decode("utf-8", errors="replace")


def markdown_title(text: str) -> str:
    """取文档中第一个一级标题作为文档标题，没有则空串。"""
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m and len(m.group(1)) == 1:
            return m.group(2).strip()
    return ""


@register("loader", "md")
class MarkdownLoader:
    """Markdown 解析。语料整理阶段统一转为 md，因此这是主路径。"""

    extensions = {".md", ".markdown"}

    def load(self, path: Path) -> LoadedDoc:
        raw = path.read_bytes()
        text = decode_bytes(raw)
        return LoadedDoc(
            doc_id=content_doc_id(raw),
            source=str(path),
            file_type="md",
            pages=[LoadedPage(text=text)],
            doc_title=markdown_title(text) or path.stem,
        )


@register("loader", "pdf")
class PdfLoader:
    """PDF 解析，按页拆分以保留页码用于溯源。"""

    extensions = {".pdf"}

    def load(self, path: Path) -> LoadedDoc:
        raw = path.read_bytes()
        doc = pymupdf.open(stream=raw, filetype="pdf")
        pages: list[LoadedPage] = []
        try:
            for i, page in enumerate(doc):
                pages.append(LoadedPage(text=page.get_text("text"), page=i + 1))
        finally:
            doc.close()
        return LoadedDoc(
            doc_id=content_doc_id(raw),
            source=str(path),
            file_type="pdf",
            pages=pages,
            doc_title=path.stem,
        )


@register("loader", "docx")
class DocxLoader:
    """DOCX 解析。只取段落文本，表格内容由 python-docx 另行处理。"""

    extensions = {".docx"}

    def load(self, path: Path) -> LoadedDoc:
        raw = path.read_bytes()
        document = Document(BytesIO(raw))
        parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))
        return LoadedDoc(
            doc_id=content_doc_id(raw),
            source=str(path),
            file_type="docx",
            pages=[LoadedPage(text="\n\n".join(parts))],
            doc_title=path.stem,
        )


@register("loader", "txt")
class TxtLoader:
    """纯文本解析。"""

    extensions = {".txt"}

    def load(self, path: Path) -> LoadedDoc:
        raw = path.read_bytes()
        return LoadedDoc(
            doc_id=content_doc_id(raw),
            source=str(path),
            file_type="txt",
            pages=[LoadedPage(text=decode_bytes(raw))],
            doc_title=path.stem,
        )


def all_loaders() -> list:
    """实例化全部已注册的 loader，供装配层构造入库流水线。

    延迟到函数内部读取注册表，这样调用方可以在调用前注册自有实现。
    """
    from rag_core.registry import available, create

    return [create("loader", name) for name in available("loader")]
