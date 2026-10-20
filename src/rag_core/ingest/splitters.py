"""文本切分：fixed 与 hierarchical 两种策略。

两种策略是 E1/E2 对照实验的自变量，因此它们的差异必须只有一处：
是否按标题层级组织 chunk 并把标题链注入检索文本。
除此之外（递归切分的实现、overlap 语义）两者共用同一套底层函数，
否则实验差异无法归因到「切片策略」这一个变量上。

切分器自研而非引入框架：框架的文本切分器只提供递归回退分隔符，
而本项目真正需要的是「标题链注入」，这部分无论如何都要自己写。
"""

from __future__ import annotations

import re
from typing import Any

from rag_core.core.contracts import Chunk, LoadedDoc
from rag_core.registry import register

# 中文优先的递归分隔符。顺序即优先级：
# 先按段落断，再按句断，最后按标点与空格断。
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _split_keep(text: str, sep: str) -> list[str]:
    """按分隔符切分并保留分隔符，保证拼接后能还原原文。"""
    if sep == "":
        return list(text)
    parts = text.split(sep)
    return [part + sep for part in parts[:-1]] + [parts[-1]]


def _hard_split(text: str, size: int) -> list[str]:
    """分隔符用尽仍超限时按字符硬切，保证不超过 size。"""
    return [text[i : i + size] for i in range(0, len(text), size)]


def _split_recursive(text: str, size: int, sep_idx: int = 0) -> list[str]:
    """逐级回退分隔符，产出长度不超过 size 的单元。"""
    if len(text) <= size:
        return [text]
    if sep_idx >= len(_SEPARATORS):
        return _hard_split(text, size)
    sep = _SEPARATORS[sep_idx]
    units: list[str] = []
    for part in _split_keep(text, sep):
        if not part:
            continue
        if len(part) > size:
            units.extend(_split_recursive(part, size, sep_idx + 1))
        else:
            units.append(part)
    return units


def _merge_units(units: list[str], size: int, overlap: int) -> list[str]:
    """贪心合并单元，相邻块之间保留 overlap 个字符的重叠。"""
    chunks: list[str] = []
    cur = ""
    for unit in units:
        if len(unit) > size:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(unit)
            continue
        if len(cur) + len(unit) <= size:
            cur += unit
            continue
        if cur:
            chunks.append(cur)
            cur = (cur[-overlap:] if 0 < overlap < len(cur) else "") + unit
        else:
            cur = unit
    if cur:
        chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]


def _path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(title for _, title in stack)


def _parse_markdown(text: str) -> list[tuple[str, str, str]]:
    """把 Markdown 解析为 (标题链, 块类型, 文本) 三元组。

    块类型区分 table / code / text：表格与代码块按整体保留，
    不按行切断——切断后的表格行失去表头，检索到也是无效证据。
    """
    lines = text.splitlines()
    blocks: list[tuple[str, str, str]] = []
    stack: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            # 标题自身先记为一块：若其后没有正文，这一行仍是有效信息
            blocks.append((_path(stack[:-1]), "heading", title))
            i += 1
            continue

        if _FENCE_RE.match(stripped):
            start = i
            i += 1
            while i < len(lines) and not _FENCE_RE.match(lines[i].strip()):
                i += 1
            if i < len(lines):
                i += 1
            blocks.append((_path(stack), "code", "\n".join(lines[start:i])))
            continue

        if stripped.startswith("|"):
            start = i
            while i < len(lines) and lines[i].strip().startswith("|"):
                i += 1
            blocks.append((_path(stack), "table", "\n".join(lines[start:i])))
            continue

        # 普通段落：连续无空行、非标题非代码非表格的行视为一段
        start = i
        para: list[str] = []
        while i < len(lines):
            cur = lines[i].strip()
            if not cur or _HEADING_RE.match(cur) or _FENCE_RE.match(cur) or cur.startswith("|"):
                break
            para.append(lines[i].rstrip())
            i += 1
        if para:
            blocks.append((_path(stack), "text", "\n".join(para)))
    return blocks


def _attach_headings(blocks: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """把标题块并入其后的正文块：标题原文字成为正文首行。

    标题行不单独成块——那样会产生「只有几个字」的碎片 chunk。
    并入正文后每个 chunk 自包含，脱离上下文也能读懂。
    """
    out: list[tuple[str, str, str]] = []
    pending: tuple[str, str] | None = None
    for path, kind, text in blocks:
        if kind == "heading":
            if pending is not None:
                # 上一个标题没有正文跟随，只能单独成块
                out.append((_join(pending[0], pending[1]), "text", pending[1]))
            pending = (path, text)
            continue
        if pending is not None:
            out.append((_join(pending[0], pending[1]), "text", f"{pending[1]}\n\n{text}"))
            pending = None
        else:
            out.append((path, kind, text))
    if pending is not None:
        out.append((_join(pending[0], pending[1]), "text", pending[1]))
    return out


def _join(parent: str, title: str) -> str:
    return f"{parent} > {title}" if parent else title


def _shared_prefix(a: str, b: str) -> str:
    """取两条标题链的最长公共前缀。"""
    pa = a.split(" > ") if a else []
    pb = b.split(" > ") if b else []
    n = 0
    while n < min(len(pa), len(pb)) and pa[n] == pb[n]:
        n += 1
    return " > ".join(pa[:n])


def _merge_consecutive(blocks: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """合并标题链相同且类型为 text 的相邻块。

    Markdown 中一个列表与其前的引导句之间常有空行，
    _parse_markdown 会把空行看作块边界，导致同标题下的两段被切成两个 chunk。
    """
    out: list[tuple[str, str, str]] = []
    for path, kind, text in blocks:
        if out and out[-1][0] == path and out[-1][1] == "text" and kind == "text":
            out[-1] = (path, "text", out[-1][2] + "\n\n" + text)
        else:
            out.append((path, kind, text))
    return out


def _pack_by_path(blocks: list[tuple[str, str, str]], size: int) -> list[tuple[str, str, str]]:
    """自底向上合并相邻块，直到接近 size 上限。

    只按小节切会让每个小节单独成 chunk，实测平均不足 200 字，
    语义碎片化：一条赔付规则可能与它的条款编号被切到两个 chunk。
    这里把共享父标题的相邻小节合并，合并后的标题链退化为公共父路径——
    子标题文本已经并入正文，信息不会丢。

    公共前缀为空时不合并：那说明两块分属不同一级章节，合在一起语义无关。
    """
    out: list[tuple[str, str, str]] = []
    for path, kind, text in blocks:
        if kind in ("table", "code"):
            out.append((path, kind, text))
            continue
        if out and out[-1][1] == "text":
            prev_path, prev_text = out[-1][0], out[-1][2]
            # 已经攒到 size 的 60% 就不再往里并：
            # 继续并下去会把下一章的内容也拉进来，标题链被迫退化为公共父路径，
            # 定位精度反而变差。半满即止，宁可多一个 chunk 也不要路径失真。
            if len(prev_text) < size * 0.6 and len(prev_text) + 2 + len(text) <= size:
                shared = _shared_prefix(prev_path, path)
                if shared:
                    out[-1] = (shared, "text", prev_text + "\n\n" + text)
                    continue
        out.append((path, "text", text))
    return out


def _make_chunk(
    doc: LoadedDoc,
    page: "LoadedPage | None",
    piece: str,
    heading_path: str,
    chunk_index: int,
) -> Chunk:
    return Chunk(
        text=piece,
        embed_text=piece,
        doc_id=doc.doc_id,
        source=doc.source,
        doc_title=doc.doc_title,
        heading_path=heading_path,
        file_type=doc.file_type,
        page=page.page if page is not None else None,
        chunk_index=chunk_index,
    )


@register("splitter", "fixed")
class FixedSplitter:
    """baseline 切片：按分隔符递归切，不注入任何标题上下文。

    inject_heading 参数在 fixed 策略下被刻意忽略——
    E1 baseline 的口径是「干净的固定窗口」，加了前缀就不再是 baseline。
    """

    def __init__(
        self,
        size: int = 500,
        overlap: int = 50,
        inject_heading: bool = True,  # noqa: ARG002 - 刻意忽略，保持 baseline 干净
        options: dict[str, Any] | None = None,
    ):
        self.size = size
        self.overlap = overlap
        self.options = options or {}

    def split(self, doc: LoadedDoc) -> list[Chunk]:
        from rag_core.core.contracts import LoadedPage

        chunks: list[Chunk] = []
        idx = 0
        for page in doc.pages:
            text = page.text
            if not text.strip():
                continue
            pieces = _merge_units(_split_recursive(text, self.size), self.size, self.overlap)
            for piece in pieces:
                chunks.append(
                    _make_chunk(doc, page if isinstance(page, LoadedPage) else None, piece, "", idx)
                )
                idx += 1
        return chunks


@register("splitter", "hierarchical")
class HierarchicalSplitter:
    """按 Markdown 标题层级切分，把标题链作为前缀注入 embed_text。"""

    def __init__(
        self,
        size: int = 500,
        overlap: int = 50,
        inject_heading: bool = True,
        options: dict[str, Any] | None = None,
    ):
        self.size = size
        self.overlap = overlap
        self.inject_heading = inject_heading
        self.options = options or {}

    def split(self, doc: LoadedDoc) -> list[Chunk]:
        from rag_core.core.contracts import LoadedPage

        chunks: list[Chunk] = []
        idx = 0
        for page in doc.pages:
            text = page.text
            if not text.strip():
                continue
            page_obj = page if isinstance(page, LoadedPage) else None
            blocks = _pack_by_path(
                _merge_consecutive(_attach_headings(_parse_markdown(text))), self.size
            )
            for path, _kind, body in blocks:
                body = body.strip()
                if not body:
                    continue
                # 表格与代码块按整体保留，文本块按 size 拆分
                if _kind in ("table", "code") or len(body) <= self.size:
                    pieces = [body]
                else:
                    pieces = _merge_units(_split_recursive(body, self.size), self.size, self.overlap)
                for piece in pieces:
                    embed = (path + "\n" + piece) if (self.inject_heading and path) else piece
                    chunks.append(
                        Chunk(
                            text=piece,
                            embed_text=embed,
                            doc_id=doc.doc_id,
                            source=doc.source,
                            doc_title=doc.doc_title,
                            heading_path=path,
                            file_type=doc.file_type,
                            page=page_obj.page if page_obj else None,
                            chunk_index=idx,
                        )
                    )
                    idx += 1
        return chunks
