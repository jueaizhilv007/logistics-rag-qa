"""生成流水线：检索 → 拼装 → 生成 → 解析引用 → 判定拒答。

### 拒答有两个来源，且都可能带着非空证据

1. 检索为空：一条资料都没有，直接拒答，不调 LLM
2. 检索有结果但模型没引用：强制引用是硬约束，
   一个编号都没引，说明它认为资料不足以作答，或者正在凭自己的知识回答——
   两种情况下答案都不可信，同样按拒答处理

第 2 条是这套设计里最硬的一处：宁可多拒答，也不放过一个没有出处的答案。
`refused=True` 时 `retrieved` 可以非空，界面要看 `retrieved` 决定渲染证据面板，
展示「我找到了什么、但不足以回答」，这本身就是透明度。

### 越界编号要丢，但丢的动作要留痕

模型偶尔会引用 `[7]` 而实际只给了 4 条资料。
这类编号不能进 `cited_indices`——它是可追溯率的分母，
留着会算出「引用了 5 条、3 条可追溯」这种说不清的指标。
但直接静默丢弃会掩盖模型编造编号这件事，所以丢的同时记 warning。
"""

from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Optional

import tiktoken

from rag_core.core.contracts import QAResult, RetrievedChunk
from rag_core.core.logging import get_logger
from rag_core.core.protocols import BaseLLM, BaseRetriever
from rag_core.generate.prompt import build_context, build_messages

CITATION_RE = re.compile(r"\[(\d+)\]")


@lru_cache(maxsize=1)
def _encoder():
    """tiktoken 首次取编码要联网下载，约 10 秒。

    缓存住，一个进程只付一次。这个代价只在冷启动出现，
    不缓存的话每次问答都要等十秒。
    """
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def select_within_budget(
    hits: list[RetrievedChunk],
    max_tokens: int,
) -> list[RetrievedChunk]:
    """按排名顺序累加片段，超出 token 预算即停。

    截断放在拼装之前而不是之后：先拼成一大段再从中间切，
    会切在半句话上，模型拿到的是一个断掉的片段，
    而这恰恰是它开始猜测的诱因。

    `if kept` 保证至少进一条。全部片段都超预算时若返回空列表，
    后面的拒答判据会把它判成「检索为空」，
    而真实情况是「检索到了但太长」——这是两种不同的失败，不该混为一谈。
    """
    if max_tokens <= 0:
        return list(hits)
    kept: list[RetrievedChunk] = []
    used = 0
    for hit in hits:
        cost = count_tokens(hit.chunk.text)
        if kept and used + cost > max_tokens:
            break
        kept.append(hit)
        used += cost
    return kept


def parse_cited_indices(answer: str) -> list[int]:
    """提取答案中的引用编号，去重后升序返回。

    只做正则提取，不判断编号是否越界——
    越界与否取决于这次实际给了几条资料，那是调用方才知道的事。
    职责分开之后这个函数可以独立单测。
    """
    return sorted({int(m.group(1)) for m in CITATION_RE.finditer(answer)})


class GeneratePipeline:
    """问答流水线。持有检索器与 LLM，不持有全局状态。"""

    def __init__(
        self,
        llm: BaseLLM,
        retriever: BaseRetriever,
        *,
        empty_answer: str,
        max_context_tokens: int = 6000,
        top_k: Optional[int] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.empty_answer = empty_answer
        self.max_context_tokens = max_context_tokens
        # 默认不覆盖检索器自己的 top_k：那一侧是按「每路召回多少、融合后截多少」定的，
        # 在这里再砍一刀会让两条配置互相打架
        self.top_k = top_k
        self.log = get_logger("rag_core.generate")

    def ask(self, question: str, k: Optional[int] = None) -> QAResult:
        """回答一个问题，返回带引用与拒答标志的结果。"""
        started = time.perf_counter()

        hits = self.retriever.retrieve(question, k or self.top_k)
        if not hits:
            self.log.info("检索为空，拒答：%s", question)
            return QAResult(
                question=question,
                answer=self.empty_answer,
                retrieved=[],
                refused=True,
                elapsed_ms=_elapsed_ms(started),
            )

        kept = select_within_budget(hits, self.max_context_tokens)
        messages = build_messages(question, build_context(kept), self.empty_answer)
        answer = self.llm.chat(messages)

        cited = self._resolve_citations(answer, kept)
        refused = self._is_refusal(answer, cited)

        self.log.info(
            "检索 %d 条，送进 prompt %d 条，引用 %s，拒答=%s",
            len(hits), len(kept), cited, refused,
        )
        return QAResult(
            question=question,
            answer=answer,
            retrieved=hits,
            cited_indices=cited,
            refused=refused,
            elapsed_ms=_elapsed_ms(started),
        )

    def _resolve_citations(
        self,
        answer: str,
        kept: list[RetrievedChunk],
    ) -> list[int]:
        """把解析出的编号收敛到实际给出的资料范围内。"""
        valid = {hit.final_rank for hit in kept}
        raw = parse_cited_indices(answer)
        cited = [i for i in raw if i in valid]
        dropped = [i for i in raw if i not in valid]
        if dropped:
            self.log.warning(
                "答案引用了资料中不存在的编号 %s（实际给出 %d 条），已丢弃",
                dropped, len(kept),
            )
        return cited

    def _is_refusal(self, answer: str, cited: list[int]) -> bool:
        if self.empty_answer and self.empty_answer in answer:
            return True
        # 强制引用是硬约束：一个编号都没引，等于模型自己承认资料不够，
        # 或者它在用资料之外的知识作答。两者都不能当作可信答案放行。
        if not cited:
            return True
        return False


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
