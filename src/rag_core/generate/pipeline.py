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

### 拒答原因要随结果一起交出去

`refused` 只回答「拒没拒」，回答不了「为什么拒」。
而两种来源的含义完全不同：检索为空是资料库的问题，
有资料却不引是模型认为资料不够、或正在用外部知识作答。
评测里这两个数必须分开统计——「检索不到」和「不敢答」的优化方向相反，
混成一个拒答率就既不能指导调检索，也不能指导调 prompt。

渲染侧同样依赖它：`no_citation` 下 `answer` 是模型没有出处的自由发挥，
渲染方要知道手里这份 `answer` 不能当答案展示。
"""

from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Optional

import tiktoken

from rag_core.core.contracts import QAResult, RefusalReason, RetrievedChunk
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

    ### 计量口径是粗算，误差方向安全

    只数 `chunk.text`，不含 `[N] 来源：…` 这些拼装元信息，对实际 prompt 长度偏低；
    而 cl100k 对中文按字切，比 qwen 一类的中文词表切得碎，单条又偏高。
    两者方向相反，且当前配置下（4 条 × 500 字对 6000 预算）余量远大于误差，
    因此不引入端点 tokenizer——那是为一个跑不到的分支付依赖成本。
    预算真的开始咬人时，日志里的「送进 prompt N 条」会先于超窗暴露出来。
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


def _parse_cited_indices(answer: str) -> list[int]:
    """提取答案中的引用编号，去重后升序返回。

    只做正则提取，不判断编号是否越界——
    越界与否取决于这次实际给了几条资料，那是调用方才知道的事。
    职责分开之后这个函数可以独立单测。

    下划线前缀是刻意的：未收敛的编号（含模型编造的 `[99]`）不该流出本模块。
    对外只暴露经 `_resolve_citations` 收敛过的版本，
    让「拿到的一定是合法编号」成为结构保证，而不是靠调用方记得再过滤一次。
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
                refusal_reason="no_hits",
                elapsed_ms=_elapsed_ms(started),
            )

        kept = select_within_budget(hits, self.max_context_tokens)
        messages = build_messages(question, build_context(kept), self.empty_answer)
        answer = self.llm.chat(messages)

        cited = self._resolve_citations(answer, kept)
        reason = self._classify_refusal(answer, cited)
        refused = bool(reason)

        self.log.info(
            "检索 %d 条，送进 prompt %d 条，引用 %s，拒答=%s（%s）",
            len(hits), len(kept), cited, refused, reason or "-",
        )
        return QAResult(
            question=question,
            answer=answer,
            retrieved=hits,
            cited_indices=cited,
            refused=refused,
            refusal_reason=reason,
            elapsed_ms=_elapsed_ms(started),
        )

    def _resolve_citations(
        self,
        answer: str,
        kept: list[RetrievedChunk],
    ) -> list[int]:
        """把解析出的编号收敛到实际给出的资料范围内。"""
        valid = {hit.final_rank for hit in kept}
        raw = _parse_cited_indices(answer)
        cited = [i for i in raw if i in valid]
        dropped = [i for i in raw if i not in valid]
        if dropped:
            self.log.warning(
                "答案引用了资料中不存在的编号 %s（实际给出 %d 条），已丢弃",
                dropped, len(kept),
            )
        return cited

    def _classify_refusal(self, answer: str, cited: list[int]) -> RefusalReason:
        """判定拒答原因，返回空串表示未拒答。

        主判据是「一个编号都没引」：强制引用是硬约束，
        没引等于模型自己承认资料不够，或者它在用资料之外的知识作答——
        两者都不能当作可信答案放行。

        话术判据退到「整句等于话术」而不是子串包含。
        话术是配置项，业务方会改它，而子串匹配意味着改文案就等于改判别阈值：
        话术一旦改短（比如「不清楚」），带引用的正常答案只要顺带用到这几个字就被误杀。
        收紧到整句级之后，话术无论怎么改，都只命中「模型原样输出话术」这一种情况。
        """
        if not cited:
            return "no_citation"
        if self._normalized(answer) == self._normalized(self.empty_answer):
            return "refusal_phrase"
        return ""

    @staticmethod
    def _normalized(text: str) -> str:
        """去掉首尾空白与句尾标点。

        模型照话术回答时常常多吐一个句号，直接 `==` 会漏判。
        """
        return text.strip().rstrip("。.！!？?；;：:").strip()


def _elapsed_ms(started: float) -> int:
    """计算从 started 开始到现在的毫秒数。"""
    return int((time.perf_counter() - started) * 1000)
