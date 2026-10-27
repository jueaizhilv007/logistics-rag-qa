"""数据契约：定义各模块之间流动的数据结构。

本文件只允许依赖标准库，不得引入任何第三方包或具体实现。
它是 ingest / retrieval / generate 三层的公共口径，
修改字段前应先同步设计文档，否则三层会各自漂移。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(frozen=True)
class LoadedPage:
    """解析后的单个页面或段落。"""
    text: str
    page: Optional[int] = None
    heading_path: str = ""


@dataclass(frozen=True)
class LoadedDoc:
    """解析后的完整文档，一个文件对应一个实例。"""
    doc_id: str
    source: str
    file_type: str
    pages: list[LoadedPage]
    doc_title: str = ""


@dataclass(frozen=True)
class Chunk:
    """索引与检索的最小单位。

    `@dataclass(frozen=True)` 让实例创建后 任何字段都不许修改:一个`Chunk` 从切分层产出后，
    会流经向量化、入库、检索、展示多个环节。
    如果它是可变的，某个环节随手改一下`text` （比如做个清洗），
    下游所有环节拿到的都是被污染的数据，而且极难排查。
    冻结之后，想变只能新建一个实例，数据的出处永远清晰。

    text 与 embed_text 分离的原因：层级切片会把父级标题链拼进检索文本，
    但展示给用户时要的是干净正文。两者混用会让界面出现重复标题。
    """
    text: str # 展示用正文，干净、不含标题链
    embed_text: str # 检索用文本，含标题链前缀
    doc_id: str
    source: str
    doc_title: str = ""
    heading_path: str = ""
    file_type: str = ""
    page: Optional[int] = None
    chunk_index: int = 0

    @property
    def chunk_id(self) -> str:
        """全局唯一标识，格式 {doc_id}_{chunk_index}。"""
        return f"{self.doc_id}_{self.chunk_index}"


@dataclass(frozen=True)
class TextRecord:
    """纯文本记录，供 BM25 等稀疏检索器构建索引。

    刻意不包含向量列：向量只在 ANN 检索时参与计算，
    若连同向量一起全量拉取，整个向量库会被读进内存。
    """
    text: str
    doc_id: str
    source: str
    doc_title: str = ""
    heading_path: str = ""
    file_type: str = ""
    page: Optional[int] = None
    chunk_index: int = 0

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}_{self.chunk_index}"

    def to_chunk(self) -> Chunk:
        """还原为 Chunk。

        稀疏检索路径不依赖 embedding 文本，
        因此还原出的 Chunk 其 embed_text 与 text 一致。
        """
        return Chunk(
            text=self.text,
            embed_text=self.text,
            doc_id=self.doc_id,
            source=self.source,
            doc_title=self.doc_title,
            heading_path=self.heading_path,
            file_type=self.file_type,
            page=self.page,
            chunk_index=self.chunk_index,
        )


@dataclass(frozen=True)
class VectorHit:
    """向量检索的原始命中项。"""
    record: TextRecord
    score: float


@dataclass
class RetrievedChunk:
    """检索层输出。

    三路排名字段同时保留，是为了能解释「这个结果为什么排第一」。
    只看最终分数无法归因到具体某一路检索器，
    而归因能力是对照实验和界面证据面板的共同前提。

    ### 与 Chunk 的关系：组合，不是继承
    注意第一个字段是`chunk: Chunk` ——`RetrievedChunk` 包着 一个完整的`Chunk` ，
    而不是继承它。

    这是刻意的。因为“一条检索结果 = 原始切片 + 一堆检索过程的度量信息”，
    这是两个正交的概念：切片信息在入库时就定死了，度量信息每次查询都不同。
    组合的好处是，同一个`Chunk` 在不同查询、不同策略下可以产生无数个`RetrievedChunk` ，
    各自记录各自的检索过程，互不干扰。

    `RetrievedChunk` 不冻结。原因在于 字段是分阶段填进去的 ：
    1. 索引阶段：Chunk 向量化后进 LanceDB（落库的是 8 个非向量列）；
               BM25 侧再经iter_text_records 全表回读组装 TextRecord 建倒排索引——此阶段只是数据就绪，RetrievedChunk 尚未产生
    2. 查询阶段：向量路径在 LanceDB 上 ANN 搜索，返回的命中载荷组装成 TextRecord，包成 VectorHit
    3. 查询阶段：BM25 路径在倒排索引上查询（与 VectorHit 无关，两路独立执行）
    4. 融合阶段：RRF 汇合两路，填 fused_score / final_rank
    5. 重排阶段：cross-encoder（重排器） 打分填 rerank_score，可能重写 final_rank
    """
    chunk: Chunk # 检索到的文本片段
    final_rank: int # 最终排名，从 1 开始
    fused_score: float = 0.0 # 最终分数（RRF 融合分），综合所有检索器的分数
    vector_rank: Optional[int] = None # 向量检索排名，从 1 开始，若未命中则为 None
    vector_score: Optional[float] = None # 向量检索分数，若未命中则为 None
    bm25_rank: Optional[int] = None # BM25 检索排名，从 1 开始，若未命中则为 None
    bm25_score: Optional[float] = None # BM25 检索分数，若未命中则为 None
    rerank_score: Optional[float] = None # 重排序分数，若未命中则为 None


# 拒答原因。空串表示未拒答。
# 封闭集合而非开放集合：消费方按这三种取值穷举渲染分支，新实现不应发明第四种，
# 判断标准与 IngestResult.status 一致（那边是开放/封闭取舍的另一半，不要统一口径）。
RefusalReason = Literal["", "no_hits", "no_citation", "refusal_phrase"]


@dataclass
class QAResult:
    """问答结果，供命令行、HTTP 接口与界面消费。
    
    `strategy` ：答案自带的“身份证” 。
    每条`QAResult` 自己声明“我是哪套配置产出来的”（如`fixed+vector` 或`hierarchical+hybrid+rerank` ）。
    这样评测结果落盘之后，哪怕文件混在一起，仍然能按 strategy 分组归因——不需要任何外部上下文。
    而且它由`describe()` 从配置实时推导，配置改了标签自动跟着变，杜绝“标签写着 hybrid、实际跑的是 vector”这类实验事故。

    `refused` 是独立标志，不是“空答案” 。
    拒答有两种来源：检索为空（一条证据都没有）、检索有结果但覆盖不足。
    所以`refused=True` 时`retrieved` 可能非空。
    这就是为什么界面判断“要不要渲染证据面板”应该看`retrieved` 而不是`refused` ——证据面板展示的是“我找到了什么、但不足以回答”，
    这本身就是有价值的透明度。拒答话术来自配置`generate.empty_answer` ，不硬编码在代码里。

    ### `answer` 在拒答时的可信度不一致，靠 `refusal_reason` 区分

    `refused=True` 下 `answer` 有两种截然不同的东西：

    - `no_hits` ：检索为空，没调 LLM，`answer` 就是配置话术，可信
    - `no_citation` ：有资料但模型一个编号都没引，`answer` 是模型没有出处的自由发挥，不可信
    - `refusal_phrase` ：模型照话术作答，`answer` 即话术，可信

    不区分的话，渲染方只能靠注释约定「拒答时别信 `answer` 」，
    而第 2 种情况一旦被当成答案展示，界面上看不出任何异常——
    这正是 RAG 最致命的失败模式。区分之后渲染契约按原因分三种：

    - `no_hits` ：展示话术，`retrieved` 为空，没有证据面板可展示
    - `refusal_phrase` ：展示话术与证据面板
    - `no_citation` ：展示话术、证据面板，外加把 `answer` 降级成「未获资料支持的参考」

    第 3 种不能简单隐藏：那份 `answer` 虽然不可信，却是唯一能看出模型差在哪的线索
    （答非所问，还是答对了只是没标引用），隐藏它等于断掉排查路径。
    所以它换个身份出现，而不是不出现。
    """
    question: str
    answer: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    cited_indices: list[int] = field(default_factory=list) # 被引用的片段排名，从 1 开始
    refused: bool = False
    refusal_reason: RefusalReason = ""
    elapsed_ms: int = 0
    strategy: str = ""

    @property
    def cited_chunks(self) -> list[RetrievedChunk]:
        """按引用编号取出被引用的片段，即可追溯率指标的统计对象。"""
        by_rank = {hit.final_rank: hit for hit in self.retrieved}
        return [by_rank[i] for i in self.cited_indices if i in by_rank]


@dataclass
class IngestResult:
    """单个文档的入库结果。

    status 是封闭集合而非开放集合：indexed 已入库 / skipped 已存在跳过 /
    error 解析或写入失败。调用方会按这三种取值穷举统计，新实现不应发明第四种，
    因此用 Literal 约束，让类型系统挡住拼写不一致。

    注意这与配置里的 provider 类字段相反：provider 是开放集合，
    宿主可在外部注册自有实现，用 str 而非 Literal。两者判断标准不同，不要统一。
    """
    source: str
    status: Literal["indexed", "skipped", "error"]
    doc_id: str = ""
    chunks: int = 0
    reason: str = ""
