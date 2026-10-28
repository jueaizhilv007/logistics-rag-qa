"""重排实现：用服务端的 cross-encoder 给召回候选精排。

### 为什么用服务端重排，而不是 design 里写的 `bge_local`

design 原本写 `bge_local`，那是离线部署的思路：模型下载到本地，
推理不依赖外部服务。代价是 torch 与 sentence-transformers，
体积以 GB 计——而本项目其余依赖全是轻量库，
为一个对照组引入这一坨，会把「换台机器能一键跑起来」这个前提赔进去。

### 为什么也不用 LLM 打分（design 第 7 节给出的退路）

那条退路零新依赖，但它**不是 cross-encoder**：
让聊天模型读若干候选再输出一个排序，是 listwise 的提示工程——
各候选之间没有可比分数，输出格式还可能不合规。
而 design 5.3 节论证重排价值时用的论据是
「cross-encoder 把 query 与文档拼成一个序列，交互更充分」。
用 LLM 打分会得到与该论证不符的实现，
测出来的结论也就回答不了原本提出的那个问题。

服务端的 `/rerank` 端点是真正的 cross-encoder，零本地依赖、零额外体积。
因此取这条路线，选型变更按 design 第 7 节的要求记进笔记 04。

### 为什么失败要抛而不是静默降级

重排挂掉时若退回融合排序，E4 这组实验跑出来的数据
实际是「没有重排」的数据，而它会被当成「重排无效」写进结论。
静默降级制造的是无法察觉的假数据，比直接失败危险得多。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from rag_core.core.contracts import RetrievedChunk
from rag_core.core.logging import get_logger
from rag_core.registry import register

_DEFAULT_BASE_URL = "https://api.agicto.cn/v1/"


@register("reranker", "agicto")
class ApiCrossEncoderReranker:
    """调用 OpenAI 兼容服务端的 `/rerank` 端点做精排。

    协议形状与 Cohere 一致：POST `{model, query, documents}`，
    返回 `results[{index, relevance_score}]`（按分数降序）。
    """

    def __init__(
        self,
        top_n: int = 20,
        options: dict[str, Any] | None = None,
        api_key: str = "",
    ):
        opts = options or {}
        self.base_url = str(opts.get("base_url", _DEFAULT_BASE_URL)).rstrip("/")
        self.model = str(opts.get("model", "gte-rerank-v2"))
        self.timeout = float(opts.get("timeout", 30))
        self.top_n = top_n
        # 单次请求的文档上限，超出则分批。服务端实测硬性限 10 个，
        # 保持默认即与其对齐；调大只会多分几批，不会抬高候选池
        self.batch_size = int(opts.get("batch_size", 10))
        # 密钥由装配处从 Secrets 注入，不在实现内部读环境变量：
        # 读环境变量会让「密钥从哪来」出现两条路径，
        # 而这个问题不该有两种答案。「重排借用 llm 的密钥槽」这一决定
        # 因此也只写在 container 一处，实现本身不关心它属于谁
        self.api_key = (api_key or "").strip()
        if not self.api_key:
            raise ValueError(
                "重排器缺少密钥：装配未传入 api_key。"
                "请检查 .env 中 LLM_API_KEY 是否配置"
            )
        self.log = get_logger("rag_core.rerank")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """按 query 与候选的相关度重排，返回前 top_k 个。

        送入请求的文本取 `embed_text` 而不是 `text`：
        `embed_text` 才是参与检索的那一版（含标题链前缀），
        重排是在这条语义链上的延续，换用干净正文会让两级排序口径不一致——
        而这恰恰是 E2（层级切片）的价值要看的地方。
        """
        if not candidates:
            return []

        docs = [c.chunk.embed_text or c.chunk.text for c in candidates]
        scores = self._score(query, docs)

        if len(scores) < len(docs):
            # 缺分的排末尾而不是剔除：重排是「重排」不是「过滤」，
            # 因为没拿到分数就悄悄丢掉一条候选，是看不见的行为
            self.log.warning(
                "重排只取得 %d/%d 条候选的分数，缺分的排在末尾",
                len(scores), len(docs),
            )

        ranked = list(candidates)
        for i, item in enumerate(ranked):
            item.rerank_score = scores.get(i)

        # 缺分（None）排最后，其余按分数降序。
        # sort 是稳定的，分数并列时保持融合排序的相对次序，
        # 同一个输入因此每次都得到同一个输出
        ranked.sort(key=lambda x: (x.rerank_score is None, -(x.rerank_score or 0.0)))
        out: list[RetrievedChunk] = []
        for rank, item in enumerate(ranked[:top_k], start=1):
            item.final_rank = rank
            out.append(item)
        return out

    def _score(self, query: str, docs: list[str]) -> dict[int, float]:
        """给全部候选打分，按服务端上限分批。

        ### 为什么要分批

        实测该端点单次最多接受 10 个文档：请求 10 个正常返回 10 个，
        请求 11 / 15 / 20 个一律只返回 10 个，且不报错。
        不分批的话第 11 名之后拿不到分数——配置写的是 top_n=20，
        实际参与竞争的却只有前 10 条，界面与指标都看不出这件事。
        这就是「候选池被静默截断」，也正是本模块宁可失败抛错、
        也不做静默降级的原因所在。

        ### 为什么分批不影响分数可比性

        cross-encoder 对每个 (query, doc) 独立做一次前向传播，
        分数只取决于这一对；它不像某些 embedding 那样在批内做统计量归一化。
        因此同一批内与跨批的分数处在同一把尺子上，可以直接比。
        """
        scores: dict[int, float] = {}
        for start in range(0, len(docs), self.batch_size):
            batch = docs[start : start + self.batch_size]
            got = 0
            for row in self._post(query, batch):
                idx = row.get("index")
                if not isinstance(idx, int) or not 0 <= idx < len(batch):
                    continue
                scores[start + idx] = round(float(row.get("relevance_score", 0.0)), 6)
                got += 1
            if got < len(batch):
                self.log.warning(
                    "重排服务端只返回该批 %d 条中的 %d 条分数", len(batch), got
                )
        return scores

    def _post(self, query: str, docs: list[str]) -> list[dict]:
        """请求一次 `/rerank`，返回 results 列表。失败直接抛。"""
        payload = {
            "model": self.model,
            "query": query,
            "documents": docs,
            # 要该批全部文档的分数，而不是只让服务端给前几名：
            # 由它截断的话，落选者的分数既拿不到也无从推断
            "top_n": len(docs),
        }
        req = urllib.request.Request(
            self.base_url + "/rerank",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"重排请求失败（HTTP {exc.code}）："
                f"{exc.read().decode('utf-8', 'ignore')[:200]}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"重排请求失败：{exc}") from exc
        return data.get("results", [])
