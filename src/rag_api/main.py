"""FastAPI 应用：把内核的流式事件映射成 SSE 帧。

### 内核与接入层的边界

内核不 import FastAPI，接入层不 import 检索或生成的任何细节。
本模块只做四件事：装配内核、按模式缓存实例、把 `StreamEvent` 转成 SSE 帧、挂载静态页面。
检索怎么召回、引用怎么解析、拒答怎么判定，一概不在这里——
那些逻辑一旦在接入层重写一遍，界面看到的结果就会与评测跑出的指标悄悄分叉。

### 为什么按模式缓存内核实例

切换检索模式要换 retriever，而 `HybridRetriever` 的 BM25 倒排索引是实例级的，
新建实例就得把全表文本读一遍重建。若每次请求都按 mode 现造一个，
方案切换开关每点一次都要付一次全表扫描，而在界面上这表现为「点一下卡一下」。
按 mode 缓存之后每种模式只建一次，切换变成 O(1) 取用。

### 为什么事件名要在这一层换一遍

内核用 `delta`，SSE 用 `token`。这不是啰嗦：`delta` 描述的是
「这是一块增量文本」，是语义；`token` 是与 `rag_mvp` 对齐的协议字段名，是形状。
两者共用一个名字，将来改协议就会连带改内核，而内核不该知道 HTTP 的存在。

### 为什么每个事件都过一次线程池

内核的 `stream()` 是同步生成器（OpenAI SDK 用的同步客户端）。
直接拿它喂给 async 生成器会阻塞事件循环：一个请求在等 LLM 吐字，
其他所有请求连同 SSE 心跳一起卡住。
每次 `next()` 丢进线程池，单请求只多一次线程切换，
换来的是事件循环始终可响应。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Iterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from rag_core.container import RagCore
from rag_core.core.contracts import RetrievedChunk, StreamEvent
from rag_core.plugins import load_builtin_plugins

# src/rag_api/main.py → 项目根
ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "web"

VALID_MODES = ("vector", "hybrid")

app = FastAPI(title="rag-qa", description="物流政策问答，检索证据可溯源")


class ChatRequest(BaseModel):
    """一次问答请求。

    `mode` 与 `k` 都可空：空则沿用配置文件的值。
    界面上的方案切换开关靠 `mode` 实现，因此它是请求级的，不是服务级的——
    两个人同时用不同方案提问必须互不影响。
    """

    question: str
    mode: Optional[str] = None
    k: Optional[int] = None


class _Runtime:
    """按检索模式缓存内核实例。

    检索器是内核的一部分，换 mode 就得换内核实例；
    而 embedding 客户端、向量库连接、LLM 客户端这些与 mode 无关的东西
    在每个实例里各建一份。这不是浪费：它们都是轻量句柄，
    真正昂贵的是 BM25 索引，而那份索引在同一 mode 下是复用的。
    """

    def __init__(self) -> None:
        load_builtin_plugins()
        self._cores: dict[str, RagCore] = {}
        self._cfg = None
        # 先用配置的默认模式建一个，其余模式等第一次被请求时再建
        core, cfg, _secrets = RagCore.from_config()
        self._cores[cfg.retrieval.mode] = core
        self._cfg = cfg
        self.default_mode = cfg.retrieval.mode

    def core(self, mode: Optional[str]) -> RagCore:
        key = mode or self.default_mode
        if key not in VALID_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的检索模式 {key!r}，可选 {list(VALID_MODES)}",
            )
        if key not in self._cores:
            core, cfg, _secrets = RagCore.from_config(retrieval_mode=key)
            self._cores[key] = core
            self._cfg = cfg
        return self._cores[key]

    @property
    def top_k(self) -> int:
        return self._cfg.retrieval.top_k


runtime = _Runtime()


def _hit_payload(hit: RetrievedChunk) -> dict:
    """把一个命中片段摊平成前端要的字段。

    `vector_rank` / `bm25_rank` 为 None 表示该路压根没召回它，
    保留 None 而不是填 0 或 -1：0 会被误读成「排第一」，
    -1 会参与排序。空就是空，界面显示「—」即可。
    """
    return {
        "rank": hit.final_rank,
        "source": Path(hit.chunk.source).name,
        "doc_title": hit.chunk.doc_title,
        "heading_path": hit.chunk.heading_path,
        "text": hit.chunk.text,
        "fused_score": round(hit.fused_score, 6),
        "vector_rank": hit.vector_rank,
        "bm25_rank": hit.bm25_rank,
        "rerank_score": hit.rerank_score,
    }


def _frame(event: str, payload: dict) -> dict:
    """拼一帧 SSE。ensure_ascii=False 让中文不被转义成 \\uXXXX。"""
    return {"event": event, "data": json.dumps(payload, ensure_ascii=False)}


def _to_frame(event: StreamEvent) -> dict:
    if event.type == "sources":
        return _frame("sources", {"hits": [_hit_payload(h) for h in event.hits]})
    if event.type == "delta":
        return _frame("token", {"text": event.text})
    if event.type == "error":
        return _frame("error", {"message": event.text})
    done = event.done
    return _frame(
        "done",
        {
            "cited_indices": done.cited_indices if done else [],
            "refused": done.refused if done else False,
            "refusal_reason": done.refusal_reason if done else "",
            "elapsed_ms": done.elapsed_ms if done else 0,
            "strategy": done.strategy if done else "",
        },
    )


async def _sse(core: RagCore, question: str, k: Optional[int]) -> Iterator[dict]:
    """把同步的流式事件转成异步的 SSE 帧序列。

    每个 `next()` 都过一次线程池，避免同步等待 LLM 时卡住事件循环。
    sentinel 用 None 而不是靠 StopIteration：异常穿过 `to_thread` 之后
    类型会变，用返回值判断结束比在两层之间传异常更不容易出错。
    """
    stream = core.stream(question, k)
    while True:
        event = await asyncio.to_thread(next, stream, None)
        if event is None:
            break
        yield _to_frame(event)


@app.get("/api/meta")
def meta() -> dict:
    """前端方案开关要的元信息。

    `strategy` 由 `describe()` 实时推导而非写死，
    因此界面上显示的标签与评测结果里的分组依据必然一致。
    """
    return {
        "modes": list(VALID_MODES),
        "default_mode": runtime.default_mode,
        "strategies": {
            mode: runtime.core(mode).describe() for mode in VALID_MODES
        },
        "top_k": runtime.top_k,
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> EventSourceResponse:
    """流式问答。

    返回 `text/event-stream`，事件四型：`sources` / `token` / `done` / `error`。
    LLM 中途失败时发 `error` 而不是断流——半截断的答案在界面上
    与「说完了」长得一模一样，是比报错更危险的一种失败。
    """
    core = runtime.core(req.mode)
    return EventSourceResponse(_sse(core, req.question, req.k))


# 静态页挂在最后：它是一个兜底路由，放前面的话会吃掉 /api 之外的所有路径，
# 而 API 的路径已经先注册过了，这里只是让根路径返回页面
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def run() -> None:
    """控制台脚本入口，等价于 uvicorn rag_api.main:app --reload。"""
    import uvicorn

    uvicorn.run("rag_api.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    run()
