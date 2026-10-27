"""FastAPI + SSE 接入层。

可选依赖（`uv sync --extra api` 才装 fastapi / uvicorn / sse-starlette）；
内核 `rag_core` 不依赖本包，也不 import 它。
HTTP 只是内核的一种外挂，与 CLI 同级，不是内核的一部分。
"""
