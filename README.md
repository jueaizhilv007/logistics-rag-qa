# rag-qa · 物流智能客服 RAG 问答系统

基于 `AI_Study_Assistant` 检索内核改造的物流规则问答系统。内核包名 `rag_core`，可独立嵌入宿主项目，也可作为独立应用运行。

**无编排框架**：不引入 LangChain / LlamaIndex。RRF 融合、标题链注入、引用编号解析、拒答判定、评测指标全部自研。

## 当前进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| S1 | 库化改造与契约 | ✅ 已完成 |
| S2 | 语料采集与索引链路 | ✅ 已完成 |
| S3 | 检索层（纯向量 / 混合 + RRF） | ✅ 已完成（rerank 延后至 S7） |
| S4 | 生成层（强制引用 + 拒答） | ✅ 已完成 |
| S5 | 评测集与评测脚本 | 待开始 |
| S6 | 接入层与界面 | 待开始 |
| S7 | 四组对照实验与交付收尾 | 待开始 |

阶段划分与完成判据见仓库外的 `../tasks.md`。

## 安装

```bash
uv python pin 3.11     # 已在 .python-version 中固定，不碰系统 Python
uv sync                # 内核依赖
uv sync --extra api    # 接入层依赖（fastapi / uvicorn / sse-starlette），S6 起需要
cp .env.example .env   # 填入 LLM 与 Embedding 的 API Key
```

验证内核可 import 且无副作用：

```bash
uv run python -c "import rag_core; print(rag_core.__version__)"
```

## 两种装配方式

**显式注入**——把 `rag_core` 当库嵌入，不读配置文件、不依赖工作目录：

```python
from rag_core import RagCore

core = RagCore(embedding=emb, store=store, llm=llm)
```

**配置驱动**——独立应用与命令行使用，按 `config.yaml` 装配：

```python
from rag_core import RagCore

core, cfg, secrets = RagCore.from_config("config.yaml")
print(core.describe())   # 策略标签，如 hierarchical+hybrid+rerank
```

配置驱动路径要求调用方已 import 具体实现模块以触发注册，未注册时抛 `UnknownKindError` 并列出可用项，不静默降级。

## 契约分层

- `core/contracts.py`：数据契约，只允许依赖标准库，是三层的公共口径
- `core/protocols.py`：行为契约，用 `Protocol` 结构化子类型，实现类无需继承本包基类即可被接受

## 目录结构

```
src/rag_core/
├── __init__.py        # 对外 API 收口
├── config.py          # pydantic 配置模型
├── container.py       # 装配根 RagCore
├── registry.py        # 插件注册，仅 from_config 路径使用
├── core/              # contracts / protocols / logging
├── ingest/            # loader / splitter / pipeline   [S2]
├── retrieval/         # vector / bm25 / hybrid / rerank [S3]
├── generate/          # prompt / pipeline              [S4]
└── cli.py             # Typer 入口                     [S2]
```

依赖方向单向：`rag_core` 不得 import 接入层 `rag_api`。
