# rag-qa · 物流智能客服 RAG 问答系统

基于 `AI_Study_Assistant` 检索内核改造的物流规则问答系统。内核包名 `rag_core`，可独立嵌入宿主项目，也可作为独立应用运行。

**无编排框架**：不引入 LangChain / LlamaIndex。RRF 融合、标题链注入、引用编号解析、拒答判定、评测指标全部自研。

## 当前进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| S1 | 库化改造与契约 | ✅ 已完成 |
| S2 | 语料采集与索引链路 | ✅ 已完成 |
| S3 | 检索层（纯向量 / 混合 + RRF） | ✅ 已完成 |
| S4 | 生成层（强制引用 + 拒答） | ✅ 已完成 |
| S5 | 评测集与评测脚本 | ✅ 已完成（自研检索指标优先于 RAGAS，见下） |
| S6 | 接入层与界面 | ✅ 已完成 |
| S7 | 四组对照实验与交付收尾 | ✅ 已完成（结果见下） |

阶段划分与完成判据见仓库外的 `../tasks.md`。

## 快速开始

```bash
# 1) 安装 uv（已装可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh
# 国内网络：上面这条会卡在 GitHub 下载，改用 PyPI 镜像——
#   pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2) 依赖与密钥
uv sync --extra api         # 内核依赖 + 接入层依赖（fastapi / uvicorn / sse-starlette）
cp .env.example .env        # 复制模板
vim .env                    # 填入 EMBEDDING_API_KEY 与 LLM_API_KEY

# 3) 建索引、起服务
uv run rag-qa ingest        # 建索引：12 份物流规则文档 → 188 块
                            # 块数随切片策略变：hierarchical=188、fixed=168，见下方对照表
uv run rag-api              # 起服务，浏览器打开 http://127.0.0.1:8000
```

- Python 版本由 `.python-version` 固定为 3.11，`uv sync` 自动准备，不碰系统 Python。
  国内网络若卡在这步：uv 下载 Python 同样走 GitHub，先用 conda/系统装好 3.11+，uv 会直接使用。
- 国内网络 `uv sync` 慢：`export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`
  （`pyproject.toml` 已配清华源但未设 `default`，PyPI 仍是默认。）
- 验证内核可 import 且无副作用：

```bash
uv run python -c "import rag_core; print(rag_core.__version__)"
```

- `ingest` 是幂等的（按 sha256 判重）。**若整个目录是从别的机器拷来的，会看到 12 skipped / 0 indexed**——跳过只说明索引已存在，不代表这台机器真的建过库，也不代表 `.env` 里的 key 有效。验证已有索引：

```bash
uv run python -c "
import lancedb
rows = lancedb.connect('./data/index').open_table('docs').to_arrow().to_pylist()
h = sum(1 for r in rows if (r.get('heading_path') or '').strip())
print(len(rows), '块 /', h, '块带标题链')
"
```

  期望 **188 / 188**。其他数字的含义：168 = 用 `fixed` 策略建的、164 = splitters 未更新（见下方说明）、376 = 重复入库。

- 想真正验证新机器能从零建库（切块 → 调 embedding → 写库，顺带验证 `.env`）：

```bash
rm -rf data/index && uv run rag-qa ingest
```

  期望 12 份全部 indexed、0 skipped、0 error，块数 188。

命令行问答：

```bash
uv run rag-qa ask "未保价的快件丢失怎么赔偿"
```

跑评测：

```bash
uv run python eval/run_retrieval.py --mode hybrid -k 5
```

> 首次问答要等十几秒（tiktoken 下载编码 + 建 BM25 索引），之后单条 4–6 秒。

## 四组对照实验

评测集 80 条自编题（70 条可答 / 10 条无解），覆盖 amount / policy / waybill / region 四类问法，其中 10 条跨文档题要求聚合多家快递的说法。**每次只改一个变量。**

> 题型分布受语料结构制约（amount 36 / policy 39 / waybill 4 / region 1），**waybill 与 region 样本太小，仅作参考、不进对比结论**——`run_retrieval.py` 的按题型统计只打印在终端，不写进 `summary.csv`，正是为了不让它被当成可比指标误读。跨文档分层则相反，它有明确的 60 / 10 分组，并且结论里就用到了它。

### 结果

| 实验 | 变量 | Recall@5 | HitRate@3 | 可追溯率 | 误拒 | 误引 | 单文档 | 跨文档 | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 `fixed+vector` | —（baseline） | 0.7714 | 0.6857 | 0.6857 | 0.2429 | 0.0714 | 0.7500 | 0.9000 | 3711 |
| E2 `hierarchical+vector` | 切片策略 | 0.9000 | 0.8000 | 0.8571 | 0.0714 | 0.0714 | 0.9167 | **0.8000** | 4267 |
| E3 `hierarchical+hybrid` | 检索策略 | **0.9857** | 0.9286 | 0.9571 | 0.0286 | 0.0143 | 0.9833 | 1.0000 | 4405 |
| E4 `+rerank` | 是否重排 | **0.9857** | **0.9714** | 0.9714 | 0.0286 | **0.0000** | 0.9833 | 1.0000 | 5697 |
| *（参考）*`fixed+hybrid` | — | 0.9571 | 0.8857 | 0.9143 | 0.0571 | 0.0286 | 0.9500 | 1.0000 | 3596 |

完整 16 列数据在 `data/eval/summary.csv`，逐题明细在 `data/eval/results/*.json`。

### 这张表要这样读

**一，只看 Recall@5 会得出错误结论。** E4 在 Recall@5 上与 E3 完全持平，但 HitRate@3 +4.3pp、误引归零。重排不是没用，是它**不改变「有没有召回」，只改变顺序**——E3 时 Recall@5 已到 0.9857，天花板就在那儿。反过来，E2 的 Recall@5 涨了 12.9pp，但收益几乎全部来自误拒（17 条 → 5 条），误引一条没少。**误拒与误引是两种失败，要分别治理。**

**二，同一项改动加在不同基线上，收益完全不同。**

| 改动 | 加在哪种基线上 | Recall@5 |
| --- | --- | --- |
| 加 BM25 | `fixed`（E1 → `fixed+hybrid`） | +18.6pp |
| 加 BM25 | `hierarchical`（E2 → E3） | +8.6pp |
| 换层级切片 | `vector`（E1 → E2） | +12.9pp |
| 换层级切片 | `hybrid`（`fixed+hybrid` → E3） | +2.9pp |

这说明两者**补的是同一个缺口、从两个方向**：标题链让每块自带归属信息，BM25 让精确词项不被语义平均稀释。因此叠用时边际递减——不是某套方案更好的证据，而是它们解决了同一批题的证据。

**三，注意 E2 那一行的跨文档掉了。** 层级切片并不是全面更好：它让块更小更聚焦，单文档精确定位大幅变准（+16.7pp），但跨文档题要在 5 个位置里凑齐多家说法，信息量变小的块反而凑不齐（−10pp）。这个缺口是 E3 的 BM25 补回来的。脱离「在哪套配置下」谈瓶颈会指错方向。

### 复现每一组

| 实验 | `chunking.strategy` | `retrieval.mode` | `rerank.enabled` | 重建索引 | 命令 |
| --- | --- | --- | --- | --- | --- |
| E1 | `fixed` | `vector` | `false` | 是（168 块） | `uv run rag-qa ingest && uv run python eval/run_retrieval.py --mode vector -k 5` |
| E2 | `hierarchical` | `vector` | `false` | 是（188 块） | `uv run rag-qa ingest && uv run python eval/run_retrieval.py --mode vector -k 5` |
| E3 | `hierarchical` | `hybrid` | `false` | **否** | `uv run python eval/run_retrieval.py --mode hybrid -k 5` |
| E4 | `hierarchical` | `hybrid` | `true` | **否** | `uv run python eval/run_retrieval.py --mode hybrid -k 5` |

`--mode` 可以临时覆盖配置里的 `retrieval.mode`，所以 E2→E3→E4 不需要重建索引。切**切片策略必须重建索引**——`uv run rag-qa ingest` 是幂等的（sha256 判重），但换了 strategy 要先 `rm -rf data/index`。

进库块数**因切片策略而异，两个数都是对的**：`fixed` = **168** 块、`hierarchical` = **188** 块（每块都带标题链）。所以上方「快速开始」写 188、而 E1 那行写 168，不是笔误，是两组实验本就用的不同策略。

**若你在新环境跑出 164 块，那是错的**：164 是 overlap 按字符硬切、句读对齐修复之前的旧值，出现它说明本地 splitters 没更新——那份索引里 164 个块有 13 个（7.9%）以标点开头，句子会被从中间切开。

仓库当前 `config.yaml` 停在 E4 那套（`hierarchical` + `rerank.enabled: true`）。 `retrieval.mode` 保持 `vector` 是有意的：baseline 口径必须是纯向量，默认开 hybrid 会让 E2 之后的归因失效。

### 逐步分析

- [01 · E1 baseline](./docs/notes/01-e1-baseline.md)
- [02 · E2 层级切片](./docs/notes/02-e2-层级切片.md)
- [03 · E3 混合检索](./docs/notes/03-e3-混合检索.md)
- [04 · E4 重排](./docs/notes/04-e4-重排.md)（含重排选型变更记录与两个实测踩坑）

### 关于 RAGAS

未接入，优先级低于自研检索指标。Recall@5 / HitRate@3 是判定「检索有没有召回」的硬指标，不依赖 LLM 评分、可复现；RAGAS 的四指标依赖评分模型且其分数与该评分模型强耦合。本项目要回答的问题是「切片与检索策略各自贡献多少」，自研指标已足够，且不会因为换评分模型而变动。

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
