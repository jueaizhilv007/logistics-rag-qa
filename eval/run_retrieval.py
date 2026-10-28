"""自研检索指标评测。

优先于 RAGAS 的原因：Recall@5 / HitRate@3 是判定「检索有没有召回」的硬指标，
不依赖 LLM 评分，可复现；RAGAS 的四指标依赖评分模型，且要额外依赖，放到有余力时再接。

### 指标口径

| 指标 | 分母 | 分子 |
| --- | --- | --- |
| Recall@5 | 70 条可答题 | Top5 中存在金标准片段的条数 |
| HitRate@3 | 70 条可答题 | Top3 中存在金标准片段的条数 |
| 可追溯率 | 70 条可答题 | `cited_chunks` 里确实包含答案要点的条数 |
| 误拒率-可答题 | 70 条可答题 | 可答题被判拒答的条数 |
| 误引率 | 70 条可答题 | 没拒答、带了引用、但引的不是金标准片段的条数 |
| 拒答率 | 10 条无解题 | 按 `refusal_reason` 拆成两个数 |

**可追溯率必须拆开看**：它等于「1 − 误拒 − 误引」，只看这一个数分不出失败在哪。
误拒是看得见的失败（界面有拒答框，用户转人工）；
误引是看不见的失败（界面是正常答案加假引用，用户照着错的答案执行）——
后者的产品风险更高，却在汇总里被埋没了，所以单列。

**分层是必须的**：跨文档题（10 条）要求聚合多家说法，失败率远高于单文档题（60 条），
混在总数里看不出切片策略与 rerank 有没有帮到这类题，
而那恰恰是 E2 / E4 唯一可能拉开差距的地方。
按题型同样分层，但它样本太小（waybill 仅 3 条、region 仅 1 条），
只进明细 json 不进汇总表——列进横向对比表会被当成可比指标误读。

**金标准片段** = 来源文档 ∈ `source_docs`，且正文命中足够多的 `answer_keywords`。
用列表而不是单值，单文档题也写成只含一个元素的列表，
判定逻辑因此只有一套，运行时不必判断「这题是不是跨文档」。

**可追溯率的分母取全部可答题**（口径 A）：可答题被误拒答时引用为空，该题记 0，
不从分母里剔除。误拒是真实失败，剔掉等于给自己的错误开脱，指标会偏好看，
且看不出「检索没问题、是模型不敢答」这一类问题。

**拒答率拆成两个数**，因为它们的优化方向相反：
- 「检索不到」＝ `no_hits` → 指向补语料
- 「有资料不敢答」＝ `no_citation` / `refusal_phrase` → 指向调 prompt 或阈值
混成一个数既不能指导调检索，也不能指导调 prompt。

用法：
    uv run python eval/run_retrieval.py --mode hybrid --k 5
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import typer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "data" / "eval" / "results"
SUMMARY_CSV = ROOT / "data" / "eval" / "summary.csv"

SUMMARY_HEADER = [
    "实验名", "切片策略", "检索策略", "重排", "k",
    "Recall@5", "HitRate@3", "可追溯率",
    "误拒率-可答题", "误引率",
    "Recall@5-单文档", "Recall@5-跨文档",
    "延迟p50", "延迟p95",
    "拒答率-检索不到", "拒答率-不敢答",
]

app = typer.Typer(add_completion=False)


def load_questions(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def is_gold(chunk_text: str, source: str, source_docs: list[str], keywords: list[str]) -> bool:
    """判定一个片段是否命中该条标注的金标准。

    两条约束都要过：来源文档 ∈ `source_docs`，且正文命中任一 `answer_keywords`。

    关键词取「任一命中」而不是「全部命中」：`answer_keywords` 里放的往往是
    同一事实的几种表述（如「保价费最低为 1 元」与「保价费为 1 元」），
    它们并不共现；要求全部命中会让这类标注永远判不中，
    表现出来的现象是 Recall 莫名偏低，然后被误读成检索策略不行。

    来源文档的约束已经足够强：同一份文档里出现金额、时限类关键词，
    基本可以认定为答案所在片段。
    """
    if Path(source).name not in source_docs:
        return False
    if not keywords:
        return False
    return any(kw in chunk_text for kw in keywords)


def percentile(values: list[int], p: float) -> int:
    """取百分位。80 条样本不插值，直接取排序后的对应位置。"""
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * p))]


def evaluate(core, questions: list[dict], k: int) -> dict:
    """逐条跑问答，累计指标，并按单/跨文档与题型分层。"""
    total_answerable = 0
    recall5 = 0
    hit3 = 0
    traceable = 0

    total_unanswerable = 0
    refused_no_hits = 0
    refused_hesitant = 0

    # 引用写法不被解析器识别的条数（如 【2】、[1,2]）。
    # 它此前混在 no_citation 里，于是解析器的缺口被记成了模型的过失。
    # 单独计数，但**不进 summary.csv**：历史那几行没有这一列，
    # 补列会让新老行的列含义不一致，正是此前在汇总表上犯过的那种错误
    bad_format = 0

    # 可答题的两种失败分开记：拒答看得见，误引看不见
    mis_refused = 0
    mis_cited = 0

    single_total = single_hit = 0
    multi_total = multi_hit = 0
    by_type: dict[str, dict[str, int]] = {}

    details: list[dict] = []

    for q in questions:
        result = core.ask(q["question"], k=k)
        hits = result.retrieved

        if result.refusal_reason == "bad_format":
            bad_format += 1

        def gold_of(hit) -> bool:
            return is_gold(
                hit.chunk.text,
                hit.chunk.source,
                q["source_docs"],
                q["answer_keywords"],
            )

        if q["answerable"]:
            total_answerable += 1
            gold_in_top5 = any(gold_of(h) for h in hits[:5])
            if gold_in_top5:
                recall5 += 1
            if any(gold_of(h) for h in hits[:3]):
                hit3 += 1

            cited = result.cited_chunks
            cited_gold = bool(cited) and any(gold_of(c) for c in cited)
            if cited_gold:
                traceable += 1
            elif result.refused:
                mis_refused += 1
            else:
                # 没拒答、有引用、但引的不是答案所在片段：
                # 界面上是绿框答案加一排引用角标，看起来完全正常
                mis_cited += 1

            multi = len(q["source_docs"]) > 1
            if multi:
                multi_total += 1
                multi_hit += int(gold_in_top5)
            else:
                single_total += 1
                single_hit += int(gold_in_top5)

            bucket = by_type.setdefault(
                q["type"], {"n": 0, "recall5": 0, "mis_refused": 0, "mis_cited": 0}
            )
            bucket["n"] += 1
            bucket["recall5"] += int(gold_in_top5)
            if not cited_gold:
                bucket["mis_refused" if result.refused else "mis_cited"] += 1
        else:
            gold_in_top5 = None
            multi = len(q["source_docs"]) > 1
            total_unanswerable += 1
            if result.refusal_reason == "no_hits":
                refused_no_hits += 1
            elif result.refusal_reason in ("no_citation", "refusal_phrase"):
                refused_hesitant += 1

        details.append(
            {
                "id": q["id"],
                "question": q["question"],
                "type": q["type"],
                "answerable": q["answerable"],
                "source_docs": q["source_docs"],
                "refused": result.refused,
                "refusal_reason": result.refusal_reason,
                "cited_indices": result.cited_indices,
                "elapsed_ms": result.elapsed_ms,
                "gold_hit_in_top5": gold_in_top5,
                "multi_doc": multi,
            }
        )

    def ratio(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    latency = [d["elapsed_ms"] for d in details]

    return {
        "n": len(questions),
        "n_answerable": total_answerable,
        "n_unanswerable": total_unanswerable,
        "recall@5": ratio(recall5, total_answerable),
        "hitrate@3": ratio(hit3, total_answerable),
        "traceable_rate": ratio(traceable, total_answerable),
        "mis_refused_rate": ratio(mis_refused, total_answerable),
        "mis_cited_rate": ratio(mis_cited, total_answerable),
        "recall@5_single": ratio(single_hit, single_total),
        "recall@5_multi": ratio(multi_hit, multi_total),
        "n_single": single_total,
        "n_multi": multi_total,
        "by_type": by_type,
        "latency_p50": percentile(latency, 0.5),
        "latency_p95": percentile(latency, 0.95),
        "refusal_no_hits_rate": ratio(refused_no_hits, total_unanswerable),
        "refusal_hesitant_rate": ratio(refused_hesitant, total_unanswerable),
        # 只进明细 json，不写进 summary.csv 的列定义——见 write_summary
        "bad_format": bad_format,
        "details": details,
    }


def print_report(strategy: str, summary: dict) -> None:
    print(f"\n策略 {strategy}")
    print(f"  题目        {summary['n']} 条（可答 {summary['n_answerable']} / 无解 {summary['n_unanswerable']}）")
    print(f"  Recall@5    {summary['recall@5']:.4f}")
    print(f"  HitRate@3   {summary['hitrate@3']:.4f}")
    print(f"  可追溯率    {summary['traceable_rate']:.4f}")
    print(f"    误拒      {summary['mis_refused_rate']:.4f}（看得见的失败：界面有拒答框）")
    print(f"    误引      {summary['mis_cited_rate']:.4f}（看不见的失败：正常答案加假引用）")
    print(
        f"  Recall@5·单文档 {summary['recall@5_single']:.4f}（{summary['n_single']} 条）"
        f" · 跨文档 {summary['recall@5_multi']:.4f}（{summary['n_multi']} 条）"
    )
    print(f"  延迟 p50/p95   {summary['latency_p50']} / {summary['latency_p95']} ms")
    print(f"  拒答率·检索不到   {summary['refusal_no_hits_rate']:.4f}")
    print(f"  拒答率·有资料不敢答 {summary['refusal_hesitant_rate']:.4f}")
    # 只在非 0 时出现：它是诊断信号而非常规指标，
    # 每次都打印会让它退化成一行没人看的背景噪音
    if summary.get("bad_format"):
        print(
            f"  引用写法不被识别 {summary['bad_format']} 条"
            f"（已计入误拒，此处单列：这是解析器的缺口，不是模型不引用）"
        )
    print("  按题型（样本小，只作参考，不进汇总表）")
    for kind, bucket in sorted(summary["by_type"].items()):
        n = bucket["n"]
        print(
            f"    {kind:<8} n={n:<3} Recall@5 {bucket['recall5'] / n:.4f}"
            f" · 误拒 {bucket['mis_refused']} · 误引 {bucket['mis_cited']}"
        )


@app.command()
def main(
    questions: Path = typer.Option(DEFAULT_QUESTIONS, "--questions", "-q", help="评测集路径"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="config.yaml 路径"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="临时覆盖 retrieval.mode"),
    k: int = typer.Option(5, "--top-k", "-k", help="检索条数，需 ≥5 才能算 Recall@5"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="实验名，默认用策略标签"),
) -> None:
    """跑评测集并输出四项指标。"""
    from rag_core.container import RagCore
    from rag_core.plugins import load_builtin_plugins

    load_builtin_plugins()
    core, cfg, _secrets = RagCore.from_config(
        str(config) if config else None, retrieval_mode=mode
    )

    rows = load_questions(questions)
    print(f"载入 {len(rows)} 条评测题 · 索引 {core.store.count()} 条 · 检索 {k} 条/题")

    summary = evaluate(core, rows, k)
    strategy = core.describe()
    # 默认名带上 k：同一策略跑 k=3 与 k=5 若都叫 fixed+hybrid，
    # 后一次会覆盖前一次，而这组对比正是要看的东西
    summary["实验名"] = name or f"{strategy}-k{k}"
    summary["k"] = k
    summary["切片策略"] = cfg.chunking.strategy
    summary["检索策略"] = cfg.retrieval.mode
    summary["重排"] = "是" if core.retriever is not None and getattr(core.retriever, "reranker", None) else "否"
    summary["strategy"] = strategy

    print_report(strategy, summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"{summary['实验名']}.json"
    out_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_summary(summary)
    print(f"\n明细：{out_json.relative_to(ROOT)}")
    print(f"汇总：{SUMMARY_CSV.relative_to(ROOT)}")


def write_summary(summary: dict) -> None:
    """追加一行到汇总表。同名的实验覆盖旧行，避免重复跑污染表格。

    列数对不上的旧行直接跳过：口径一变，旧行的每个数字都无法与新列对齐，
    硬拼进去会得到一张「每行含义不同」的表，比没有更误导。
    明细仍在 results/ 下，重跑即可并入。
    """
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing: list[list[str]] = []
    legacy = 0
    if SUMMARY_CSV.exists():
        with SUMMARY_CSV.open(encoding="utf-8", newline="") as fh:
            for row in csv.reader(fh):
                # 表头按首列认：旧口径的表头列数与现在不同，
                # 用 row == SUMMARY_HEADER 认不出来，会被误算进「旧口径数据行」
                if not row or row[0] in (SUMMARY_HEADER[0], summary["实验名"]):
                    continue
                if len(row) != len(SUMMARY_HEADER):
                    legacy += 1
                    continue
                existing.append(row)

    if legacy:
        print(f"\n跳过 {legacy} 行旧口径汇总（列数已变），明细仍在 results/ 下，重跑即可并入")

    existing.append(
        [
            summary["实验名"],
            summary["切片策略"],
            summary["检索策略"],
            summary["重排"],
            str(summary["k"]),
            f"{summary['recall@5']:.4f}",
            f"{summary['hitrate@3']:.4f}",
            f"{summary['traceable_rate']:.4f}",
            f"{summary['mis_refused_rate']:.4f}",
            f"{summary['mis_cited_rate']:.4f}",
            f"{summary['recall@5_single']:.4f}",
            f"{summary['recall@5_multi']:.4f}",
            str(summary["latency_p50"]),
            str(summary["latency_p95"]),
            f"{summary['refusal_no_hits_rate']:.4f}",
            f"{summary['refusal_hesitant_rate']:.4f}",
        ]
    )
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SUMMARY_HEADER)
        writer.writerows(existing)


if __name__ == "__main__":
    app()
