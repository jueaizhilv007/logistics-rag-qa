"""自研检索指标评测。

优先于 RAGAS 的原因：Recall@5 / HitRate@3 是判定「检索有没有召回」的硬指标，
不依赖 LLM 评分，可复现；RAGAS 的四指标依赖评分模型，且要额外依赖，放到有余力时再接。

### 四个指标的口径

| 指标 | 分母 | 分子 |
| --- | --- | --- |
| Recall@5 | 70 条可答题 | Top5 中存在金标准片段的条数 |
| HitRate@3 | 70 条可答题 | Top3 中存在金标准片段的条数 |
| 可追溯率 | 70 条可答题 | `cited_chunks` 里确实包含答案要点的条数 |
| 拒答率 | 10 条无解题 | 按 `refusal_reason` 拆成两个数 |

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
    "实验名", "切片策略", "检索策略", "重排",
    "Recall@5", "HitRate@3", "可追溯率",
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


def evaluate(core, questions: list[dict], k: int) -> dict:
    """逐条跑问答，累计四项指标。"""
    total_answerable = 0
    recall5 = 0
    hit3 = 0
    traceable = 0

    total_unanswerable = 0
    refused_no_hits = 0
    refused_hesitant = 0

    details: list[dict] = []

    for q in questions:
        result = core.ask(q["question"], k=k)
        hits = result.retrieved

        def gold_of(hit) -> bool:
            return is_gold(
                hit.chunk.text,
                hit.chunk.source,
                q["source_docs"],
                q["answer_keywords"],
            )

        if q["answerable"]:
            total_answerable += 1
            if any(gold_of(h) for h in hits[:5]):
                recall5 += 1
            if any(gold_of(h) for h in hits[:3]):
                hit3 += 1
            cited = result.cited_chunks
            if cited and any(gold_of(c) for c in cited):
                traceable += 1
        else:
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
                "gold_hit_in_top5": any(gold_of(h) for h in hits[:5]) if q["answerable"] else None,
            }
        )

    def ratio(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    return {
        "n": len(questions),
        "n_answerable": total_answerable,
        "n_unanswerable": total_unanswerable,
        "recall@5": ratio(recall5, total_answerable),
        "hitrate@3": ratio(hit3, total_answerable),
        "traceable_rate": ratio(traceable, total_answerable),
        "refusal_no_hits_rate": ratio(refused_no_hits, total_unanswerable),
        "refusal_hesitant_rate": ratio(refused_hesitant, total_unanswerable),
        "details": details,
    }


def print_report(strategy: str, summary: dict) -> None:
    print(f"\n策略 {strategy}")
    print(f"  题目        {summary['n']} 条（可答 {summary['n_answerable']} / 无解 {summary['n_unanswerable']}）")
    print(f"  Recall@5    {summary['recall@5']:.4f}")
    print(f"  HitRate@3   {summary['hitrate@3']:.4f}")
    print(f"  可追溯率    {summary['traceable_rate']:.4f}")
    print(f"  拒答率·检索不到   {summary['refusal_no_hits_rate']:.4f}")
    print(f"  拒答率·有资料不敢答 {summary['refusal_hesitant_rate']:.4f}")


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
    summary["实验名"] = name or strategy
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
    """追加一行到汇总表。同名的实验覆盖旧行，避免重复跑污染表格。"""
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing: list[list[str]] = []
    if SUMMARY_CSV.exists():
        with SUMMARY_CSV.open(encoding="utf-8", newline="") as fh:
            existing = [row for row in csv.reader(fh) if row and row[0] != summary["实验名"]]

    existing.append(
        [
            summary["实验名"],
            summary["切片策略"],
            summary["检索策略"],
            summary["重排"],
            f"{summary['recall@5']:.4f}",
            f"{summary['hitrate@3']:.4f}",
            f"{summary['traceable_rate']:.4f}",
            f"{summary['refusal_no_hits_rate']:.4f}",
            f"{summary['refusal_hesitant_rate']:.4f}",
        ]
    )
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SUMMARY_HEADER)
        writer.writerows(existing[1:] if existing and existing[0] == SUMMARY_HEADER else existing)


if __name__ == "__main__":
    app()
