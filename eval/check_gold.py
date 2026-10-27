"""评测集标注自检。

评测集的可信度取决于一件事：每题的 gold 标注在语料里**确实存在**。
标注写错（文件名不存在、关键词是自己概括的、数值记错）不会报错，
只会让 Recall 莫名偏低，然后被误读成"检索策略不行"。

所以这个脚本只做静态校验，不跑检索、不调 LLM：
把每条标注拿回原文去比对，把对不上的条目点出来。

### 校验项

1. `id` 唯一
2. `source_docs` 里的每个文件名在 `data/docs/` 下真实存在
3. `answer_keywords` 里的每个词在对应文档原文中真实出现
4. `type` 属于封闭集合 waybill / region / amount / policy
5. 构成符合 60 单文档 / 10 跨文档 / 10 无解
6. 无解题不得带 ground_truth 与 answer_keywords（拒答判定不依赖它们）

第 3 项最容易出问题：关键词必须是原文里抄下来的词，
不能是自己概括的说法——概括出来的词在原文里根本不存在，
判定自然永远落空。

用法：uv run python eval/check_gold.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
DOCS = ROOT / "data" / "docs"

VALID_TYPES = {"waybill", "region", "amount", "policy"}
REQUIRED_FIELDS = {"id", "question", "source_docs", "answer_keywords", "type", "answerable"}
EXPECTED = {"single": 60, "cross": 10, "unanswerable": 10}


def load_questions() -> list[dict]:
    import json

    rows: list[dict] = []
    for lineno, line in enumerate(QUESTIONS.open(encoding="utf-8"), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append((lineno, json.loads(line)))
        except json.JSONDecodeError as exc:
            print(f"[致命] 第 {lineno} 行不是合法 JSON：{exc}")
            sys.exit(1)
    return rows


def classify(row: dict) -> str:
    if not row.get("answerable", True):
        return "unanswerable"
    return "cross" if len(row.get("source_docs") or []) >= 2 else "single"


def main() -> int:
    if not QUESTIONS.exists():
        print(f"[致命] 找不到评测集：{QUESTIONS}")
        return 1

    rows = load_questions()
    problems: list[str] = []

    # 1. id 唯一
    ids = Counter(row["id"] for _lineno, row in rows if "id" in row)
    for key, count in ids.items():
        if count > 1:
            problems.append(f"id 重复：{key} 出现 {count} 次")

    # 2/3/4. 逐条校验
    kind_count = Counter()
    cache: dict[str, str] = {}

    for lineno, row in rows:
        kind = classify(row)
        kind_count[kind] += 1

        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            problems.append(f"第 {lineno} 行缺字段：{sorted(missing)}")
            continue

        if row["type"] not in VALID_TYPES:
            problems.append(f"第 {lineno} 行 type 非法：{row['type']}")

        # 无解题按定义就没有来源文档，第 2、3 项对它不适用
        if kind == "unanswerable":
            continue

        sources = row["source_docs"] or []
        if not sources:
            problems.append(f"第 {lineno} 行 source_docs 为空")
            continue

        texts: list[str] = []
        for name in sources:
            if name not in cache:
                path = DOCS / name
                cache[name] = path.read_text(encoding="utf-8") if path.exists() else ""
                if not path.exists():
                    problems.append(f"第 {lineno} 行 source 文件不存在：{name}")
            if cache[name]:
                texts.append(cache[name])

        if not texts:
            continue

        for kw in row["answer_keywords"]:
            if not any(kw in t for t in texts):
                problems.append(
                    f"第 {lineno} 行关键词未在原文命中：{kw!r}"
                    f"（source_docs={sources}）"
                )

    # 5. 构成
    for kind, expect in EXPECTED.items():
        got = kind_count.get(kind, 0)
        if got != expect:
            problems.append(f"{kind} 期望 {expect} 条，实际 {got} 条")

    # 6. 无解题不该带 gold
    for lineno, row in rows:
        if classify(row) == "unanswerable":
            if row.get("ground_truth") or row.get("answer_keywords"):
                problems.append(f"第 {lineno} 行无解题带了 gold 标注")

    total = len(rows)
    print(f"评测集：{total} 条")
    print(
        f"构成：单文档 {kind_count.get('single', 0)} / "
        f"跨文档 {kind_count.get('cross', 0)} / "
        f"无解 {kind_count.get('unanswerable', 0)}"
    )

    if not problems:
        print("标注自检通过：文件名与关键词全部命中原文")
        return 0

    print(f"\n发现 {len(problems)} 个问题：")
    for item in problems:
        print(f"  - {item}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
