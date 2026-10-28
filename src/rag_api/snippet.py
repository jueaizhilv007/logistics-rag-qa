"""从命中片段里挑出与问题最相关的那几句。

### 为什么检索单元和展示单元不是同一个东西

检索要 500 字的块，是因为一块里得装下完整的条款上下文——
否则「按有效货值的 20% 赔偿」会与它的适用前提
「须提供有效货值证明」被切到两个块里，模型拿到的是断章。

展示要给百来字，是因为客服只需要那一句赔付规则本身：
整块 449 字里真正相关的常常只有最后三行，
把 449 字整个糊上去，等于让客服自己从噪音里找答案。

两者需求相反，且不能互相迁就——
为了界面好看去缩小 chunk，是拿检索的质量去换展示的观感。
正确做法是检索照旧，展示时在块内再定位一次。

### 为什么用关键词打分而不是句向量

句向量要给每个候选句各调一次 embedding，而 `sources`
是在第一个 token 之前发出的：多一次批量调用就把首帧时间推后，
界面上表现为「点了以后先白等一阵」。

关键词打分是启发式的，会挑错——提问说「丢失」、片段写「损毁」
时词形不同就选不中。挑错的代价是「这一段没聚焦到最准的句子」，
而句向量的代价是每次提问都变慢。前者可以接受，后者不行。
真要上句向量，应做成可选项，让延迟敏感的场景能关掉。

### 为什么要带上命中句的前后各一句

条款的适用性写在前提里。只摘「按有效货值的 20% 赔偿」，
丢掉前一句的「能提供有效货值证明的」，
客服看到的是一个不完整、可能误用的规则。
宁可多带两句也不能摘出断章——这和检索要 500 字是同一个理由。
"""

from __future__ import annotations

import logging
import re

import jieba

# jieba 首次分词默认以 DEBUG 级别打印词典加载过程，与检索层的处理一致
jieba.setLogLevel(logging.WARNING)

# 提问里几乎每个片段都有的词。它们参与了也只是给所有句子加同一分，
# 没有区分度；用来高亮则会把整段标黄，等于没标。
_STOP = {
    "怎么", "什么", "多少", "是否", "可以", "需要", "如何", "哪些", "应当",
    "请问", "有没有", "是不是", "的是", "哪些", "标准", "规定", "情况", "处理",
}

# 句子的切分点：句末标点之后，以及空行（列表项之间靠空行分隔）
_SENT_SPLIT = re.compile(r"(?<=[。；！？])|\n+")

# 只保留中英文数字，滤掉标点与空白
_WORD_RE = re.compile(r"^[一-龥A-Za-z0-9]+$")

# 展示上限。180 字约等于两条中文长句，够装下「规则 + 前提」
_MAX_CHARS = 180


def terms_of(question: str) -> list[str]:
    """取问题里的检索词，供前端高亮用。

    滤掉单字与停用词：单字（的、是、在）会把整段都标黄，
    停用词出现在几乎每个片段里，标出来没有信息量。
    """
    seen: set[str] = set()
    out: list[str] = []
    for t in jieba.cut(question):
        t = t.strip()
        if len(t) < 2 or t in _STOP or t in seen:
            continue
        if not _WORD_RE.match(t):
            continue
        seen.add(t)
        out.append(t)
    return out


def _score(sentence: str, terms: list[str]) -> int:
    return sum(1 for t in terms if t in sentence)


def _clip(text: str, limit: int = _MAX_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit] + "…"


def pick_snippet(text: str, question: str, limit: int = _MAX_CHARS) -> str:
    """挑出片段中与问题最相关的部分，带上前后各一句。

    挑不出（片段与问题的词形完全不重叠）时退回开头两句：
    这时挑任何一句都是瞎猜，而瞎猜出的句子看起来像是有依据的聚焦，
    反而比一段明显不聚焦的开头更误导——后者至少让人知道要去展开看。
    """
    sentences = [s for s in _SENT_SPLIT.split(text) if s and s.strip()]
    if not sentences:
        return ""

    terms = terms_of(question)
    if not terms:
        return _clip("".join(sentences[:2]), limit)

    scores = [_score(s, terms) for s in sentences]
    best = max(range(len(scores)), key=lambda i: scores[i])
    if scores[best] == 0:
        return _clip("".join(sentences[:2]), limit)

    lo = max(0, best - 1)
    hi = min(len(sentences), best + 2)
    # 太短的片段读起来像残句：「公式为：」这类句子单独出现没有任何信息，
    # 而它恰恰常因为含关键词被选中。往后补到 80 字为止——
    # 宁可多带一点，也不要给客服一句没头没尾的话
    while hi < len(sentences) and len("".join(sentences[lo:hi]).strip()) < 80:
        hi += 1
    return _clip("".join(sentences[lo:hi]).strip(), limit)
