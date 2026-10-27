"""提示词构造：把检索结果编成带编号的参考资料。

### 编号直接用 final_rank，不再另编一套

模型输出的 `[N]` 会被回查成具体的 chunk（`QAResult.cited_chunks` 按 `final_rank` 查），
所以编号就是证据面板上的行号。用 `final_rank` 而不是重新编号，
是为了让「答案里的编号」与「证据面板的行」始终指向同一条记录。
两套编号并存时一旦错位，溯源面板展开的是另一段原文，
而这个错误在界面上完全看不出来——引用看着是好的，点开却是别的话。

### 拒答话术由调用方注入

话术取自 `generate.empty_answer` 配置，不写死在模板里。
客服话术是要给业务方调的，写死会让「改一句提示语」变成改代码、重新发版。
prompt 与拒答判定用的是同一个字符串，因此模型照话术回答时一定能被判出来。

### 原文自带的 [N] 必须转义

语料正文里天然存在方括号编号（本项目的法规文末就是 `[1] 《邮政法》` 这类引用列表），
它与我们给资料编的号同形。模型照抄进答案后会被解析成引用，
而这些编号往往落在合法区间内——越界校验抓不到，warning 也不会记，
`cited_indices` 和可追溯率就这样被静默污染。

所以拼装时把原文里的 `[N]` 转成全角 ［N］，两种方括号在形式上可区分。
转义放在拼装侧而不是解析侧：解析侧拿到的只是一段模型输出，
无从判断某个 `[3]` 是模型标的引用还是它从原文抄来的法规序号。
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_core.core.contracts import RetrievedChunk

_INLINE_BRACKET_NUM = re.compile(r"\[(\d+)\]")

SYSTEM_TEMPLATE = """你是物流服务规则问答助手。只依据下面给出的参考资料作答，不得使用参考资料之外的知识。

回答时必须在句末标注引用编号，格式为 [1]、[2]，编号对应参考资料序号。
如果参考资料不足以回答问题，只输出「{empty_answer}」，不要猜测。"""

USER_TEMPLATE = """参考资料：

{context}

问题：{question}"""


def escape_inline_brackets(text: str) -> str:
    """把原文自带的 [N] 转成全角 ［N］，与资料编号区分开。

    只作用在拼进 prompt 的这份字符串上，不碰 `Chunk` ——
    它是 frozen，展示给用户的正文要保持原样。
    """
    return _INLINE_BRACKET_NUM.sub(r"［\1］", text)


def build_context(hits: list[RetrievedChunk]) -> str:
    """把命中片段编成带 [N] 编号的参考资料块。

    编号取 `final_rank`，来源优先取文档标题、退化到文件名。
    带上标题链是为了让模型在「两段原文说法相近」时能靠上下文区分，
    否则它容易把赔付上限张冠李戴到另一家快递的规则上。

    编号标记在转义之后才拼上，否则会把自己的 `[1]` 也转成全角。
    """
    blocks: list[str] = []
    for hit in hits:
        chunk = hit.chunk
        title = chunk.doc_title or Path(chunk.source).name
        lines = [f"[{hit.final_rank}] 来源：{escape_inline_brackets(title)}"]
        if chunk.heading_path:
            lines.append(f"    标题链：{escape_inline_brackets(chunk.heading_path)}")
        lines.append(f"    {escape_inline_brackets(chunk.text).strip()}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    context: str,
    empty_answer: str,
) -> list[dict]:
    """装配 system / user 两条消息。"""
    return [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(empty_answer=empty_answer)},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]
