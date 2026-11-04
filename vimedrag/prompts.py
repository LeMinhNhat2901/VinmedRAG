from __future__ import annotations

from typing import Callable

from vimedrag.data import Doc

SYSTEM_BASE = "Bạn là trợ lý y khoa. Trả lời ngắn gọn, chính xác bằng tiếng Việt."
SYSTEM_DOCS = (
    SYSTEM_BASE
    + " Chỉ sử dụng thông tin trong các tài liệu được cung cấp. Nếu tài liệu không chứa câu trả lời,"
    + ' hãy nói "Tôi không tìm thấy thông tin trong tài liệu."'
)


def render_chatml(messages: list[dict]) -> str:
    body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return body + "<|im_start|>assistant\n"


def _format_docs(docs: list[Doc]) -> str:
    return "\n".join(f"[{i}] {d.title}\n{d.text}" for i, d in enumerate(docs, 1))


def _doc_messages(question: str, docs: list[Doc]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_DOCS},
        {"role": "user", "content": f"Tài liệu:\n{_format_docs(docs)}\n\nCâu hỏi: {question}"},
    ]


def build_messages(mode: str, question: str, docs: list[Doc], count_tokens: Callable[[str], int],
                   max_prompt_tokens: int = 3000) -> list[dict]:
    if mode == "closed":
        return [{"role": "system", "content": SYSTEM_BASE}, {"role": "user", "content": f"Câu hỏi: {question}"}]
    if mode not in ("rag", "oracle"):
        raise ValueError(f"mode không hợp lệ: {mode!r}")
    if not docs:
        raise ValueError(f"mode {mode!r} cần ít nhất một tài liệu")

    def fits(msgs: list[dict]) -> bool:
        return count_tokens(render_chatml(msgs)) <= max_prompt_tokens

    kept = list(docs)
    msgs = _doc_messages(question, kept)
    while not fits(msgs) and len(kept) > 1:
        kept.pop()
        msgs = _doc_messages(question, kept)
    if fits(msgs):
        return msgs
    last = kept[-1]
    words = last.text.split(" ")
    while words and not fits(msgs):
        words = words[: int(len(words) * 0.9)]
        kept[-1] = Doc(last.doc_id, last.title, " ".join(words), last.topic)
        msgs = _doc_messages(question, kept)
    return msgs
