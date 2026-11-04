import pytest

from vimedrag.data import Doc
from vimedrag.prompts import SYSTEM_BASE, SYSTEM_DOCS, build_messages, render_chatml


def wc(s):
    return len(s.split())


DOCS = [Doc("a", "Tiêu đề A", "nội dung một", "drug"), Doc("b", "Tiêu đề B", "nội dung hai", "drug")]


def test_system_prompts_verbatim():
    assert SYSTEM_BASE == "Bạn là trợ lý y khoa. Trả lời ngắn gọn, chính xác bằng tiếng Việt."
    assert SYSTEM_DOCS.startswith(SYSTEM_BASE + " Chỉ sử dụng thông tin")
    assert SYSTEM_DOCS.endswith('"Tôi không tìm thấy thông tin trong tài liệu."')


def test_closed():
    m = build_messages("closed", "Hỏi?", DOCS, wc)
    assert m == [{"role": "system", "content": SYSTEM_BASE}, {"role": "user", "content": "Câu hỏi: Hỏi?"}]


def test_rag_format():
    m = build_messages("rag", "Hỏi?", DOCS, wc)
    assert m[0] == {"role": "system", "content": SYSTEM_DOCS}
    assert m[1]["content"] == (
        "Tài liệu:\n[1] Tiêu đề A\nnội dung một\n[2] Tiêu đề B\nnội dung hai\n\nCâu hỏi: Hỏi?"
    )


def test_render_chatml():
    s = render_chatml([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert s == "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"


def test_drops_lowest_ranked_docs_when_too_long():
    full = wc(render_chatml(build_messages("rag", "Hỏi?", DOCS, wc)))
    m = build_messages("rag", "Hỏi?", DOCS, wc, max_prompt_tokens=full - 1)
    assert "[2]" not in m[1]["content"]
    assert "[1] Tiêu đề A\nnội dung một" in m[1]["content"]


def test_truncates_single_doc_text():
    big = [Doc("a", "T", " ".join(["từ"] * 500), "drug")]
    m = build_messages("oracle", "Hỏi?", big, wc, max_prompt_tokens=100)
    assert wc(render_chatml(m)) <= 100
    assert "[1] T\ntừ" in m[1]["content"]
    assert m[1]["content"].endswith("Câu hỏi: Hỏi?")


def test_requires_docs_and_valid_mode():
    with pytest.raises(ValueError):
        build_messages("rag", "Hỏi?", [], wc)
    with pytest.raises(ValueError):
        build_messages("foo", "Hỏi?", DOCS, wc)
