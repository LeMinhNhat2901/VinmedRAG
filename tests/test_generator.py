from vimedrag.data import Doc
from vimedrag.generator import FakeGenerator, _needs_bos, clean_output
from vimedrag.prompts import build_messages


def test_fake_generator():
    g = FakeGenerator()
    msgs = build_messages("rag", "Hỏi?", [Doc("a", "T", "Câu một. Câu hai.", "drug")], g.count_tokens)
    assert g.generate(msgs) == "Câu một."
    assert g.generate(build_messages("closed", "Hỏi?", [], g.count_tokens)) == ""
    assert g.count_tokens("a b") == 2


def test_clean_output():
    assert clean_output("  Trả lời.<|im_end|>\nrác") == "Trả lời."
    assert clean_output("Trả lời <|im_start|>user") == "Trả lời"
    assert clean_output("  ok ") == "ok"


def test_vinallama_class_importable_without_gpu():
    import vimedrag.generator as g

    assert hasattr(g, "VinaLLaMAGenerator")


def test_needs_bos():
    assert _needs_bos("<s><|im_start|>user\nx", "<s>") is False
    assert _needs_bos("<|im_start|>user\nx", "<s>") is True
    assert _needs_bos("<s>x", None) is True
    assert _needs_bos("<|im_start|>user\nx", None) is True
