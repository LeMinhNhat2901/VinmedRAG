import unicodedata
from collections import Counter

import pytest

from vimedrag.data import (
    Doc,
    base_doc_id,
    build_corpus,
    check_or_write_eval_ids,
    chunk_docs,
    load_jsonl,
    make_doc_id,
    normalize_text,
    sample_eval,
    save_jsonl,
)

TOPICS = ["body-part", "disease", "drug", "medicine"]


def _row(i, topic, ctx):
    return {
        "question_idx": f"{topic}_{i}",
        "question": f"Câu hỏi {i}?",
        "answer": f"Trả lời {i}.",
        "context": ctx,
        "title": f"Bài {i}",
        "topic": topic,
    }


def _val_rows():
    return [_row(i, t, f"context {t} {i}") for t in TOPICS for i in range(10)]


def test_normalize_text_nfc_and_whitespace():
    nfd = unicodedata.normalize("NFD", " Bệnh  tiểu\nđường ")
    out = normalize_text(nfd)
    assert out == "Bệnh tiểu đường"
    assert unicodedata.is_normalized("NFC", out)
    assert normalize_text(None) == ""


def test_doc_id_stable_and_normalized():
    a = make_doc_id("Thuốc  A")
    assert a == make_doc_id(" Thuốc A ")
    assert len(a) == 12
    assert a != make_doc_id("Thuốc B")


def test_build_corpus_dedups_and_skips_empty():
    rows = [_row(1, "drug", "ctx A"), _row(2, "drug", "ctx  A"), _row(3, "disease", "ctx B"), _row(4, "disease", "  ")]
    docs = build_corpus(rows)
    assert [d.text for d in docs] == ["ctx A", "ctx B"]
    assert docs[0].title == "Bài 1"
    assert docs[0].embed_text() == "Bài 1\nctx A"


def test_sample_eval_balanced_and_deterministic():
    rows = _val_rows()
    ids = {d.doc_id for d in build_corpus(rows)}
    a = sample_eval(rows, 8, 42, ids)
    b = sample_eval(rows, 8, 42, ids)
    assert [x.question_idx for x in a] == [x.question_idx for x in b]
    assert Counter(x.topic for x in a) == {t: 2 for t in TOPICS}
    assert [x.question_idx for x in a] == sorted(x.question_idx for x in a)
    assert all(x.gold_doc_id in ids for x in a)
    assert a[0].question.startswith("Câu hỏi")
    c = sample_eval(rows, 8, 7, ids)
    assert [x.question_idx for x in c] != [x.question_idx for x in a]


def test_sample_eval_remainder_goes_to_first_topics():
    rows = _val_rows()
    ids = {d.doc_id for d in build_corpus(rows)}
    counts = Counter(x.topic for x in sample_eval(rows, 6, 42, ids))
    assert counts == {"body-part": 2, "disease": 2, "drug": 1, "medicine": 1}


def test_sample_eval_independent_of_corpus_subset():
    rows = _val_rows()
    ids = {d.doc_id for d in build_corpus(rows)}
    full = sample_eval(rows, 8, 42, ids)
    subset = {x.gold_doc_id for x in full}
    assert sample_eval(rows, 8, 42, subset) == full


def test_sample_eval_errors():
    rows = _val_rows()
    ids = {d.doc_id for d in build_corpus(rows)}
    with pytest.raises(ValueError, match="gold"):
        sample_eval(rows, 4, 42, set())
    with pytest.raises(ValueError, match="topic"):
        sample_eval(rows, 44, 42, ids)


def test_chunk_docs():
    long = Doc("abc", "T", " ".join(f"w{i}" for i in range(700)), "drug")
    short = Doc("def", "T", "ngắn", "drug")
    out = chunk_docs([long, short], max_words=300, overlap=40)
    assert [d.doc_id for d in out] == ["abc#0", "abc#1", "abc#2", "def"]
    assert out[1].text.split()[0] == "w260"
    assert out[2].text.split()[-1] == "w699"
    assert out[0].title == "T"
    assert base_doc_id("abc#2") == "abc"
    assert base_doc_id("def") == "def"


def test_jsonl_roundtrip_and_truncated_tail(tmp_path):
    p = tmp_path / "x.jsonl"
    save_jsonl(p, [{"a": "Bệnh"}, {"a": 2}])
    with p.open("a", encoding="utf-8") as f:
        f.write('{"a": 3')
    with pytest.warns(UserWarning):
        rows = load_jsonl(p)
    assert rows == [{"a": "Bệnh"}, {"a": 2}]
    assert load_jsonl(tmp_path / "missing.jsonl") == []


def test_eval_ids_guard(tmp_path):
    p = tmp_path / "sub" / "eval_ids.json"
    check_or_write_eval_ids(p, ["a", "b"])
    check_or_write_eval_ids(p, ["a", "b"])
    with pytest.raises(ValueError, match="eval_ids"):
        check_or_write_eval_ids(p, ["a", "c"])
