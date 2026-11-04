import unicodedata

import pytest

from vimedrag.evaluate import (
    aggregate,
    is_refusal,
    mrr_at_k,
    recall_at_k,
    rouge_l_f1,
    token_f1,
    tokenize_vi,
)


def test_tokenize_vi_keeps_diacritics_and_normalizes():
    nfd = unicodedata.normalize("NFD", "Thuốc Paracetamol, 500mg!")
    assert tokenize_vi(nfd) == ["thuốc", "paracetamol", "500mg"]
    assert tokenize_vi(None) == []


def test_rouge_l():
    assert rouge_l_f1("a b c d", "a b c d") == 1.0
    assert rouge_l_f1("a x c", "a b c d") == pytest.approx(4 / 7)
    assert rouge_l_f1("", "a b") == 0.0
    assert rouge_l_f1("x y", "a b") == 0.0
    nfd = unicodedata.normalize("NFD", "Đau đầu")
    assert rouge_l_f1(nfd, "đau đầu.") == 1.0


def test_token_f1():
    assert token_f1("có có không", "có không không") == pytest.approx(2 / 3)
    assert token_f1("", "a") == 0.0
    assert token_f1("a b", "a b") == 1.0


def test_recall_mrr_with_chunks():
    ids = ["x", "g#1", "y"]
    assert recall_at_k(ids, "g", 1) == 0.0
    assert recall_at_k(ids, "g", 2) == 1.0
    assert mrr_at_k(ids, "g", 10) == 0.5
    assert mrr_at_k(ids, "g", 1) == 0.0
    assert mrr_at_k(ids, "z", 10) == 0.0


def test_is_refusal():
    assert is_refusal("Tôi không tìm thấy thông tin trong tài liệu.")
    assert is_refusal("TÔI KHÔNG TÌM THẤY THÔNG TIN.")
    assert not is_refusal("Có, thuốc trị đầy hơi.")


def test_aggregate():
    preds = [
        {"question_idx": "1", "topic": "drug", "prediction": "a b", "gold_answer": "a b",
         "gold_doc_id": "g", "retrieved": [{"doc_id": "g", "score": 0.9}]},
        {"question_idx": "2", "topic": "disease", "prediction": "", "gold_answer": "x y",
         "gold_doc_id": "h", "retrieved": [{"doc_id": "z", "score": 0.5}, {"doc_id": "h", "score": 0.4}],
         "error": "RuntimeError: CUDA out of memory"},
    ]
    m = aggregate(preds, bertscore_fn=lambda p, r: [1.0, 0.0])
    o = m["overall"]
    assert o["n"] == 2
    assert o["rougeL"] == 0.5
    assert o["recall@1"] == 0.5
    assert o["recall@3"] == 1.0
    assert o["mrr@10"] == 0.75
    assert o["error"] == 0.5
    assert o["refusal"] == 0.0
    assert o["bertscore_f1"] == 0.5
    assert m["by_topic"]["drug"]["n"] == 1
    assert m["by_topic"]["drug"]["token_f1"] == 1.0


def test_aggregate_closed_mode_has_no_retrieval_metrics():
    preds = [{"question_idx": "1", "topic": "drug", "prediction": "a", "gold_answer": "a",
              "gold_doc_id": "g", "retrieved": []}]
    o = aggregate(preds)["overall"]
    assert "recall@1" not in o
    assert o["rougeL"] == 1.0


def test_aggregate_empty():
    assert aggregate([]) == {"overall": {"n": 0}, "by_topic": {}}
