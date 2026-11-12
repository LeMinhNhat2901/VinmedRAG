import pytest

from vimedrag.data import Doc
from vimedrag.reranker import FakeReranker, rerank

CORPUS = {
    "a": Doc("a", "Tim", "tim mạch huyết áp", "body-part"),
    "b": Doc("b", "Paracetamol", "thuốc giảm đau hạ sốt paracetamol", "drug"),
    "c": Doc("c", "Sốt", "hạ sốt cho trẻ", "disease"),
}


def test_fake_reranker_scores_by_overlap():
    r = FakeReranker()
    assert r.score("thuốc hạ sốt", ["thuốc hạ sốt", "tim mạch", ""]) == [3.0, 0.0, 0.0]


def test_rerank_orders_by_score_and_keeps_all_candidates():
    out = rerank("paracetamol hạ sốt", ["a", "c", "b"], CORPUS, FakeReranker())
    assert [d for d, _ in out] == ["b", "c", "a"]
    assert out[0][1] >= out[1][1] >= out[2][1]


def test_rerank_ties_keep_original_order():
    out = rerank("không khớp gì", ["c", "a", "b"], CORPUS, FakeReranker())
    assert [d for d, _ in out] == ["c", "a", "b"]


def test_rerank_unknown_doc_raises():
    with pytest.raises(KeyError):
        rerank("x", ["zzz"], CORPUS, FakeReranker())


def test_bge_reranker_importable_without_gpu():
    import vimedrag.reranker as m

    assert hasattr(m, "BGEReranker")


def test_rerank_rejects_duplicate_candidates():
    with pytest.raises(ValueError, match="trùng"):
        rerank("x", ["a", "a"], CORPUS, FakeReranker())
