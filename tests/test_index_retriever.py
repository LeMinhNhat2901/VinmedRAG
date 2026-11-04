import json

import numpy as np
import pytest

from vimedrag.data import Doc
from vimedrag.index import HashingEmbedder, build_index, length_stats, load_index
from vimedrag.retriever import DenseRetriever

DOCS = [
    Doc("d1", "Tiểu đường", "bệnh tiểu đường insulin đường huyết", "disease"),
    Doc("d2", "Paracetamol", "thuốc giảm đau hạ sốt paracetamol", "drug"),
    Doc("d3", "Tim", "tim mạch huyết áp", "body-part"),
]


def test_hashing_embedder_normalized_and_deterministic():
    e = HashingEmbedder(dim=64)
    a = e.encode(["xin chào", "xin chào", ""])
    assert a.shape == (3, 64)
    assert a.dtype == np.float32
    assert np.allclose(a[0], a[1])
    assert np.allclose(np.linalg.norm(a[:2], axis=1), 1.0)
    assert np.allclose(a[2], 0.0)
    assert e.count_tokens("a b c") == 3


def test_length_stats():
    # embed_text word counts: d1 = 2 + 6 = 8, d2 = 1 + 6 = 7, d3 = 1 + 4 = 5 -> sorted [5, 7, 8]
    s = length_stats(DOCS, lambda t: len(t.split()))
    assert s == {"n": 3, "p50": 7, "p95": 8, "max": 8}
    assert length_stats([], len) == {"n": 0, "p50": 0, "p95": 0, "max": 0}


def test_build_load_search(tmp_path):
    e = HashingEmbedder(dim=256)
    build_index(DOCS, e, tmp_path, meta={"dry_run": True})
    idx = load_index(tmp_path)
    assert idx.doc_ids == ["d1", "d2", "d3"]
    assert idx.docs["d2"].title == "Paracetamol"
    assert idx.meta["dry_run"] is True
    assert idx.meta["n_docs"] == 3 and idx.meta["dim"] == 256
    assert idx.embeddings.dtype == np.float32
    r = DenseRetriever(idx, e)
    hits = r.search("thuốc paracetamol hạ sốt", k=2)
    assert [h.rank for h in hits] == [1, 2]
    assert hits[0].doc_id == "d2"
    assert hits[0].score >= hits[1].score
    assert len(r.search("tim", k=10)) == 3


def test_load_index_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        load_index(tmp_path / "none")
    build_index(DOCS, HashingEmbedder(dim=16), tmp_path)
    ids = json.loads((tmp_path / "doc_ids.json").read_text(encoding="utf-8"))
    (tmp_path / "doc_ids.json").write_text(json.dumps(ids[:2]), encoding="utf-8")
    with pytest.raises(ValueError, match="không khớp"):
        load_index(tmp_path)
