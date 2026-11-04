import json

import pytest

from vimedrag.config import Config
from vimedrag.report import check_index_compat, summarize_runs


def test_summarize_runs(tmp_path):
    for name, overall in {
        "rag_k3_vinallama7b": {"n": 200, "recall@1": 0.61234, "rougeL": 0.4},
        "closed_vinallama7b": {"n": 200, "rougeL": 0.2},
    }.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.json").write_text(json.dumps({"overall": overall}), encoding="utf-8")
    table = summarize_runs(tmp_path).splitlines()
    assert table[0].startswith("| run | n | recall@1")
    assert table[2].startswith("| closed_vinallama7b | 200 | - |")
    assert "| 0.6123 |" in table[3]
    assert len(table) == 4


def test_check_index_compat():
    check_index_compat({"dry_run": False, "embed_model": "BAAI/bge-m3"}, Config(mode="rag"))
    check_index_compat({"dry_run": True, "embed_model": "hashing"}, Config(mode="rag", dry_run=True, fake_embedder=True))
    check_index_compat({"dry_run": False, "embed_model": "hashing"}, Config(mode="closed"))
    with pytest.raises(ValueError, match="dry_run"):
        check_index_compat({"dry_run": True, "embed_model": "BAAI/bge-m3"}, Config(mode="rag"))
    with pytest.raises(ValueError, match="embed_model"):
        check_index_compat({"dry_run": False, "embed_model": "hashing"}, Config(mode="rag"))
