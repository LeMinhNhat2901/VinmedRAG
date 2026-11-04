import json

import pytest

from vimedrag.config import Config
from vimedrag.data import Doc, EvalItem, load_jsonl
from vimedrag.generator import FakeGenerator
from vimedrag.index import HashingEmbedder, build_index, load_index
from vimedrag.pipeline import run_experiment
from vimedrag.retriever import DenseRetriever


class CountingGen(FakeGenerator):
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def generate(self, messages):
        q = messages[-1]["content"].rsplit("Câu hỏi: ", 1)[-1]
        self.calls.append(q)
        if q in self.fail_on:
            raise RuntimeError("CUDA out of memory")
        return super().generate(messages)


def _setup(tmp_path):
    docs = [
        Doc("g1", "T1", "đái tháo đường là bệnh mạn tính. Thêm.", "disease"),
        Doc("g2", "T2", "paracetamol hạ sốt. Thêm.", "drug"),
        Doc("n1", "T3", "tim mạch", "body-part"),
    ]
    emb = HashingEmbedder(dim=256)
    build_index(docs, emb, tmp_path / "idx")
    idx = load_index(tmp_path / "idx")
    items = [
        EvalItem("q1", "đái tháo đường là gì", "đái tháo đường là bệnh mạn tính.", "disease", "g1"),
        EvalItem("q2", "paracetamol dùng làm gì", "paracetamol hạ sốt.", "drug", "g2"),
    ]
    return idx, DenseRetriever(idx, emb), items


def _cfg(tmp_path, **kw):
    return Config(runs_dir=str(tmp_path / "runs"), bertscore=False, **kw)


def test_rag_run_writes_outputs(tmp_path):
    idx, ret, items = _setup(tmp_path)
    cfg = _cfg(tmp_path, mode="rag", top_k=1, retrieve_k=3)
    m = run_experiment(cfg, items, idx.docs, CountingGen(), retriever=ret)
    d = tmp_path / "runs" / "rag_k1_vinallama7b"
    preds = load_jsonl(d / "predictions.jsonl")
    assert [p["question_idx"] for p in preds] == ["q1", "q2"]
    assert len(preds[0]["retrieved"]) == 3
    assert set(preds[0]) == {"question_idx", "topic", "question", "gold_answer", "gold_doc_id",
                             "retrieved", "prediction", "latency_s"}
    assert json.loads((d / "config.json").read_text(encoding="utf-8"))["top_k"] == 1
    assert json.loads((d / "metrics.json").read_text(encoding="utf-8"))["overall"]["n"] == 2
    assert m["overall"]["recall@3"] == 1.0
    assert m["run_name"] == "rag_k1_vinallama7b"


def test_oracle_uses_gold_and_closed_has_no_docs(tmp_path):
    idx, _, items = _setup(tmp_path)
    m = run_experiment(_cfg(tmp_path, mode="oracle"), items, idx.docs, CountingGen())
    assert m["overall"]["rougeL"] == 1.0
    m0 = run_experiment(_cfg(tmp_path, mode="closed"), items, idx.docs, CountingGen())
    assert m0["overall"]["rougeL"] == 0.0
    assert "recall@1" not in m0["overall"]


def test_oracle_with_chunked_gold(tmp_path):
    docs = {"g1#0": Doc("g1#0", "T", "phần một.", "drug"), "g1#1": Doc("g1#1", "T", "phần hai.", "drug")}
    items = [EvalItem("q1", "hỏi", "phần một.", "drug", "g1")]
    m = run_experiment(_cfg(tmp_path, mode="oracle"), items, docs, CountingGen())
    assert m["overall"]["rougeL"] == 1.0


def test_resume_skips_done_and_retries_errors(tmp_path):
    idx, _, items = _setup(tmp_path)
    cfg = _cfg(tmp_path, mode="oracle")
    m1 = run_experiment(cfg, items, idx.docs, CountingGen(fail_on={"paracetamol dùng làm gì"}))
    assert m1["overall"]["error"] == 0.5
    g2 = CountingGen()
    m2 = run_experiment(cfg, items, idx.docs, g2)
    assert g2.calls == ["paracetamol dùng làm gì"]
    assert m2["overall"]["error"] == 0.0
    assert m2["overall"]["n"] == 2


def test_resume_repairs_truncated_line(tmp_path):
    idx, _, items = _setup(tmp_path)
    cfg = _cfg(tmp_path, mode="oracle")
    run_experiment(cfg, items[:1], idx.docs, CountingGen())
    p = tmp_path / "runs" / cfg.run_name / "predictions.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write('{"question_idx": "q2", "predic')
    g = CountingGen()
    with pytest.warns(UserWarning):
        run_experiment(cfg, items, idx.docs, g)
    assert g.calls == ["paracetamol dùng làm gì"]
    assert [r["question_idx"] for r in load_jsonl(p)] == ["q1", "q2"]


def test_metrics_only_cover_requested_items(tmp_path):
    idx, _, items = _setup(tmp_path)
    cfg = _cfg(tmp_path, mode="oracle")
    run_experiment(cfg, items, idx.docs, CountingGen())
    m = run_experiment(cfg, items[:1], idx.docs, CountingGen())
    assert m["overall"]["n"] == 1


def test_rag_requires_retriever(tmp_path):
    idx, _, items = _setup(tmp_path)
    with pytest.raises(ValueError, match="retriever"):
        run_experiment(_cfg(tmp_path, mode="rag"), items, idx.docs, CountingGen())
