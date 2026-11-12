import json

import pytest

from vimedrag.config import Config
from vimedrag.data import Doc, EvalItem, load_jsonl
from vimedrag.generator import FakeGenerator
from vimedrag.index import HashingEmbedder, build_index, load_index
from vimedrag.pipeline import _docs_by_base, run_experiment
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
                             "retrieved", "prediction", "latency_s", "used_doc_ids"}
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


def test_fail_fast_on_systematic_error(tmp_path):
    idx, _, _ = _setup(tmp_path)
    items = [
        EvalItem("q1", "câu a", "x", "disease", "g1"),
        EvalItem("q2", "câu b", "x", "drug", "g2"),
        EvalItem("q3", "câu c", "x", "body-part", "n1"),
    ]
    cfg = _cfg(tmp_path, mode="oracle")
    gen = CountingGen(fail_on={"câu a", "câu b", "câu c"})
    with pytest.raises(RuntimeError, match="dừng sớm"):
        run_experiment(cfg, items, idx.docs, gen)
    preds = load_jsonl(tmp_path / "runs" / cfg.run_name / "predictions.jsonl")
    assert len(preds) == 3
    assert all("error" in p for p in preds)


def test_oracle_missing_gold_is_error(tmp_path):
    idx, _, _ = _setup(tmp_path)
    items = [EvalItem("q1", "hỏi", "x", "disease", "zzz")]
    m = run_experiment(_cfg(tmp_path, mode="oracle"), items, idx.docs, CountingGen())
    assert m["overall"]["error"] == 1.0


def test_progress_output(tmp_path, capsys):
    idx, ret, items = _setup(tmp_path)
    run_experiment(_cfg(tmp_path, mode="rag", top_k=1, retrieve_k=3), items, idx.docs, CountingGen(), retriever=ret)
    out = capsys.readouterr().out
    assert "rag_k1_vinallama7b" in out and "cần chạy 2 câu, đã xong 0" in out
    assert "[1/2] q1 " in out and "[2/2] q2 " in out


def test_bertscore_failure_keeps_metrics(tmp_path, capsys):
    idx, ret, items = _setup(tmp_path)

    def boom(preds, golds):
        raise RuntimeError("no model")

    cfg = _cfg(tmp_path, mode="rag", top_k=1, retrieve_k=3)
    m = run_experiment(cfg, items, idx.docs, CountingGen(), retriever=ret, bertscore_fn=boom)
    assert "BERTScore lỗi, bỏ qua: no model" in capsys.readouterr().out
    saved = json.loads((tmp_path / "runs" / cfg.run_name / "metrics.json").read_text(encoding="utf-8"))
    assert "bertscore_f1" not in json.dumps(saved) and "bertscore_f1" not in json.dumps(m)


def test_bertscore_success_written(tmp_path):
    idx, ret, items = _setup(tmp_path)
    cfg = _cfg(tmp_path, mode="rag", top_k=1, retrieve_k=3)
    m = run_experiment(cfg, items, idx.docs, CountingGen(), retriever=ret,
                       bertscore_fn=lambda p, g: [0.5] * len(p))
    saved = (tmp_path / "runs" / cfg.run_name / "metrics.json").read_text(encoding="utf-8")
    assert "bertscore_f1" in saved and "bertscore_f1" in json.dumps(m)


def test_docs_by_base_numeric_order():
    corpus = {k: Doc(k, "t", "x", "disease") for k in ["a#10", "a#2", "a", "a#1", "b"]}
    assert [d.doc_id for d in _docs_by_base(corpus)["a"]] == ["a", "a#1", "a#2", "a#10"]


from vimedrag.pipeline import load_candidates  # noqa: E402
from vimedrag.reranker import FakeReranker  # noqa: E402


class PromptSpyGen(CountingGen):
    def __init__(self):
        super().__init__()
        self.prompts = []

    def generate(self, messages):
        self.prompts.append(messages[-1]["content"])
        return super().generate(messages)


def _candidates_file(tmp_path, rows):
    p = tmp_path / "baseline.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return p


def test_load_candidates_takes_latest_record(tmp_path):
    p = _candidates_file(tmp_path, [
        {"question_idx": "q1", "retrieved": [{"doc_id": "x", "score": 0.1}]},
        {"question_idx": "q1", "retrieved": [{"doc_id": "g1", "score": 0.9}, {"doc_id": "n1", "score": 0.5}]},
    ])
    assert load_candidates(p) == {"q1": [{"doc_id": "g1", "score": 0.9}, {"doc_id": "n1", "score": 0.5}]}


def test_rerank_mode_reuses_candidates_and_logs_both_orders(tmp_path):
    idx, _, items = _setup(tmp_path)
    # baseline order puts the gold doc last; the reranker must move it to the front
    cands = {
        "q1": [{"doc_id": "n1", "score": 0.9}, {"doc_id": "g2", "score": 0.8}, {"doc_id": "g1", "score": 0.7}],
        "q2": [{"doc_id": "n1", "score": 0.9}, {"doc_id": "g1", "score": 0.8}, {"doc_id": "g2", "score": 0.7}],
    }
    cfg = _cfg(tmp_path, mode="rerank", top_k=1, retrieve_k=3)
    gen = PromptSpyGen()
    m = run_experiment(cfg, items, idx.docs, gen, reranker=FakeReranker(), candidates=cands)
    preds = load_jsonl(tmp_path / "runs" / cfg.run_name / "predictions.jsonl")
    p1 = preds[0]
    assert [x["doc_id"] for x in p1["retrieved_before"]] == ["n1", "g2", "g1"]
    assert p1["retrieved"][0]["doc_id"] == "g1"
    assert {x["doc_id"] for x in p1["retrieved"]} == {"n1", "g2", "g1"}
    assert p1["retrieved"][0]["rank_before"] == 3
    assert p1["used_doc_ids"] == ["g1"]
    assert p1["used_doc_ids_before"] == ["n1"]
    assert p1["rerank_s"] >= 0
    assert gen.prompts[0].startswith("Tài liệu:\n[1] T1\n")
    assert m["overall"]["recall@1"] == 1.0
    assert m["overall"]["rougeL"] == 1.0


def test_rerank_mode_uses_same_prompt_as_rag(tmp_path):
    idx, ret, items = _setup(tmp_path)
    rag_gen, rr_gen = PromptSpyGen(), PromptSpyGen()
    run_experiment(_cfg(tmp_path, mode="rag", top_k=1, retrieve_k=3), items, idx.docs, rag_gen, retriever=ret)
    rag_preds = load_jsonl(tmp_path / "runs" / "rag_k1_vinallama7b" / "predictions.jsonl")
    cands = {p["question_idx"]: p["retrieved"] for p in rag_preds}

    class KeepOrder:
        def score(self, query, texts):
            return [float(-i) for i in range(len(texts))]

    run_experiment(_cfg(tmp_path, mode="rerank", top_k=1, retrieve_k=3), items, idx.docs, rr_gen,
                   reranker=KeepOrder(), candidates=cands)
    assert rr_gen.prompts == rag_gen.prompts


def test_rerank_mode_missing_candidates_is_error(tmp_path):
    idx, _, items = _setup(tmp_path)
    cfg = _cfg(tmp_path, mode="rerank", top_k=1, retrieve_k=3)
    m = run_experiment(cfg, items[:1], idx.docs, CountingGen(), reranker=FakeReranker(), candidates={})
    assert m["overall"]["error"] == 1.0


def test_rerank_mode_requires_reranker(tmp_path):
    idx, _, items = _setup(tmp_path)
    with pytest.raises(ValueError, match="reranker"):
        run_experiment(_cfg(tmp_path, mode="rerank"), items, idx.docs, CountingGen(), candidates={})


def test_used_doc_ids_logged_for_rag(tmp_path):
    idx, ret, items = _setup(tmp_path)
    run_experiment(_cfg(tmp_path, mode="rag", top_k=2, retrieve_k=3), items, idx.docs, CountingGen(), retriever=ret)
    p = load_jsonl(tmp_path / "runs" / "rag_k2_vinallama7b" / "predictions.jsonl")[0]
    assert p["used_doc_ids"] == [x["doc_id"] for x in p["retrieved"][:2]]
