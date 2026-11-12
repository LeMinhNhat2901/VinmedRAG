from __future__ import annotations

import json
import re
import time
from pathlib import Path

from vimedrag.config import Config
from vimedrag.data import Doc, EvalItem, base_doc_id, load_jsonl, save_jsonl
from vimedrag.evaluate import aggregate
from vimedrag.prompts import build_messages
from vimedrag.reranker import rerank

FAIL_FAST_N = 3  # stop the run if the first N questions attempted in this run all fail
_DOC_MARKER = re.compile(r"^\[(\d+)\] ", re.MULTILINE)


def load_candidates(path) -> dict[str, list[dict]]:
    """Read a previous run's predictions.jsonl and return {question_idx: retrieved list} (latest record wins)."""
    out: dict[str, list[dict]] = {}
    for r in load_jsonl(path):
        out[r["question_idx"]] = r.get("retrieved") or []
    return out


def _used_doc_ids(messages: list[dict], docs: list[Doc]) -> list[str]:
    """IDs of the documents that actually made it into the prompt (build_messages may drop the tail)."""
    kept = len(_DOC_MARKER.findall(messages[-1]["content"]))
    return [d.doc_id for d in docs[:kept]]


def _docs_by_base(corpus: dict[str, Doc]) -> dict[str, list[Doc]]:
    out: dict[str, list[Doc]] = {}
    def key(doc_id: str):
        base, sep, idx = doc_id.partition("#")
        return (base, 0, 0) if not sep else (base, 1, int(idx) if idx.isdigit() else 0)

    for doc_id in sorted(corpus, key=key):
        out.setdefault(base_doc_id(doc_id), []).append(corpus[doc_id])
    return out


def run_experiment(cfg: Config, items: list[EvalItem], corpus: dict[str, Doc], generator,
                   retriever=None, bertscore_fn=None, reranker=None, candidates: dict | None = None) -> dict:
    if cfg.mode == "rag" and retriever is None:
        raise ValueError("mode 'rag' cần retriever")
    if cfg.mode == "rerank":
        if reranker is None:
            raise ValueError("mode 'rerank' cần reranker")
        if candidates is None and retriever is None:
            raise ValueError("mode 'rerank' cần candidates (top-k của baseline) hoặc retriever")
    out_dir = Path(cfg.runs_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    pred_path = out_dir / "predictions.jsonl"
    existing = load_jsonl(pred_path)
    if pred_path.exists():
        save_jsonl(pred_path, existing)  # drop malformed/truncated lines before appending
    done = {r["question_idx"] for r in existing if not r.get("error")}
    by_base = _docs_by_base(corpus) if cfg.mode == "oracle" else {}

    todo = sum(1 for i in items if i.question_idx not in done)
    print(f"== {cfg.run_name}: cần chạy {todo} câu, đã xong {len(items) - todo}", flush=True)
    attempted = 0
    first_errors: list[str] = []
    with pred_path.open("a", encoding="utf-8") as f:
        for item in items:
            if item.question_idx in done:
                continue
            rec = {
                "question_idx": item.question_idx,
                "topic": item.topic,
                "question": item.question,
                "gold_answer": item.answer,
                "gold_doc_id": item.gold_doc_id,
                "retrieved": [],
                "prediction": "",
                "latency_s": 0.0,
            }
            t0 = time.perf_counter()
            try:
                if cfg.mode == "rag":
                    hits = retriever.search(item.question, cfg.retrieve_k)
                    rec["retrieved"] = [{"doc_id": h.doc_id, "score": round(h.score, 4)} for h in hits]
                    docs = [corpus[h.doc_id] for h in hits[: cfg.top_k]]
                elif cfg.mode == "rerank":
                    if candidates is not None:
                        if item.question_idx not in candidates:
                            raise KeyError(f"không có top-{cfg.retrieve_k} baseline cho {item.question_idx}")
                        before = candidates[item.question_idx][: cfg.retrieve_k]
                    else:
                        before = [{"doc_id": h.doc_id, "score": round(h.score, 4)}
                                  for h in retriever.search(item.question, cfg.retrieve_k)]
                    rec["retrieved_before"] = before
                    rank_before = {x["doc_id"]: i for i, x in enumerate(before, 1)}
                    ranked = rerank(item.question, [x["doc_id"] for x in before], corpus, reranker)
                    rec["retrieved"] = [{"doc_id": d, "score": round(sc, 4), "rank_before": rank_before[d]}
                                        for d, sc in ranked]
                    docs = [corpus[d] for d, _ in ranked[: cfg.top_k]]
                elif cfg.mode == "oracle":
                    docs = by_base.get(item.gold_doc_id, [])
                    if not docs:
                        raise ValueError(f"không tìm thấy gold doc {item.gold_doc_id} trong corpus")
                else:
                    docs = []
                prompt_mode = "rag" if cfg.mode == "rerank" else cfg.mode  # identical prompt to the RAG baseline
                messages = build_messages(prompt_mode, item.question, docs, generator.count_tokens, cfg.max_prompt_tokens)
                if docs:
                    rec["used_doc_ids"] = _used_doc_ids(messages, docs)
                rec["prediction"] = generator.generate(messages)
            except Exception as e:  # keep going; failed items are retried on the next run
                rec["error"] = f"{type(e).__name__}: {e}"
            rec["latency_s"] = round(time.perf_counter() - t0, 3)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            attempted += 1
            shown = "ERR " + rec["error"] if rec.get("error") else rec["prediction"][:80]
            print(f"[{attempted}/{todo}] {item.question_idx} {rec['latency_s']:.1f}s {shown!r}", flush=True)
            if "error" in rec and attempted <= FAIL_FAST_N:
                first_errors.append(rec["error"])
            if attempted == FAIL_FAST_N and len(first_errors) == FAIL_FAST_N:
                raise RuntimeError(f"{FAIL_FAST_N} câu đầu đều lỗi, dừng sớm: {first_errors[-1]}")

    wanted = {i.question_idx for i in items}
    latest: dict[str, dict] = {}
    for r in load_jsonl(pred_path):
        if r["question_idx"] in wanted:
            latest[r["question_idx"]] = r
    preds = [latest[i.question_idx] for i in items if i.question_idx in latest]
    metrics = aggregate(preds)
    metrics["run_name"] = cfg.run_name
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if bertscore_fn is not None:
        try:
            with_bs = aggregate(preds, bertscore_fn=bertscore_fn)
            with_bs["run_name"] = cfg.run_name
            metrics_path.write_text(json.dumps(with_bs, ensure_ascii=False, indent=2), encoding="utf-8")
            metrics = with_bs
        except Exception as e:
            print(f"BERTScore lỗi, bỏ qua: {e}", flush=True)
    return metrics
