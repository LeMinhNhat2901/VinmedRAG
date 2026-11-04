from __future__ import annotations

import json
import time
from pathlib import Path

from vimedrag.config import Config
from vimedrag.data import Doc, EvalItem, base_doc_id, load_jsonl, save_jsonl
from vimedrag.evaluate import aggregate
from vimedrag.prompts import build_messages

FAIL_FAST_N = 3  # stop the run if the first N questions attempted in this run all fail


def _docs_by_base(corpus: dict[str, Doc]) -> dict[str, list[Doc]]:
    out: dict[str, list[Doc]] = {}
    for doc_id in sorted(corpus):
        out.setdefault(base_doc_id(doc_id), []).append(corpus[doc_id])
    return out


def run_experiment(cfg: Config, items: list[EvalItem], corpus: dict[str, Doc], generator,
                   retriever=None, bertscore_fn=None) -> dict:
    if cfg.mode == "rag" and retriever is None:
        raise ValueError("mode 'rag' cần retriever")
    out_dir = Path(cfg.runs_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    pred_path = out_dir / "predictions.jsonl"
    existing = load_jsonl(pred_path)
    if pred_path.exists():
        save_jsonl(pred_path, existing)  # drop malformed/truncated lines before appending
    done = {r["question_idx"] for r in existing if not r.get("error")}
    by_base = _docs_by_base(corpus) if cfg.mode == "oracle" else {}

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
                elif cfg.mode == "oracle":
                    docs = by_base.get(item.gold_doc_id, [])
                    if not docs:
                        raise ValueError(f"không tìm thấy gold doc {item.gold_doc_id} trong corpus")
                else:
                    docs = []
                messages = build_messages(cfg.mode, item.question, docs, generator.count_tokens, cfg.max_prompt_tokens)
                rec["prediction"] = generator.generate(messages)
            except Exception as e:  # keep going; failed items are retried on the next run
                rec["error"] = f"{type(e).__name__}: {e}"
            rec["latency_s"] = round(time.perf_counter() - t0, 3)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            attempted += 1
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
    metrics = aggregate(preds, bertscore_fn=bertscore_fn)
    metrics["run_name"] = cfg.run_name
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics
