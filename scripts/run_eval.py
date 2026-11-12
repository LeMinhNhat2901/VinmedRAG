from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

from vimedrag.config import parse_args  # noqa: E402
from vimedrag.data import base_doc_id, check_or_write_eval_ids, load_rows, sample_eval  # noqa: E402
from vimedrag.index import BGEM3Embedder, HashingEmbedder, load_index  # noqa: E402
from vimedrag.pipeline import load_candidates, run_experiment  # noqa: E402
from vimedrag.report import check_index_compat  # noqa: E402
from vimedrag.retriever import DenseRetriever  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    idx = load_index(cfg.index_dir)
    check_index_compat(idx.meta, cfg)

    rows = load_rows(cfg.dataset, cfg.dataset_config, cfg.data_dir)
    n = cfg.dry_run_questions if cfg.dry_run else cfg.n_eval
    corpus_ids = {base_doc_id(d) for d in idx.doc_ids}
    items = sample_eval(rows["validation"], n, cfg.seed, corpus_ids)
    check_or_write_eval_ids(Path(cfg.runs_dir) / f"eval_ids_n{n}_seed{cfg.seed}.json", [i.question_idx for i in items])
    print(f"{cfg.run_name}: {len(items)} câu hỏi")

    retriever = None
    candidates = None
    if cfg.mode == "rerank" and not cfg.candidates_from and not cfg.dry_run:
        raise SystemExit("--mode rerank cần --candidates-from <predictions.jsonl của baseline> để dùng lại đúng top-10.")
    if cfg.mode == "rerank" and cfg.candidates_from:
        candidates = load_candidates(cfg.candidates_from)
        missing = [i.question_idx for i in items if i.question_idx not in candidates]
        short = [i.question_idx for i in items
                 if i.question_idx in candidates and len(candidates[i.question_idx]) < cfg.retrieve_k]
        unknown = sorted({x["doc_id"] for i in items for x in candidates.get(i.question_idx, [])[: cfg.retrieve_k]}
                         - set(idx.docs))
        if missing or short or unknown:
            raise SystemExit(
                f"Baseline {cfg.candidates_from} không dùng được: thiếu {len(missing)} câu (vd {missing[:3]}), "
                f"{len(short)} câu có < {cfg.retrieve_k} ứng viên (vd {short[:3]}), "
                f"{len(unknown)} doc_id không có trong index (vd {unknown[:3]}).")
        print(f"Dùng lại top-{cfg.retrieve_k} của baseline từ {cfg.candidates_from}", flush=True)
    if cfg.mode == "rag" or (cfg.mode == "rerank" and candidates is None):
        embedder = HashingEmbedder() if cfg.fake_embedder else BGEM3Embedder(cfg.embed_model)
        retriever = DenseRetriever(idx, embedder)

    reranker = None
    if cfg.mode == "rerank":
        from vimedrag.reranker import BGEReranker, FakeReranker
        reranker = FakeReranker() if cfg.dry_run else BGEReranker(
            cfg.reranker_model, batch_size=cfg.rerank_batch_size, max_length=cfg.rerank_max_length)
        run_dir = Path(cfg.runs_dir) / cfg.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        info = getattr(reranker, "info", {"model_id": "fake"})
        info = {**info, "candidates_from": cfg.candidates_from}
        (run_dir / "rerank_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.dry_run:
        from vimedrag.generator import FakeGenerator
        generator = FakeGenerator()
    else:
        from vimedrag.generator import VinaLLaMAGenerator
        generator = VinaLLaMAGenerator(cfg.model_id, cfg.max_new_tokens, cfg.repetition_penalty)

    bertscore_fn = None
    if cfg.bertscore:
        from vimedrag.evaluate import make_bertscore_fn
        bertscore_fn = make_bertscore_fn()

    metrics = run_experiment(cfg, items, idx.docs, generator, retriever=retriever, bertscore_fn=bertscore_fn,
                             reranker=reranker, candidates=candidates)
    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
