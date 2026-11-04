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
from vimedrag.pipeline import run_experiment  # noqa: E402
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
    if cfg.mode == "rag":
        embedder = HashingEmbedder() if cfg.fake_embedder else BGEM3Embedder(cfg.embed_model)
        retriever = DenseRetriever(idx, embedder)

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

    metrics = run_experiment(cfg, items, idx.docs, generator, retriever=retriever, bertscore_fn=bertscore_fn)
    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
