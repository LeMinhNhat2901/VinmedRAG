from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

from vimedrag.config import parse_args  # noqa: E402
from vimedrag.data import build_corpus, chunk_docs, load_rows, sample_eval  # noqa: E402
from vimedrag.index import BGEM3Embedder, HashingEmbedder, build_index, length_stats  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    rows = load_rows(cfg.dataset, cfg.dataset_config, cfg.data_dir)
    all_rows = [r for split_rows in rows.values() for r in split_rows]
    docs = build_corpus(all_rows)
    print(f"Corpus: {len(docs)} context khác nhau từ {len(all_rows)} dòng ({', '.join(rows)})")

    if cfg.dry_run:
        items = sample_eval(rows["validation"], cfg.dry_run_questions, cfg.seed, {d.doc_id for d in docs})
        gold = {i.gold_doc_id for i in items}
        others = sorted((d for d in docs if d.doc_id not in gold), key=lambda d: d.doc_id)
        docs = [d for d in docs if d.doc_id in gold] + others[: max(0, cfg.dry_run_docs - len(gold))]
        print(f"Dry-run: giữ {len(docs)} doc (gồm {len(gold)} gold doc)")

    embedder = HashingEmbedder() if cfg.fake_embedder else BGEM3Embedder(cfg.embed_model)
    stats = length_stats(docs, embedder.count_tokens)
    do_chunk = cfg.chunk == "on" or (cfg.chunk == "auto" and stats["p95"] > 512)
    if do_chunk:
        docs = chunk_docs(docs)
    stats.update(chunked=do_chunk, n_after_chunk=len(docs))
    print("Độ dài title+context (token):", json.dumps(stats))

    meta = {
        "dry_run": cfg.dry_run,
        "embed_model": "hashing" if cfg.fake_embedder else cfg.embed_model,
        "dataset": cfg.dataset,
        "length_stats": stats,
    }
    out = build_index(docs, embedder, cfg.index_dir, meta=meta, show_progress=True)
    print(f"Đã lưu index ({len(docs)} doc) vào {out}")


if __name__ == "__main__":
    main()
