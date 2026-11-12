from __future__ import annotations

import json
from pathlib import Path

from vimedrag.config import Config

COLUMNS = ["n", "recall@1", "recall@3", "recall@5", "mrr@10", "rougeL", "token_f1", "bertscore_f1", "refusal", "error"]


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, int):
        return str(v)
    return f"{v:.4f}"


def summarize_runs(runs_dir) -> str:
    lines = ["| run | " + " | ".join(COLUMNS) + " |", "|" + "---|" * (len(COLUMNS) + 1)]
    for path in sorted(Path(runs_dir).glob("*/metrics.json")):
        overall = json.loads(path.read_text(encoding="utf-8"))["overall"]
        cells = [path.parent.name] + [_fmt(overall.get(c)) for c in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


_KNOWN_EMBEDDERS = ("bge-m3",)


def _embed_key(name: str | None) -> str:
    """Identify an embedding model regardless of where it is loaded from.

    "BAAI/bge-m3" and a Kaggle input path such as "/kaggle/input/bge-m3/transformers/default/1"
    are the same model, so both map to "bge-m3". Unknown names are compared verbatim.
    """
    lowered = (name or "").lower()
    for key in _KNOWN_EMBEDDERS:
        if key in lowered:
            return key
    return name or ""


def check_index_compat(meta: dict, cfg: Config) -> None:
    if bool(meta.get("dry_run")) != cfg.dry_run:
        raise ValueError(
            f"Index có dry_run={bool(meta.get('dry_run'))} nhưng lệnh chạy dry_run={cfg.dry_run}. "
            "Dùng --index-dir đúng với chế độ chạy."
        )
    if cfg.mode == "rag" or (cfg.mode == "rerank" and not cfg.candidates_from):
        expected = "hashing" if cfg.fake_embedder else cfg.embed_model
        if _embed_key(meta.get("embed_model")) != _embed_key(expected):
            raise ValueError(f"Index dựng bằng embed_model={meta.get('embed_model')!r} nhưng lệnh chạy dùng {expected!r}.")
