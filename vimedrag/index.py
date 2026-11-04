from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from vimedrag.data import Doc, load_jsonl, save_jsonl


class HashingEmbedder:
    """CPU-only stand-in for BGE-M3 used in tests and --fake-embedder runs."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


class BGEM3Embedder:
    """BGE-M3 dense embeddings: CLS vector of the last hidden state, L2-normalized."""

    def __init__(self, model_id: str = "BAAI/bge-m3", device: str | None = None,
                 batch_size: int = 32, max_length: int = 512):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()

    def count_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=True)["input_ids"])

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        import torch

        dim = self.model.config.hidden_size
        if not texts:
            return np.zeros((0, dim), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
        out = np.zeros((len(texts), dim), dtype=np.float32)
        starts = range(0, len(order), self.batch_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm
                starts = tqdm(starts, desc="encode", total=(len(order) + self.batch_size - 1) // self.batch_size)
            except ImportError:
                pass
        for start in starts:
            idx = order[start:start + self.batch_size]
            batch = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                cls = self.model(**batch).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls.float(), dim=-1)
            out[idx] = cls.cpu().numpy()
        return out


def length_stats(docs: list[Doc], count_tokens) -> dict:
    lens = sorted(count_tokens(d.embed_text()) for d in docs)
    if not lens:
        return {"n": 0, "p50": 0, "p95": 0, "max": 0}

    def pct(p: float) -> int:
        return lens[min(len(lens) - 1, int(round(p * (len(lens) - 1))))]

    return {"n": len(lens), "p50": pct(0.5), "p95": pct(0.95), "max": lens[-1]}


@dataclass
class LoadedIndex:
    doc_ids: list[str]
    embeddings: np.ndarray
    docs: dict[str, Doc]
    meta: dict


def build_index(docs: list[Doc], embedder, out_dir, meta: dict | None = None,
                show_progress: bool = False) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = embedder.encode([d.embed_text() for d in docs], show_progress=show_progress)
    save_jsonl(out / "corpus.jsonl", (asdict(d) for d in docs))
    np.save(out / "embeddings.npy", emb.astype(np.float16))
    (out / "doc_ids.json").write_text(json.dumps([d.doc_id for d in docs]), encoding="utf-8")
    full_meta = {**(meta or {}), "n_docs": len(docs), "dim": int(emb.shape[1])}
    (out / "meta.json").write_text(json.dumps(full_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_index(out_dir) -> LoadedIndex:
    out = Path(out_dir)
    needed = ["corpus.jsonl", "embeddings.npy", "doc_ids.json", "meta.json"]
    missing = [n for n in needed if not (out / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"Thiếu {missing} trong {out}. Chạy scripts/build_index.py (build_index) trước."
        )
    doc_ids = json.loads((out / "doc_ids.json").read_text(encoding="utf-8"))
    emb = np.load(out / "embeddings.npy").astype(np.float32)
    docs = {r["doc_id"]: Doc(**r) for r in load_jsonl(out / "corpus.jsonl")}
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    if not (len(doc_ids) == emb.shape[0] == len(docs)) or set(doc_ids) != set(docs):
        raise ValueError(
            f"Index không khớp corpus trong {out}: doc_ids={len(doc_ids)}, "
            f"embeddings={emb.shape[0]}, corpus={len(docs)}. Hãy dựng lại index."
        )
    return LoadedIndex(doc_ids, emb, docs, meta)
