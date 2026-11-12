from __future__ import annotations

from vimedrag.data import Doc


class FakeReranker:
    """CPU stand-in for tests and dry-runs: score = number of query words present in the text."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = query.lower().split()
        out = []
        for t in texts:
            words = set(t.lower().split())
            out.append(float(sum(1 for w in q if w in words)))
        return out


class BGEReranker:
    """Cross-encoder reranker (BAAI/bge-reranker-v2-m3): one raw relevance logit per (query, text) pair."""

    def __init__(self, model_id: str = "BAAI/bge-reranker-v2-m3", device: str | None = None,
                 batch_size: int = 16, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        import transformers

        self.info = {
            "model_id": model_id,
            "revision": getattr(self.model.config, "_commit_hash", None),
            "device": self.device,
            "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else None,
            "dtype": str(next(self.model.parameters()).dtype),
            "max_length": max_length,
            "batch_size": batch_size,
            "transformers": transformers.__version__,
            "score": "raw logit (num_labels=1), pair = (question, title + newline + context)",
        }
        print(f"[BGEReranker] {self.info}", flush=True)

    def score(self, query: str, texts: list[str]) -> list[float]:
        import torch

        out: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            pairs = [[query, t] for t in texts[start:start + self.batch_size]]
            batch = self.tok(pairs, padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**batch).logits.view(-1).float()
            out.extend(logits.cpu().tolist())
        return out


def rerank(question: str, doc_ids: list[str], corpus: dict[str, Doc], reranker) -> list[tuple[str, float]]:
    """Score every candidate against the question (same title+context text that was embedded) and
    return (doc_id, score) sorted by score descending; ties keep the original retrieval order."""
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError(f"danh sách ứng viên có ID trùng: {doc_ids}")
    texts = [corpus[d].embed_text() for d in doc_ids]
    scores = reranker.score(question, texts)
    order = sorted(range(len(doc_ids)), key=lambda i: -scores[i])
    return [(doc_ids[i], float(scores[i])) for i in order]
