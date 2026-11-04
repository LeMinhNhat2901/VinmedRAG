from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vimedrag.index import LoadedIndex


@dataclass
class Hit:
    doc_id: str
    score: float
    rank: int


class DenseRetriever:
    def __init__(self, index: LoadedIndex, embedder):
        self.index = index
        self.embedder = embedder

    def search(self, question: str, k: int) -> list[Hit]:
        q = self.embedder.encode([question])[0].astype(np.float32)
        scores = self.index.embeddings @ q
        k = min(k, len(scores))
        top = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.index.doc_ids[i], float(scores[i]), rank) for rank, i in enumerate(top, 1)]
