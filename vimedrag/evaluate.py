from __future__ import annotations

import re
import unicodedata
from collections import Counter
from statistics import mean
from typing import Callable

from vimedrag.data import base_doc_id

REFUSAL_PHRASE = "tôi không tìm thấy thông tin"
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)


def tokenize_vi(s: str | None) -> list[str]:
    s = unicodedata.normalize("NFC", s or "").lower()
    return _NON_WORD.sub(" ", s).split()


def _lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def _f1(overlap: int, n_pred: int, n_ref: int) -> float:
    if overlap == 0:
        return 0.0
    p, r = overlap / n_pred, overlap / n_ref
    return 2 * p * r / (p + r)


def rouge_l_f1(pred: str, ref: str) -> float:
    p, r = tokenize_vi(pred), tokenize_vi(ref)
    if not p or not r:
        return 1.0 if p == r else 0.0
    return _f1(_lcs_len(p, r), len(p), len(r))


def token_f1(pred: str, ref: str) -> float:
    p, r = tokenize_vi(pred), tokenize_vi(ref)
    if not p or not r:
        return 1.0 if p == r else 0.0
    overlap = sum((Counter(p) & Counter(r)).values())
    return _f1(overlap, len(p), len(r))


def recall_at_k(retrieved_ids: list[str], gold: str, k: int) -> float:
    return float(any(base_doc_id(d) == gold for d in retrieved_ids[:k]))


def mrr_at_k(retrieved_ids: list[str], gold: str, k: int) -> float:
    for rank, d in enumerate(retrieved_ids[:k], 1):
        if base_doc_id(d) == gold:
            return 1.0 / rank
    return 0.0


def is_refusal(pred: str) -> bool:
    return REFUSAL_PHRASE in " ".join(tokenize_vi(pred))


def _avg(values: list[float]):
    return round(mean(values), 4) if values else None


def _summarize(group: list[dict]) -> dict:
    keys = sorted({k for s in group for k in s if k != "topic"})
    return {"n": len(group), **{k: _avg([s[k] for s in group if k in s]) for k in keys}}


def aggregate(preds: list[dict], ks=(1, 3, 5), mrr_k: int = 10,
              bertscore_fn: Callable[[list[str], list[str]], list[float]] | None = None) -> dict:
    rows = list(preds)
    scored = []
    for r in rows:
        s = {
            "topic": r["topic"],
            "rougeL": rouge_l_f1(r["prediction"], r["gold_answer"]),
            "token_f1": token_f1(r["prediction"], r["gold_answer"]),
            "refusal": float(is_refusal(r["prediction"])),
            "error": float(bool(r.get("error"))),
        }
        retrieved = r.get("retrieved") or []
        if retrieved:
            ids = [x["doc_id"] for x in retrieved]
            for k in ks:
                s[f"recall@{k}"] = recall_at_k(ids, r["gold_doc_id"], k)
            s[f"mrr@{mrr_k}"] = mrr_at_k(ids, r["gold_doc_id"], mrr_k)
        scored.append(s)
    if bertscore_fn is not None and rows:
        values = bertscore_fn([r["prediction"] for r in rows], [r["gold_answer"] for r in rows])
        for s, v in zip(scored, values):
            s["bertscore_f1"] = float(v)
    topics = sorted({s["topic"] for s in scored})
    return {
        "overall": _summarize(scored),
        "by_topic": {t: _summarize([s for s in scored if s["topic"] == t]) for t in topics},
    }


def make_bertscore_fn(lang: str = "vi") -> Callable[[list[str], list[str]], list[float]]:
    from bert_score import score

    def fn(preds: list[str], refs: list[str]) -> list[float]:
        cands = [p if p.strip() else "." for p in preds]
        _, _, f1 = score(cands, refs, lang=lang, verbose=False)
        return [0.0 if not p.strip() else float(v) for p, v in zip(preds, f1.tolist())]

    return fn
