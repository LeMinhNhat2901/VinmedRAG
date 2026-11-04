from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_WS = re.compile(r"\s+")


@dataclass
class Doc:
    doc_id: str
    title: str
    text: str
    topic: str

    def embed_text(self) -> str:
        return f"{self.title}\n{self.text}"


@dataclass
class EvalItem:
    question_idx: str
    question: str
    answer: str
    topic: str
    gold_doc_id: str


def normalize_text(s: str | None) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", s or "")).strip()


def make_doc_id(context: str) -> str:
    return hashlib.sha1(normalize_text(context).encode("utf-8")).hexdigest()[:12]


def base_doc_id(doc_id: str) -> str:
    return doc_id.split("#", 1)[0]


def build_corpus(rows: Iterable[dict]) -> list[Doc]:
    seen: dict[str, Doc] = {}
    for r in rows:
        ctx = normalize_text(r.get("context"))
        if not ctx:
            continue
        doc_id = make_doc_id(ctx)
        if doc_id not in seen:
            seen[doc_id] = Doc(doc_id, normalize_text(r.get("title")), ctx, str(r["topic"]))
    return list(seen.values())


def sample_eval(rows: Iterable[dict], n: int, seed: int, corpus_ids: set[str]) -> list[EvalItem]:
    by_topic: dict[str, list[EvalItem]] = {}
    for r in rows:
        ctx = normalize_text(r.get("context"))
        if not ctx:
            continue
        item = EvalItem(
            question_idx=str(r["question_idx"]),
            question=normalize_text(r["question"]),
            answer=normalize_text(r["answer"]),
            topic=str(r["topic"]),
            gold_doc_id=make_doc_id(ctx),
        )
        by_topic.setdefault(item.topic, []).append(item)
    topics = sorted(by_topic)
    if not topics:
        return []
    rng = random.Random(seed)
    per, rem = divmod(n, len(topics))
    chosen: list[EvalItem] = []
    for i, topic in enumerate(topics):
        pool = sorted(by_topic[topic], key=lambda x: x.question_idx)
        want = per + (1 if i < rem else 0)
        if want > len(pool):
            raise ValueError(f"topic {topic!r} chỉ có {len(pool)} câu, cần {want}")
        chosen.extend(rng.sample(pool, want))
    # Validate only the chosen items: sampling must not depend on the corpus, so a
    # dry-run index (subset of docs) selects the same questions as the full index.
    for item in chosen:
        if item.gold_doc_id not in corpus_ids:
            raise ValueError(
                f"gold doc của {item.question_idx} không có trong corpus/index. "
                "Index được dựng với dataset/seed/n khác? Hãy dựng lại index."
            )
    return sorted(chosen, key=lambda x: x.question_idx)


def chunk_docs(docs: list[Doc], max_words: int = 300, overlap: int = 40) -> list[Doc]:
    out: list[Doc] = []
    step = max_words - overlap
    for d in docs:
        words = d.text.split(" ")
        if len(words) <= max_words:
            out.append(d)
            continue
        for i, start in enumerate(range(0, len(words), step)):
            out.append(Doc(f"{d.doc_id}#{i}", d.title, " ".join(words[start:start + max_words]), d.topic))
            if start + max_words >= len(words):
                break
    return out


def load_rows(dataset: str, config: str, data_dir: str | None = None) -> dict[str, list[dict]]:
    from datasets import load_dataset

    if data_dir:
        files = {}
        for split in ("train", "validation", "test"):
            found = sorted(str(p) for p in Path(data_dir).rglob(f"{split}-*.parquet"))
            if found:
                files[split] = found
        if not files:
            raise FileNotFoundError(f"Không tìm thấy file parquet dạng '<split>-*.parquet' trong {data_dir}")
        ds = load_dataset("parquet", data_files=files)
    else:
        ds = load_dataset(dataset, config)
    out: dict[str, list[dict]] = {}
    for split in ds:
        feature = ds[split].features.get("topic")
        names = getattr(feature, "names", None)
        rows = []
        for r in ds[split]:
            r = dict(r)
            if names is not None and isinstance(r.get("topic"), int):
                r["topic"] = names[r["topic"]]
            rows.append(r)
        out[split] = rows
    return out


def save_jsonl(path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows, bad = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        warnings.warn(f"Bỏ qua {bad} dòng JSON hỏng trong {path}")
    return rows


def check_or_write_eval_ids(path, ids: list[str]) -> None:
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != list(ids):
            raise ValueError(
                f"eval_ids khác với lần chạy trước ({path}). Dataset/corpus đã đổi; "
                "dùng runs-dir mới hoặc xoá file này nếu cố ý."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(ids), ensure_ascii=False, indent=0), encoding="utf-8")
