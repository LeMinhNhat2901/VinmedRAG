from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

MODELS = {
    "vinallama7b": "vilm/vinallama-7b-chat",
    "vinallama2.7b": "vilm/vinallama-2.7b-chat",
}
MODES = ("closed", "rag", "oracle")
CHUNK_MODES = ("auto", "on", "off")


@dataclass
class Config:
    mode: str = "rag"
    top_k: int = 3
    retrieve_k: int = 10
    n_eval: int = 200
    seed: int = 42
    model: str = "vinallama7b"
    embed_model: str = "BAAI/bge-m3"
    dataset: str = "tmnam20/ViMedAQA"
    dataset_config: str = "all"
    data_dir: str | None = None
    index_dir: str = "artifacts/index"
    runs_dir: str = "runs"
    exp_name: str | None = None
    max_new_tokens: int = 256
    max_prompt_tokens: int = 3000
    repetition_penalty: float = 1.1
    bertscore: bool = True
    dry_run: bool = False
    dry_run_docs: int = 500
    dry_run_questions: int = 8
    chunk: str = "auto"
    fake_embedder: bool = False

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode phải thuộc {MODES}, nhận {self.mode!r}")
        if self.model not in MODELS:
            raise ValueError(f"model phải thuộc {list(MODELS)}, nhận {self.model!r}")
        if self.chunk not in CHUNK_MODES:
            raise ValueError(f"chunk phải thuộc {CHUNK_MODES}, nhận {self.chunk!r}")
        if not 1 <= self.top_k <= self.retrieve_k:
            raise ValueError(f"cần 1 <= top_k <= retrieve_k, nhận top_k={self.top_k}, retrieve_k={self.retrieve_k}")

    @property
    def model_id(self) -> str:
        return MODELS[self.model]

    @property
    def run_name(self) -> str:
        if self.exp_name:
            return self.exp_name
        model = self.model.replace(".", "")
        if self.mode == "rag":
            return f"{self.mode}_k{self.top_k}_{model}"
        return f"{self.mode}_{model}"

    def to_dict(self) -> dict:
        return asdict(self)


def parse_args(argv: list[str] | None = None) -> Config:
    p = argparse.ArgumentParser(description="ViMedAQA baseline RAG")
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--top-k", type=int)
    p.add_argument("--retrieve-k", type=int)
    p.add_argument("--n-eval", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--model", choices=list(MODELS))
    p.add_argument("--embed-model")
    p.add_argument("--dataset")
    p.add_argument("--dataset-config")
    p.add_argument("--data-dir")
    p.add_argument("--index-dir")
    p.add_argument("--runs-dir")
    p.add_argument("--exp-name")
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--max-prompt-tokens", type=int)
    p.add_argument("--repetition-penalty", type=float)
    p.add_argument("--bertscore", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--dry-run", action="store_true", default=None)
    p.add_argument("--dry-run-docs", type=int)
    p.add_argument("--dry-run-questions", type=int)
    p.add_argument("--chunk", choices=CHUNK_MODES)
    p.add_argument("--fake-embedder", action="store_true", default=None)
    args = p.parse_args(argv)
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    if "bertscore" not in kwargs:
        kwargs["bertscore"] = not kwargs.get("dry_run", False)
    return Config(**kwargs)
