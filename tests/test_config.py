import pytest

from vimedrag.config import Config, parse_args


def test_defaults_and_run_name():
    c = Config()
    assert (c.mode, c.top_k, c.n_eval, c.seed) == ("rag", 3, 200, 42)
    assert c.model_id == "vilm/vinallama-7b-chat"
    assert c.run_name == "rag_k3_vinallama7b"


def test_run_name_other_modes():
    assert Config(mode="closed").run_name == "closed_vinallama7b"
    assert Config(mode="oracle", model="vinallama2.7b").run_name == "oracle_vinallama27b"
    assert Config(exp_name="custom").run_name == "custom"


def test_invalid_values():
    with pytest.raises(ValueError):
        Config(mode="foo")
    with pytest.raises(ValueError):
        Config(model="gpt")
    with pytest.raises(ValueError):
        Config(top_k=11, retrieve_k=10)
    with pytest.raises(ValueError):
        Config(chunk="maybe")


def test_parse_args():
    assert parse_args([]).bertscore is True
    assert parse_args(["--dry-run"]).bertscore is False
    assert parse_args(["--dry-run", "--bertscore"]).bertscore is True
    c = parse_args(["--mode", "oracle", "--top-k", "5", "--n-eval", "20", "--fake-embedder"])
    assert (c.mode, c.top_k, c.n_eval, c.fake_embedder) == ("oracle", 5, 20, True)
    assert c.to_dict()["mode"] == "oracle"


def test_rerank_mode_config():
    c = Config(mode="rerank")
    assert c.run_name == "rag_rerank_top3_vinallama7b"
    assert c.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert (c.rerank_max_length, c.rerank_batch_size, c.candidates_from) == (512, 16, None)
    p = parse_args(["--mode", "rerank", "--candidates-from", "runs/rag_k3_vinallama7b/predictions.jsonl",
                    "--rerank-batch-size", "8"])
    assert (p.mode, p.candidates_from, p.rerank_batch_size) == ("rerank", "runs/rag_k3_vinallama7b/predictions.jsonl", 8)
