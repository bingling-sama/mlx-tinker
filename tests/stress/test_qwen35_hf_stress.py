from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch required for Qwen3.5 HF stress tests")
pytest.importorskip("transformers", reason="transformers required for Qwen3.5 HF stress tests")

from tests.qwen35_hf_helpers import (
    compute_parity_metrics,
    QWEN35_PROMPTS,
    QWEN35_STRESS_MODEL,
    load_hf_model,
    load_mlx_model,
    load_tokenizer,
    run_hf_qwen_parity_pass,
    run_mlx_qwen_parity_pass,
    selected_layer_ids,
    tokenize_prompt,
)
from tests.stress.conftest import skip_insufficient_ram

pytestmark = [pytest.mark.stress, skip_insufficient_ram]


@pytest.fixture(scope="session")
def qwen35_4b_tokenizer():
    return load_tokenizer(QWEN35_STRESS_MODEL)


@pytest.fixture(scope="session")
def qwen35_4b_hf_bf16():
    return load_hf_model(QWEN35_STRESS_MODEL, quantized=False)


@pytest.fixture(scope="session")
def qwen35_4b_mlx_bf16():
    return load_mlx_model(QWEN35_STRESS_MODEL, quantized=False)


@pytest.fixture(scope="session")
def qwen35_4b_hf_4bit():
    return load_hf_model(QWEN35_STRESS_MODEL, quantized=True)


@pytest.fixture(scope="session")
def qwen35_4b_mlx_4bit():
    return load_mlx_model(QWEN35_STRESS_MODEL, quantized=True)


def _collect_metrics(tokenizer, reference_runner, candidate_runner, *, layer_ids: list[int], max_length: int):
    hidden_cosines: dict[str, list[float]] = {
        "embed": [],
        **{f"layer_{idx}": [] for idx in layer_ids},
        "final_norm": [],
    }
    logits_cosines: list[float] = []
    grad_cosines: list[float] = []
    logprob_rmses: list[float] = []
    top5_scores: list[float] = []
    ce_rel_diffs: list[float] = []

    for prompt in QWEN35_PROMPTS:
        token_ids = tokenize_prompt(tokenizer, prompt, max_length=max_length)
        reference = reference_runner(token_ids)
        candidate = candidate_runner(token_ids)
        metrics = compute_parity_metrics(reference, candidate, token_ids, layer_ids=layer_ids)

        for name, value in metrics["hidden_cosines"].items():
            hidden_cosines[name].append(value)

        logits_cosines.append(metrics["logits_cosine"])
        grad_cosines.append(metrics["grad_cosine"])
        logprob_rmses.append(metrics["logprob_rmse"])
        top5_scores.append(metrics["top5_exact"])
        ce_rel_diffs.append(metrics["ce_rel_diff"])

    return {
        **{f"min_{name}_cosine": float(min(values)) for name, values in hidden_cosines.items()},
        "min_logits_cosine": float(min(logits_cosines)),
        "min_grad_cosine": float(min(grad_cosines)),
        "max_logprob_rmse": float(max(logprob_rmses)),
        "min_top5": float(min(top5_scores)),
        "max_ce_rel_diff": float(max(ce_rel_diffs)),
    }


class TestQwen35StressParity:
    def test_qwen35_4b_bf16_matches_hf(
        self,
        qwen35_4b_tokenizer,
        qwen35_4b_hf_bf16,
        qwen35_4b_mlx_bf16,
    ):
        layer_ids = selected_layer_ids(qwen35_4b_mlx_bf16)
        metrics = _collect_metrics(
            qwen35_4b_tokenizer,
            lambda ids: run_hf_qwen_parity_pass(qwen35_4b_hf_bf16, ids, layer_ids=layer_ids),
            lambda ids: run_mlx_qwen_parity_pass(qwen35_4b_mlx_bf16, ids, layer_ids=layer_ids),
            layer_ids=layer_ids,
            max_length=24,
        )

        assert metrics["min_embed_cosine"] > 0.99999
        for idx in layer_ids:
            assert metrics[f"min_layer_{idx}_cosine"] > 0.9994
        assert metrics["min_final_norm_cosine"] > 0.9994
        assert metrics["min_logits_cosine"] > 0.9995
        assert metrics["min_grad_cosine"] > 0.9997
        assert metrics["max_logprob_rmse"] < 0.12
        assert metrics["max_ce_rel_diff"] < 0.02

    def test_qwen35_4b_hf_4bit_tracks_hf_bf16(
        self,
        qwen35_4b_tokenizer,
        qwen35_4b_hf_bf16,
        qwen35_4b_hf_4bit,
        qwen35_4b_mlx_bf16,
    ):
        layer_ids = selected_layer_ids(qwen35_4b_mlx_bf16)
        metrics = _collect_metrics(
            qwen35_4b_tokenizer,
            lambda ids: run_hf_qwen_parity_pass(qwen35_4b_hf_bf16, ids, layer_ids=layer_ids),
            lambda ids: run_hf_qwen_parity_pass(qwen35_4b_hf_4bit, ids, layer_ids=layer_ids),
            layer_ids=layer_ids,
            max_length=24,
        )

        assert metrics["min_embed_cosine"] > 0.9999
        for idx in layer_ids:
            assert metrics[f"min_layer_{idx}_cosine"] > 0.94
        assert metrics["min_final_norm_cosine"] > 0.93
        assert metrics["min_logits_cosine"] > 0.94
        assert metrics["min_grad_cosine"] > 0.95
        assert metrics["max_logprob_rmse"] < 1.0
        assert metrics["max_ce_rel_diff"] < 0.05

    def test_qwen35_4b_mlx_4bit_tracks_mlx_bf16(
        self,
        qwen35_4b_tokenizer,
        qwen35_4b_mlx_bf16,
        qwen35_4b_mlx_4bit,
    ):
        layer_ids = selected_layer_ids(qwen35_4b_mlx_bf16)
        metrics = _collect_metrics(
            qwen35_4b_tokenizer,
            lambda ids: run_mlx_qwen_parity_pass(qwen35_4b_mlx_bf16, ids, layer_ids=layer_ids),
            lambda ids: run_mlx_qwen_parity_pass(qwen35_4b_mlx_4bit, ids, layer_ids=layer_ids),
            layer_ids=layer_ids,
            max_length=24,
        )

        assert metrics["min_embed_cosine"] > 0.995
        for idx in layer_ids:
            assert metrics[f"min_layer_{idx}_cosine"] > 0.93
        assert metrics["min_final_norm_cosine"] > 0.92
        assert metrics["min_logits_cosine"] > 0.93
        assert metrics["min_grad_cosine"] > 0.93
        assert metrics["max_logprob_rmse"] < 1.0
        assert metrics["max_ce_rel_diff"] < 0.15
