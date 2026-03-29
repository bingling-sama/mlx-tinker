from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch required for Qwen3.5 HF parity tests")
pytest.importorskip("transformers", reason="transformers required for Qwen3.5 HF parity tests")

from tests.qwen35_hf_helpers import (
    compute_parity_metrics,
    QWEN35_PROMPTS,
    QWEN35_UNIT_MODEL,
    load_hf_model,
    load_mlx_model,
    load_tokenizer,
    run_hf_qwen_parity_pass,
    run_mlx_qwen_parity_pass,
    selected_layer_ids,
    tokenize_prompt,
)


@pytest.fixture(scope="session")
def qwen35_0_8b_tokenizer():
    return load_tokenizer(QWEN35_UNIT_MODEL)


@pytest.fixture(scope="session")
def qwen35_0_8b_hf_bf16():
    return load_hf_model(QWEN35_UNIT_MODEL, quantized=False)


@pytest.fixture(scope="session")
def qwen35_0_8b_mlx_bf16():
    return load_mlx_model(QWEN35_UNIT_MODEL, quantized=False)


@pytest.fixture(scope="session")
def qwen35_0_8b_hf_4bit():
    return load_hf_model(QWEN35_UNIT_MODEL, quantized=True)


@pytest.fixture(scope="session")
def qwen35_0_8b_mlx_4bit():
    return load_mlx_model(QWEN35_UNIT_MODEL, quantized=True)


class TestQwen35UnitParity:
    @pytest.mark.parametrize("prompt", QWEN35_PROMPTS[:2], ids=["table", "bug-report"])
    def test_qwen35_0_8b_bf16_matches_hf(
        self,
        prompt,
        qwen35_0_8b_tokenizer,
        qwen35_0_8b_hf_bf16,
        qwen35_0_8b_mlx_bf16,
    ):
        token_ids = tokenize_prompt(qwen35_0_8b_tokenizer, prompt, max_length=32)
        layer_ids = selected_layer_ids(qwen35_0_8b_mlx_bf16)

        hf = run_hf_qwen_parity_pass(qwen35_0_8b_hf_bf16, token_ids, layer_ids=layer_ids)
        mlx = run_mlx_qwen_parity_pass(qwen35_0_8b_mlx_bf16, token_ids, layer_ids=layer_ids)
        metrics = compute_parity_metrics(hf, mlx, token_ids, layer_ids=layer_ids)

        for name, cosine in metrics["hidden_cosines"].items():
            floor = 0.99999 if name == "embed" else 0.9994
            assert cosine > floor, f"{name} cosine {cosine:.6f} <= {floor:.6f}"

        assert metrics["logits_cosine"] > 0.9995
        assert metrics["grad_cosine"] > 0.9997
        assert metrics["logprob_rmse"] < 0.12
        assert metrics["ce_rel_diff"] < 0.01

    @pytest.mark.parametrize("prompt", QWEN35_PROMPTS[:2], ids=["table", "bug-report"])
    def test_qwen35_0_8b_hf_4bit_tracks_hf_bf16(
        self,
        prompt,
        qwen35_0_8b_tokenizer,
        qwen35_0_8b_hf_bf16,
        qwen35_0_8b_hf_4bit,
        qwen35_0_8b_mlx_bf16,
    ):
        token_ids = tokenize_prompt(qwen35_0_8b_tokenizer, prompt, max_length=32)
        layer_ids = selected_layer_ids(qwen35_0_8b_mlx_bf16)

        hf_bf16 = run_hf_qwen_parity_pass(qwen35_0_8b_hf_bf16, token_ids, layer_ids=layer_ids)
        hf_4bit = run_hf_qwen_parity_pass(qwen35_0_8b_hf_4bit, token_ids, layer_ids=layer_ids)
        metrics = compute_parity_metrics(hf_bf16, hf_4bit, token_ids, layer_ids=layer_ids)

        floors = {
            "embed": 0.9999,
            f"layer_{layer_ids[0]}": 0.98,
            f"layer_{layer_ids[1]}": 0.96,
            f"layer_{layer_ids[2]}": 0.96,
            "final_norm": 0.95,
        }
        for name, floor in floors.items():
            assert metrics["hidden_cosines"][name] > floor

        assert metrics["logits_cosine"] > 0.96
        assert metrics["grad_cosine"] > 0.97
        assert metrics["logprob_rmse"] < 0.7
        assert metrics["ce_rel_diff"] < 0.02

    @pytest.mark.parametrize("prompt", QWEN35_PROMPTS[:2], ids=["table", "bug-report"])
    def test_qwen35_0_8b_mlx_4bit_tracks_mlx_bf16(
        self,
        prompt,
        qwen35_0_8b_tokenizer,
        qwen35_0_8b_mlx_bf16,
        qwen35_0_8b_mlx_4bit,
    ):
        token_ids = tokenize_prompt(qwen35_0_8b_tokenizer, prompt, max_length=32)
        layer_ids = selected_layer_ids(qwen35_0_8b_mlx_bf16)

        mlx_bf16 = run_mlx_qwen_parity_pass(qwen35_0_8b_mlx_bf16, token_ids, layer_ids=layer_ids)
        mlx_4bit = run_mlx_qwen_parity_pass(qwen35_0_8b_mlx_4bit, token_ids, layer_ids=layer_ids)
        metrics = compute_parity_metrics(mlx_bf16, mlx_4bit, token_ids, layer_ids=layer_ids)

        floors = {
            "embed": 0.995,
            f"layer_{layer_ids[0]}": 0.97,
            f"layer_{layer_ids[1]}": 0.95,
            f"layer_{layer_ids[2]}": 0.95,
            "final_norm": 0.94,
        }
        for name, floor in floors.items():
            assert metrics["hidden_cosines"][name] > floor

        assert metrics["logits_cosine"] > 0.95
        assert metrics["grad_cosine"] > 0.95
        assert metrics["logprob_rmse"] < 0.8
        assert metrics["ce_rel_diff"] < 0.11
