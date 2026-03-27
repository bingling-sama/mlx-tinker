"""Stress test: MLX vs HuggingFace inference logit equivalence.

Validates that MLX produces equivalent logits to HF transformers on Qwen3.5-9B.

Thresholds (fp16):
  - Cosine similarity > 0.999
  - Max abs diff < 0.01
  - KL divergence < 0.001
  - Top-5 agreement > 99%
  - Per-token log-prob RMSE < 0.005

Thresholds (4-bit quantized):
  - Cosine similarity > 0.98
  - Top-5 agreement > 92%
  - KL divergence < 0.05
  - Per-token RMSE < 0.1
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.stress.conftest import skip_insufficient_ram

pytestmark = [pytest.mark.stress, skip_insufficient_ram]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.sum(a * b)
    norm_a = np.sqrt(np.sum(a * a))
    norm_b = np.sqrt(np.sum(b * b))
    return float(dot / (norm_a * norm_b + 1e-10))


def _kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """Compute KL(P || Q) from logits."""
    p = _softmax(p_logits)
    q = _softmax(q_logits)
    # Clip to avoid log(0)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q)), axis=-1).mean())


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _top_k_agreement(a_logits: np.ndarray, b_logits: np.ndarray, k: int = 5) -> float:
    """Fraction of positions where the top-k tokens agree."""
    a_topk = set(np.argsort(a_logits, axis=-1)[:, -k:].flatten())
    b_topk = set(np.argsort(b_logits, axis=-1)[:, -k:].flatten())
    # Per-position agreement
    seq_len = a_logits.shape[0] if a_logits.ndim == 2 else 1
    agreements = 0
    for i in range(a_logits.shape[0]):
        a_top = set(np.argsort(a_logits[i])[-k:])
        b_top = set(np.argsort(b_logits[i])[-k:])
        if a_top == b_top:
            agreements += 1
    return agreements / a_logits.shape[0]


class TestFP16InferenceEquivalence:
    """Compare MLX fp16 logits vs HF fp16 logits."""

    def test_logit_equivalence(self, mlx_model, hf_model, shared_tokenizer, wikipedia_dataset):
        import mlx.core as mx
        import torch

        mlx_m, mlx_tok = mlx_model
        hf_m = hf_model

        cosine_sims = []
        max_abs_diffs = []
        kl_divs = []
        top5_agrees = []
        rmses = []

        num_samples = min(50, len(wikipedia_dataset))

        for text in wikipedia_dataset[:num_samples]:
            tokens = shared_tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=128)
            token_list = tokens[0].tolist()

            if len(token_list) < 5:
                continue

            # HF forward
            with torch.no_grad():
                hf_out = hf_m(tokens)
                hf_logits = hf_out.logits[0].float().numpy()  # [T, V]

            # MLX forward
            mlx_input = mx.array(token_list)[None, :]
            mlx_logits = mlx_m(mlx_input)
            mx.eval(mlx_logits)
            mlx_logits_np = np.array(mlx_logits[0])  # [T, V]

            # Ensure same shape
            min_len = min(hf_logits.shape[0], mlx_logits_np.shape[0])
            hf_l = hf_logits[:min_len]
            mlx_l = mlx_logits_np[:min_len]

            # Metrics
            for i in range(min_len):
                cosine_sims.append(_cosine_similarity(hf_l[i], mlx_l[i]))

            max_abs_diffs.append(np.max(np.abs(hf_l - mlx_l)))
            kl_divs.append(_kl_divergence(hf_l, mlx_l))
            top5_agrees.append(_top_k_agreement(hf_l, mlx_l, k=5))

            # RMSE on log probs
            hf_lp = np.log(_softmax(hf_l) + 1e-10)
            mlx_lp = np.log(_softmax(mlx_l) + 1e-10)
            # Get target token log probs (shift by 1)
            target_ids = token_list[1 : min_len + 1]
            if len(target_ids) == min_len:
                hf_target_lp = np.array([hf_lp[i, target_ids[i]] for i in range(min_len)])
                mlx_target_lp = np.array([mlx_lp[i, target_ids[i]] for i in range(min_len)])
                rmse = np.sqrt(np.mean((hf_target_lp - mlx_target_lp) ** 2))
                rmses.append(rmse)

        # Aggregate
        mean_cosine = np.mean(cosine_sims)
        mean_max_abs = np.mean(max_abs_diffs)
        mean_kl = np.mean(kl_divs)
        mean_top5 = np.mean(top5_agrees)
        mean_rmse = np.mean(rmses) if rmses else 0.0

        print(f"\n=== FP16 Inference Equivalence (n={num_samples}) ===")
        print(f"  Cosine similarity: mean={mean_cosine:.6f} min={np.min(cosine_sims):.6f}")
        print(f"  Max abs diff:      mean={mean_max_abs:.6f} max={np.max(max_abs_diffs):.6f}")
        print(f"  KL divergence:     mean={mean_kl:.6f} max={np.max(kl_divs):.6f}")
        print(f"  Top-5 agreement:   mean={mean_top5:.4f}")
        print(f"  Log-prob RMSE:     mean={mean_rmse:.6f}")

        assert mean_cosine > 0.999, f"Cosine sim {mean_cosine} < 0.999"
        assert np.max(max_abs_diffs) < 0.01, f"Max abs diff {np.max(max_abs_diffs)} > 0.01"
        assert mean_kl < 0.001, f"KL divergence {mean_kl} > 0.001"
        assert mean_top5 > 0.99, f"Top-5 agreement {mean_top5} < 0.99"
        assert mean_rmse < 0.005, f"RMSE {mean_rmse} > 0.005"


class TestQuantizedInferenceEquivalence:
    """Compare 4-bit quantized MLX logits vs HF fp16 logits (relaxed thresholds)."""

    def test_quantized_logit_equivalence(
        self, hf_model, shared_tokenizer, wikipedia_dataset, model_name
    ):
        import mlx.core as mx
        import mlx.nn as nn
        import torch
        from mlx_lm import load

        # Load and quantize
        mlx_m, _ = load(model_name)
        nn.quantize(mlx_m, bits=4, group_size=64)

        hf_m = hf_model

        cosine_sims = []
        kl_divs = []
        top5_agrees = []
        rmses = []

        num_samples = min(30, len(wikipedia_dataset))

        for text in wikipedia_dataset[:num_samples]:
            tokens = shared_tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=64)
            token_list = tokens[0].tolist()

            if len(token_list) < 5:
                continue

            with torch.no_grad():
                hf_out = hf_m(tokens)
                hf_logits = hf_out.logits[0].float().numpy()

            mlx_input = mx.array(token_list)[None, :]
            mlx_logits = mlx_m(mlx_input)
            mx.eval(mlx_logits)
            mlx_logits_np = np.array(mlx_logits[0])

            min_len = min(hf_logits.shape[0], mlx_logits_np.shape[0])
            hf_l = hf_logits[:min_len]
            mlx_l = mlx_logits_np[:min_len]

            for i in range(min_len):
                cosine_sims.append(_cosine_similarity(hf_l[i], mlx_l[i]))

            kl_divs.append(_kl_divergence(hf_l, mlx_l))
            top5_agrees.append(_top_k_agreement(hf_l, mlx_l, k=5))

            hf_lp = np.log(_softmax(hf_l) + 1e-10)
            mlx_lp = np.log(_softmax(mlx_l) + 1e-10)
            target_ids = token_list[1 : min_len + 1]
            if len(target_ids) == min_len:
                hf_target_lp = np.array([hf_lp[i, target_ids[i]] for i in range(min_len)])
                mlx_target_lp = np.array([mlx_lp[i, target_ids[i]] for i in range(min_len)])
                rmses.append(np.sqrt(np.mean((hf_target_lp - mlx_target_lp) ** 2)))

        mean_cosine = np.mean(cosine_sims)
        mean_kl = np.mean(kl_divs)
        mean_top5 = np.mean(top5_agrees)
        mean_rmse = np.mean(rmses) if rmses else 0.0

        print(f"\n=== 4-bit Quantized Inference Equivalence (n={num_samples}) ===")
        print(f"  Cosine similarity: mean={mean_cosine:.6f}")
        print(f"  KL divergence:     mean={mean_kl:.6f}")
        print(f"  Top-5 agreement:   mean={mean_top5:.4f}")
        print(f"  Log-prob RMSE:     mean={mean_rmse:.6f}")

        assert mean_cosine > 0.98, f"Cosine sim {mean_cosine} < 0.98"
        assert mean_kl < 0.05, f"KL divergence {mean_kl} > 0.05"
        assert mean_top5 > 0.92, f"Top-5 agreement {mean_top5} < 0.92"
        assert mean_rmse < 0.1, f"RMSE {mean_rmse} > 0.1"
