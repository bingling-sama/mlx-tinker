"""Shared test utilities: tiny models, datum helpers, finite-difference gradient checking."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_tinker.types import (
    Datum,
    EncodedTextChunk,
    LossFnInputs,
    ModelInput,
    TensorData,
)


# ---------------------------------------------------------------------------
# Tiny models (deduplicated from test_backend_training, test_optimizers, etc.)
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Minimal model for testing: embedding + linear head."""

    def __init__(self, vocab_size: int = 32, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embed(x)
        return self.head(h)


class TinyAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)


class TinyMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim * 2, bias=False)
        self.up_proj = nn.Linear(dim, dim * 2, bias=False)
        self.down_proj = nn.Linear(dim * 2, dim, bias=False)


class TinyTransformerLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = TinyAttention(dim)
        self.mlp = TinyMLP(dim)


class TinyModelInner(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_layers: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = [TinyTransformerLayer(dim) for _ in range(num_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embed_tokens(x)
        # Pass through all sub-layers so LoRA gradients flow
        for layer in self.layers:
            attn = layer.self_attn
            # Simplified attention: sum of all projections (exercises all LoRA targets)
            q = attn.q_proj(h)
            k = attn.k_proj(h)
            v = attn.v_proj(h)
            h = h + attn.o_proj(q + k + v) * 0.1
            # Simplified MLP: gate_proj * up_proj → down_proj
            h = h + layer.mlp.down_proj(layer.mlp.gate_proj(h) * layer.mlp.up_proj(h)) * 0.1
        return h


class TinyLM(nn.Module):
    """Minimal LM matching Qwen/Llama architecture: .model + .lm_head.

    Used for LoRA and chunked-CE testing (has _has_split_lm_head() == True).
    Uses dim=64 so weight matrices are divisible by group_size=32.
    """

    def __init__(self, vocab_size: int = 128, dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.model = TinyModelInner(vocab_size, dim, num_layers)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.model(x)
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers


class TinyLayer(nn.Module):
    """Layer compatible with cache-based inference (for TinyModelWithCache)."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def __call__(self, x, cache=None):
        return self.linear(x), cache


class TinyModelWithCache(nn.Module):
    """Minimal model compatible with mlx-lm generate_step (cache support)."""

    def __init__(self, vocab_size: int = 32, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = [TinyLayer(dim)]
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        h = self.embed(x)
        if cache is not None:
            for i, layer in enumerate(self.layers):
                h, cache[i] = layer(h, cache[i])
        else:
            for layer in self.layers:
                h, _ = layer(h)
        return self.head(h)


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text):
        return [ord(c) % 32 for c in text]

    def decode(self, tokens):
        return "".join(chr((t % 26) + 65) for t in tokens)


# ---------------------------------------------------------------------------
# Datum helper
# ---------------------------------------------------------------------------


def make_datum(
    tokens: list[int],
    targets: list[int],
    weights: list[float],
    advantages: list[float] | None = None,
    logprobs: list[float] | None = None,
) -> Datum:
    """Create a Datum for testing."""
    n = len(targets)
    return Datum(
        model_input=ModelInput(chunks=[EncodedTextChunk(tokens=tokens)]),
        loss_fn_inputs=LossFnInputs(
            target_tokens=TensorData(data=targets),
            weights=TensorData(data=weights),
            advantages=TensorData(data=advantages if advantages else [0.0] * n),
            logprobs=TensorData(data=logprobs if logprobs else [0.0] * n),
        ),
    )


# ---------------------------------------------------------------------------
# Finite-difference gradient checking
# ---------------------------------------------------------------------------


def finite_difference_check(
    fn,
    x: mx.array,
    epsilon: float = 1e-4,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> None:
    """Verify mx.grad(fn)(x) matches (fn(x+ε) - fn(x-ε)) / 2ε element-wise.

    Raises AssertionError with diagnostics on failure.
    """
    # Analytical gradient
    grad_fn = mx.grad(fn)
    analytical = grad_fn(x)
    mx.eval(analytical)
    analytical_np = np.array(analytical).flatten()

    # Numerical gradient (central difference)
    x_flat = np.array(x).flatten()
    numerical = np.zeros_like(x_flat)

    for i in range(len(x_flat)):
        x_plus = x_flat.copy()
        x_plus[i] += epsilon
        x_minus = x_flat.copy()
        x_minus[i] -= epsilon

        f_plus = fn(mx.array(x_plus.reshape(x.shape)))
        f_minus = fn(mx.array(x_minus.reshape(x.shape)))
        mx.eval(f_plus, f_minus)

        numerical[i] = (f_plus.item() - f_minus.item()) / (2 * epsilon)

    # Compare
    abs_diff = np.abs(analytical_np - numerical)
    denom = np.maximum(np.abs(analytical_np), np.abs(numerical))
    denom = np.where(denom == 0, 1.0, denom)
    rel_diff = abs_diff / denom

    failures = (abs_diff > atol) & (rel_diff > rtol)
    if np.any(failures):
        worst_idx = np.argmax(abs_diff)
        raise AssertionError(
            f"Finite-difference check failed at {np.sum(failures)}/{len(x_flat)} elements.\n"
            f"  Worst: idx={worst_idx} analytical={analytical_np[worst_idx]:.6e} "
            f"numerical={numerical[worst_idx]:.6e} "
            f"abs_diff={abs_diff[worst_idx]:.6e} rel_diff={rel_diff[worst_idx]:.6e}\n"
            f"  Max abs diff: {np.max(abs_diff):.6e}, Max rel diff: {np.max(rel_diff):.6e}"
        )


def finite_difference_check_argnum(
    fn,
    args: tuple,
    argnum: int,
    epsilon: float = 1e-4,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> None:
    """Finite-difference check for multi-argument functions.

    Differentiates fn w.r.t. args[argnum], holding other args fixed.
    """

    def single_arg_fn(x):
        new_args = list(args)
        new_args[argnum] = x
        return fn(*new_args)

    finite_difference_check(single_arg_fn, args[argnum], epsilon=epsilon, rtol=rtol, atol=atol)
