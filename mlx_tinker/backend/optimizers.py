"""Memory-efficient optimizers for MLX training.

Implements 8-bit AdamW with block-wise dynamic quantization of optimizer
state (first and second moments), following the bitsandbytes/Unsloth pattern.
"""

from __future__ import annotations

from typing import Callable, Union

import mlx.core as mx
import mlx.optimizers as optim

BLOCK_SIZE = 256


def _quantize_blockwise(
    tensor: mx.array,
    block_size: int = BLOCK_SIZE,
) -> tuple[mx.array, mx.array]:
    """Quantize float32 tensor to int8 with per-block absmax scaling.

    Returns (quantized_int8, absmax_scales).
    """
    flat = tensor.reshape(-1)
    orig_size = flat.size

    # Pad to multiple of block_size
    pad_len = (-orig_size) % block_size
    if pad_len > 0:
        flat = mx.pad(flat, [(0, pad_len)])

    blocks = flat.reshape(-1, block_size)
    absmax = mx.max(mx.abs(blocks), axis=1)
    absmax = mx.maximum(absmax, 1e-12)

    quantized = mx.round(blocks / absmax[:, None] * 127.0).astype(mx.int8)
    return quantized.reshape(-1)[:orig_size], absmax


def _dequantize_blockwise(
    quantized: mx.array,
    absmax: mx.array,
    block_size: int = BLOCK_SIZE,
) -> mx.array:
    """Dequantize int8 tensor using per-block absmax scales."""
    flat = quantized.astype(mx.float32).reshape(-1)
    orig_size = flat.size

    pad_len = (-orig_size) % block_size
    if pad_len > 0:
        flat = mx.pad(flat, [(0, pad_len)])

    blocks = flat.reshape(-1, block_size)
    dequantized = blocks * (absmax[:, None] / 127.0)
    return dequantized.reshape(-1)[:orig_size]


class AdamW8Bit(optim.Optimizer):
    r"""8-bit AdamW optimizer with block-wise dynamic quantization.

    Quantizes first (m) and second (v) moment estimates to int8,
    with per-block absmax scaling. Saves ~75% optimizer state memory
    compared to fp32 AdamW.

    The update rule is identical to standard AdamW:

    .. math::

        m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
        v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2 \\
        w_{t+1} &= w_t - \alpha (\frac{m_{t+1}}{\sqrt{v_{t+1}} + \epsilon} + \lambda w_t)

    The only difference is that m and v are stored as int8 between steps.

    Args:
        learning_rate: Learning rate.
        betas: Coefficients for running averages of gradient and its square.
        eps: Numerical stability term.
        weight_decay: Decoupled weight decay.
        block_size: Number of elements per quantization block.
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]] = 1e-5,
        betas: list[float] | tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        block_size: int = BLOCK_SIZE,
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.betas = tuple(betas)
        self.eps = eps
        self.weight_decay = weight_decay
        self.block_size = block_size

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize quantized optimizer state for a single parameter."""
        flat_size = parameter.size
        n_blocks = (flat_size + self.block_size - 1) // self.block_size

        state["m"] = mx.zeros(flat_size, dtype=mx.int8)
        state["v"] = mx.zeros(flat_size, dtype=mx.int8)
        state["m_absmax"] = mx.zeros(n_blocks, dtype=mx.float32)
        state["v_absmax"] = mx.zeros(n_blocks, dtype=mx.float32)
        state["shape"] = parameter.shape

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict) -> mx.array:
        """Perform AdamW update with 8-bit state quantization."""
        lr = self.learning_rate.astype(gradient.dtype)
        b1, b2 = self.betas
        eps = self.eps
        shape = state["shape"]

        # Dequantize moments
        m = _dequantize_blockwise(state["m"], state["m_absmax"], self.block_size)
        v = _dequantize_blockwise(state["v"], state["v_absmax"], self.block_size)
        m = m[: parameter.size].reshape(shape)
        v = v[: parameter.size].reshape(shape)

        # Standard Adam moment update
        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * mx.square(gradient)

        # Clamp v to avoid underflow after int8 roundtrip
        v = mx.maximum(v, 1e-30)

        # Adam step
        update = m / (mx.sqrt(v) + eps)

        # AdamW weight decay
        new_param = parameter * (1 - lr * self.weight_decay) - lr * update

        # Requantize moments
        state["m"], state["m_absmax"] = _quantize_blockwise(m.reshape(-1), self.block_size)
        state["v"], state["v_absmax"] = _quantize_blockwise(v.reshape(-1), self.block_size)

        return new_param
