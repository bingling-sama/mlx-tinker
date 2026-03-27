"""Memory-efficient optimizers for MLX training.

Implements 8-bit AdamW with block-wise dynamic tree quantization of optimizer
state, following the bitsandbytes pattern (arXiv:2110.02861).

Key difference from linear int8: the quantization map uses logarithmically-
spaced levels that match the empirical distribution of Adam moments
(concentrated near zero with long tails), giving much better numerical
stability at higher learning rates.
"""

from __future__ import annotations

from typing import Callable, Union

import mlx.core as mx
import mlx.optimizers as optim

# Default block size for blockwise quantization.
# bitsandbytes uses 2048; we use 256 for smaller LoRA parameter tensors.
BLOCK_SIZE = 256


# ---------------------------------------------------------------------------
# Dynamic tree quantization map (bitsandbytes-compatible)
# ---------------------------------------------------------------------------


def create_dynamic_map(
    signed: bool = True, max_exponent_bits: int = 7, total_bits: int = 8
) -> list[float]:
    """Create a non-uniform quantization map following bitsandbytes.

    Generates 256 float values with logarithmic spacing that matches
    the distribution of Adam optimizer moments. Values are denser near
    zero and sparser at the tails.

    This is a direct port of bitsandbytes.functional.create_dynamic_map().

    Returns:
        Sorted list of 256 float quantization centroids.
    """
    data: list[float] = []
    non_sign_bits = total_bits - 1  # 7

    for i in range(max_exponent_bits):
        fraction_items = int(
            2 ** (i + non_sign_bits - max_exponent_bits) + 1
            if signed
            else 2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1,
        )

        # Linearly-spaced boundaries from 0.1 to 1.0
        boundaries = [0.1 + (1.0 - 0.1) * j / (fraction_items - 1) for j in range(fraction_items)]

        # Midpoints between consecutive boundaries
        means = [(boundaries[j] + boundaries[j + 1]) / 2.0 for j in range(len(boundaries) - 1)]

        # Scale by power of 10
        scale = 10.0 ** (-(max_exponent_bits - 1) + i)
        data.extend(scale * m for m in means)

        if signed:
            data.extend(-scale * m for m in means)

    # Special values
    data.append(0.0)
    data.append(1.0)

    # Pad to 256
    while len(data) < 2**total_bits:
        data.append(0.0)

    data.sort()
    return data


# Precompute the default signed dynamic map as an MLX array
_DYNAMIC_MAP: mx.array | None = None


def _get_dynamic_map() -> mx.array:
    """Get the cached 256-element dynamic quantization map."""
    global _DYNAMIC_MAP
    if _DYNAMIC_MAP is None:
        _DYNAMIC_MAP = mx.array(create_dynamic_map(signed=True), dtype=mx.float32)
    return _DYNAMIC_MAP


# ---------------------------------------------------------------------------
# Block-wise quantization with dynamic map
# ---------------------------------------------------------------------------


def _quantize_blockwise(
    tensor: mx.array,
    block_size: int = BLOCK_SIZE,
) -> tuple[mx.array, mx.array]:
    """Quantize float32 tensor using dynamic tree quantization.

    For each block:
    1. Find absmax and normalize to [-1, 1]
    2. Map each normalized value to nearest centroid in the dynamic map
       using iterative binary search (7 iterations for 256 bins)
    3. Store the centroid index (uint8) and absmax (float32)

    Returns (quantized_uint8, absmax_scales).
    """
    code = _get_dynamic_map()
    flat = tensor.reshape(-1)
    orig_size = flat.size

    # Pad to multiple of block_size
    pad_len = (-orig_size) % block_size
    if pad_len > 0:
        flat = mx.pad(flat, [(0, pad_len)])

    blocks = flat.reshape(-1, block_size)
    absmax = mx.max(mx.abs(blocks), axis=1)
    absmax = mx.maximum(absmax, 1e-12)

    # Normalize to [-1, 1]
    normalized = blocks / absmax[:, None]
    x = normalized.reshape(-1)

    # Binary search: 7 iterations to find nearest code index in [0, 255]
    # This mirrors the bitsandbytes CUDA dQuantize kernel
    pivot = mx.full(x.shape, 127, dtype=mx.int32)
    upper_pivot = mx.full(x.shape, 255, dtype=mx.int32)
    lower_pivot = mx.full(x.shape, 0, dtype=mx.int32)

    for step_size in [64, 32, 16, 8, 4, 2, 1]:
        val = code[pivot]
        go_up = x > val
        lower_pivot = mx.where(go_up, pivot, lower_pivot)
        upper_pivot = mx.where(go_up, upper_pivot, pivot)
        pivot = mx.where(go_up, pivot + step_size, pivot - step_size)
        pivot = mx.clip(pivot, 0, 255)

    # Pick nearest neighbor among pivot, lower_pivot, upper_pivot
    val = code[pivot]
    lower_val = code[lower_pivot]
    upper_val = code[upper_pivot]

    dist_pivot = mx.abs(x - val)
    dist_lower = mx.abs(x - lower_val)
    dist_upper = mx.abs(x - upper_val)

    best = pivot
    best_dist = dist_pivot
    use_lower = dist_lower < best_dist
    best = mx.where(use_lower, lower_pivot, best)
    best_dist = mx.where(use_lower, dist_lower, best_dist)
    use_upper = dist_upper < best_dist
    best = mx.where(use_upper, upper_pivot, best)

    quantized = best.astype(mx.uint8)[:orig_size]
    return quantized, absmax


def _dequantize_blockwise(
    quantized: mx.array,
    absmax: mx.array,
    block_size: int = BLOCK_SIZE,
) -> mx.array:
    """Dequantize uint8 tensor using dynamic map and per-block absmax scales."""
    code = _get_dynamic_map()
    flat = quantized.astype(mx.int32).reshape(-1)
    orig_size = flat.size

    pad_len = (-orig_size) % block_size
    if pad_len > 0:
        flat = mx.pad(flat, [(0, pad_len)])

    # Look up code values for each quantized index
    dequantized = code[flat].reshape(-1, block_size)

    # Scale by absmax
    dequantized = dequantized * absmax[:, None]
    return dequantized.reshape(-1)[:orig_size]


# ---------------------------------------------------------------------------
# 8-bit AdamW Optimizer
# ---------------------------------------------------------------------------


class AdamW8Bit(optim.Optimizer):
    r"""8-bit AdamW with dynamic tree quantization (bitsandbytes-compatible).

    Uses a non-uniform 256-level quantization map with logarithmic spacing
    that matches the empirical distribution of Adam moments. This provides
    ~75% optimizer state memory savings with better numerical stability
    than linear int8 quantization.

    The update rule is identical to standard AdamW:

    .. math::

        m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
        v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2 \\
        w_{t+1} &= w_t - \alpha (\frac{m_{t+1}}{\sqrt{v_{t+1}} + \epsilon}
                   + \lambda w_t)

    Moments m and v are stored as uint8 indices into the dynamic map
    between steps.

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

        # Store moments as uint8 indices into the dynamic map
        state["m"] = mx.zeros(flat_size, dtype=mx.uint8)
        state["v"] = mx.zeros(flat_size, dtype=mx.uint8)
        state["m_absmax"] = mx.zeros(n_blocks, dtype=mx.float32)
        state["v_absmax"] = mx.zeros(n_blocks, dtype=mx.float32)
        state["shape"] = parameter.shape

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict) -> mx.array:
        """Perform AdamW update with dynamic tree quantized state."""
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

        # Clamp v to avoid underflow after quantization roundtrip
        v = mx.maximum(v, 1e-30)

        # Adam step
        update = m / (mx.sqrt(v) + eps)

        # AdamW weight decay
        new_param = parameter * (1 - lr * self.weight_decay) - lr * update

        # Requantize moments using dynamic tree quantization
        state["m"], state["m_absmax"] = _quantize_blockwise(m.reshape(-1), self.block_size)
        state["v"], state["v_absmax"] = _quantize_blockwise(v.reshape(-1), self.block_size)

        return new_param
