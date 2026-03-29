"""Stress test: MLX dynamic tree quantization vs bitsandbytes reference.

Verifies that create_dynamic_map() produces identical values to the
bitsandbytes CUDA implementation, and that optimizer update magnitudes
are comparable.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.stress.conftest import skip_insufficient_ram

pytestmark = [pytest.mark.stress, skip_insufficient_ram]


class TestDynamicMapEquivalence:
    """Compare MLX create_dynamic_map() against bitsandbytes reference."""

    def test_dynamic_map_matches_bitsandbytes(self):
        """Signed dynamic map should match bitsandbytes element-wise."""
        try:
            import bitsandbytes.functional as bnb_F
        except ImportError:
            pytest.skip("bitsandbytes not installed")

        from mlx_tinker.backend.optimizers import create_dynamic_map

        mlx_map = np.array(create_dynamic_map(signed=True))
        bnb_map = bnb_F.create_dynamic_map(signed=True).numpy()

        assert len(mlx_map) == len(bnb_map) == 256, (
            f"Map sizes differ: MLX={len(mlx_map)} vs bnb={len(bnb_map)}"
        )

        # Both should be sorted
        assert np.all(np.diff(mlx_map) >= 0), "MLX map not sorted"
        assert np.all(np.diff(bnb_map) >= 0), "bnb map not sorted"

        # Element-wise comparison
        max_diff = np.max(np.abs(mlx_map - bnb_map))
        print(f"\n=== Dynamic Map Comparison (signed) ===")
        print(f"  Max abs diff: {max_diff:.2e}")
        print(f"  MLX range: [{mlx_map.min():.6f}, {mlx_map.max():.6f}]")
        print(f"  bnb range: [{bnb_map.min():.6f}, {bnb_map.max():.6f}]")

        assert max_diff < 1e-6, f"Dynamic maps differ by {max_diff:.2e}"

    def test_dynamic_map_unsigned_matches(self):
        """Unsigned dynamic map should also match bitsandbytes."""
        try:
            import bitsandbytes.functional as bnb_F
        except ImportError:
            pytest.skip("bitsandbytes not installed")

        from mlx_tinker.backend.optimizers import create_dynamic_map

        mlx_map = np.array(create_dynamic_map(signed=False))
        bnb_map = bnb_F.create_dynamic_map(signed=False).numpy()

        assert len(mlx_map) == len(bnb_map) == 256
        max_diff = np.max(np.abs(mlx_map - bnb_map))

        print(f"\n=== Dynamic Map Comparison (unsigned) ===")
        print(f"  Max abs diff: {max_diff:.2e}")

        assert max_diff < 1e-6, f"Unsigned maps differ by {max_diff:.2e}"


class TestOptimizerUpdateEquivalence:
    """Compare AdamW8Bit update magnitudes against bitsandbytes."""

    def test_single_step_update_direction(self):
        """One optimizer step should produce similar update direction."""
        try:
            import bitsandbytes as bnb
            import torch
        except ImportError:
            pytest.skip("bitsandbytes or torch not installed")

        import mlx.core as mx
        import mlx.nn as nn

        from mlx_tinker.backend.optimizers import AdamW8Bit

        # Create identical parameters
        mx.random.seed(42)
        torch.manual_seed(42)

        dim = 64
        param_np = np.random.randn(dim, dim).astype(np.float32)
        grad_np = np.random.randn(dim, dim).astype(np.float32) * 0.1

        # MLX optimizer
        mlx_param = mx.array(param_np.copy())
        mlx_grad = mx.array(grad_np.copy())

        mlx_model = nn.Linear(dim, dim, bias=False)
        mlx_model.weight = mlx_param
        mx.eval(mlx_model.parameters())

        mlx_opt = AdamW8Bit(learning_rate=1e-3)
        mlx_opt.init(mlx_model.trainable_parameters())
        mlx_opt.update(mlx_model, {"weight": mlx_grad})
        mx.eval(mlx_model.parameters(), mlx_opt.state)

        mlx_new = np.array(mlx_model.weight)
        mlx_update = mlx_new - param_np

        # bitsandbytes optimizer
        torch_param = torch.tensor(param_np.copy(), requires_grad=True)
        torch_param.grad = torch.tensor(grad_np.copy())

        bnb_opt = bnb.optim.AdamW8bit(
            [torch_param], lr=1e-3, betas=(0.9, 0.999), eps=1e-8
        )
        bnb_opt.step()

        bnb_new = torch_param.detach().numpy()
        bnb_update = bnb_new - param_np

        # Compare update directions via cosine similarity
        mlx_flat = mlx_update.flatten()
        bnb_flat = bnb_update.flatten()

        cos_sim = np.dot(mlx_flat, bnb_flat) / (
            np.linalg.norm(mlx_flat) * np.linalg.norm(bnb_flat) + 1e-10
        )

        # Compare update magnitudes
        mag_ratio = np.linalg.norm(mlx_flat) / (np.linalg.norm(bnb_flat) + 1e-10)

        print(f"\n=== Single-Step Optimizer Update Comparison ===")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  Magnitude ratio:   {mag_ratio:.6f}")
        print(f"  MLX update norm:   {np.linalg.norm(mlx_flat):.6e}")
        print(f"  bnb update norm:   {np.linalg.norm(bnb_flat):.6e}")

        assert cos_sim > 0.95, f"Update direction cosine {cos_sim:.4f} < 0.95"
        assert 0.5 < mag_ratio < 2.0, f"Magnitude ratio {mag_ratio:.4f} not in [0.5, 2.0]"
