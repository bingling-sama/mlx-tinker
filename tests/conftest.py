"""Root conftest: shared fixtures and markers for the mlx-tinker test suite."""

from __future__ import annotations

import os

import mlx.core as mx
import pytest


@pytest.fixture
def reset_memory():
    """Reset MLX peak memory counter before a test."""
    mx.reset_peak_memory()
    yield


skip_on_ci = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="Timing-sensitive test, skip on CI",
)
