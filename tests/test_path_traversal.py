"""Tests for checkpoint path traversal protection in MLXBackend."""

import pytest

from mlx_tinker.backend.mlx_backend import MLXBackend
from mlx_tinker.config import EngineConfig


@pytest.fixture
def backend(tmp_path):
    ckpt_base = tmp_path / "checkpoints"
    ckpt_base.mkdir()
    config = EngineConfig(
        base_model="test-model",
        checkpoints_base=ckpt_base,
    )
    return MLXBackend(config)


def test_valid_path(backend):
    """A path within checkpoints_base should be accepted."""
    result = backend._validate_checkpoint_path(
        str(backend.config.checkpoints_base / "my_run" / "step_100")
    )
    assert result == (backend.config.checkpoints_base / "my_run" / "step_100").resolve()


def test_path_traversal_dotdot(backend):
    """A path using ../ to escape checkpoints_base should be rejected."""
    malicious = str(backend.config.checkpoints_base / ".." / "secrets")
    with pytest.raises(ValueError, match="outside the allowed"):
        backend._validate_checkpoint_path(malicious)


def test_absolute_path_outside(backend):
    """An absolute path outside checkpoints_base should be rejected."""
    with pytest.raises(ValueError, match="outside the allowed"):
        backend._validate_checkpoint_path("/etc/passwd")


def test_path_equals_base(backend):
    """A path exactly equal to checkpoints_base should be accepted."""
    result = backend._validate_checkpoint_path(
        str(backend.config.checkpoints_base)
    )
    assert result == backend.config.checkpoints_base.resolve()


def test_nested_valid_path(backend):
    """A deeply nested path within checkpoints_base should be accepted."""
    deep = backend.config.checkpoints_base / "a" / "b" / "c" / "d" / "checkpoint"
    result = backend._validate_checkpoint_path(str(deep))
    assert result == deep.resolve()


def test_relative_checkpoint_name_resolves_under_base(backend):
    """A bare checkpoint name should resolve under checkpoints_base."""
    result = backend._validate_checkpoint_path("step_0016")
    assert result == (backend.config.checkpoints_base / "step_0016").resolve()
