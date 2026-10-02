"""Bidirectional mapping between tinker:// virtual URIs and local filesystem paths.

Tinker path specification:
- Training checkpoint:
    tinker://<model_id>/weights/<checkpoint_id>
- Sampler checkpoint:
    tinker://<model_id>/sampler_weights/<checkpoint_id>

Mapped filesystem location under checkpoints_base:
    checkpoints_base / <model_id> / <checkpoint_id>
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from mlx_tinker.types import CheckpointType

TINKER_PREFIX = "tinker://"
WEIGHTS_PREFIX = "weights/"
SAMPLER_WEIGHTS_PREFIX = "sampler_weights/"


def is_tinker_path(path: str) -> bool:
    """Check if the given string is a tinker:// virtual URI."""
    return path.startswith(TINKER_PREFIX)


def parse_tinker_path(tinker_path: str) -> tuple[str, CheckpointType, str]:
    """Parse a tinker:// URI into (model_id, checkpoint_type, checkpoint_name).

    Examples:
        tinker://run-1/weights/step-1 -> ('run-1', CheckpointType.TRAINING, 'step-1')
        tinker://run-1/sampler_weights/export-1 -> ('run-1', CheckpointType.SAMPLER, 'export-1')

    Raises:
        ValueError: If the URI format is invalid.
    """
    if not is_tinker_path(tinker_path):
        raise ValueError(f"Invalid tinker path (must start with '{TINKER_PREFIX}'): {tinker_path}")

    remainder = tinker_path[len(TINKER_PREFIX):]
    parts = remainder.split("/", 2)
    if len(parts) < 3 or not parts[0] or not parts[2]:
        raise ValueError(
            f"Invalid tinker path format: '{tinker_path}'. "
            f"Expected 'tinker://<model_id>/weights/<checkpoint_id>' or "
            f"'tinker://<model_id>/sampler_weights/<checkpoint_id>'"
        )

    model_id = parts[0]
    category = parts[1]
    checkpoint_name = parts[2]

    if category == "weights":
        ckpt_type = CheckpointType.TRAINING
    elif category == "sampler_weights":
        ckpt_type = CheckpointType.SAMPLER
    else:
        raise ValueError(
            f"Invalid checkpoint category '{category}' in tinker path: '{tinker_path}'. "
            f"Expected 'weights' or 'sampler_weights'."
        )

    return model_id, ckpt_type, checkpoint_name


def format_tinker_path(
    model_id: str,
    checkpoint_id: str,
    checkpoint_type: CheckpointType | Literal["training", "sampler"] = CheckpointType.TRAINING,
) -> str:
    """Format components into a standard tinker:// URI.

    If checkpoint_id already includes 'weights/' or 'sampler_weights/' prefix,
    it is normalized so double-prefixes are avoided.

    Examples:
        format_tinker_path("m1", "step-1", CheckpointType.TRAINING)
        -> "tinker://m1/weights/step-1"

        format_tinker_path("m1", "weights/step-1", CheckpointType.TRAINING)
        -> "tinker://m1/weights/step-1"

        format_tinker_path("m1", "export-1", CheckpointType.SAMPLER)
        -> "tinker://m1/sampler_weights/export-1"
    """
    ckpt_type_str = checkpoint_type.value if isinstance(checkpoint_type, CheckpointType) else checkpoint_type
    category = "weights" if ckpt_type_str == "training" else "sampler_weights"

    # Strip category prefix if already present on checkpoint_id
    clean_id = checkpoint_id
    if clean_id.startswith("weights/"):
        clean_id = clean_id[len("weights/"):]
    elif clean_id.startswith("sampler_weights/"):
        clean_id = clean_id[len("sampler_weights/"):]
    elif clean_id.startswith("sampler/"):
        clean_id = clean_id[len("sampler/"):]

    return f"{TINKER_PREFIX}{model_id}/{category}/{clean_id}"


def tinker_path_to_relative_path(tinker_path: str) -> tuple[str, CheckpointType]:
    """Convert a tinker:// path to relative disk path under checkpoints_base.

    Returns:
        tuple of (relative_path_string, checkpoint_type)

    Examples:
        "tinker://run-1/weights/step-1" -> ("run-1/step-1", CheckpointType.TRAINING)
        "tinker://run-1/sampler_weights/export-1" -> ("run-1/sampler/export-1", CheckpointType.SAMPLER)
    """
    model_id, ckpt_type, checkpoint_name = parse_tinker_path(tinker_path)
    # For sampler weights, we keep the existing convention under run-1/sampler/...
    if ckpt_type == CheckpointType.SAMPLER:
        if not checkpoint_name.startswith("sampler/"):
            rel_path = f"{model_id}/sampler/{checkpoint_name}"
        else:
            rel_path = f"{model_id}/{checkpoint_name}"
    else:
        rel_path = f"{model_id}/{checkpoint_name}"
    return rel_path, ckpt_type


def relative_path_to_tinker_path(
    relative_path: str,
    checkpoint_type: CheckpointType | Literal["training", "sampler"] | None = None,
) -> str:
    """Infer and construct a tinker:// URI from a path relative to checkpoints_base.

    If checkpoint_type is None, it is inferred from directory structure:
    if the second path component is 'sampler', it is treated as SAMPLER.
    Otherwise, it defaults to TRAINING.
    """
    parts = Path(relative_path).parts
    if not parts:
        raise ValueError(f"Empty path cannot be converted to tinker URI: '{relative_path}'")

    model_id = parts[0]
    if len(parts) == 1:
        # e.g. "step_0016" -> treat as model_id="step_0016", checkpoint_id="default"?
        # Or relative to model?
        raise ValueError(f"Path '{relative_path}' has no checkpoint sub-path")

    if parts[1] == "sampler":
        inferred_type = CheckpointType.SAMPLER
        checkpoint_name = "/".join(parts[2:]) if len(parts) > 2 else "default"
    elif parts[1] == "sampler_weights":
        inferred_type = CheckpointType.SAMPLER
        checkpoint_name = "/".join(parts[2:]) if len(parts) > 2 else "default"
    elif parts[1] == "weights":
        inferred_type = CheckpointType.TRAINING
        checkpoint_name = "/".join(parts[2:]) if len(parts) > 2 else "default"
    else:
        inferred_type = CheckpointType.TRAINING
        checkpoint_name = "/".join(parts[1:])

    actual_type = checkpoint_type or inferred_type
    return format_tinker_path(model_id, checkpoint_name, actual_type)
