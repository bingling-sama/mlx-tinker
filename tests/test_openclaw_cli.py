from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mlx_tinker.openclaw.cli import (
    DEFAULT_MODEL_LARGE,
    DEFAULT_MODEL_SMALL,
    DEFAULT_MODEL_ID,
    DEFAULT_PROVIDER_ID,
    ManagedState,
    build_paths,
    build_provider_config,
    build_gateway_remote_url,
    ensure_git_checkout,
    merge_string_list_item,
    recommended_model_for_memory,
    save_state,
    load_state,
)


def test_recommended_model_for_memory_prefers_4b_at_24gb() -> None:
    assert recommended_model_for_memory(24 * 1024**3) == DEFAULT_MODEL_LARGE
    assert recommended_model_for_memory(16 * 1024**3) == DEFAULT_MODEL_SMALL
    assert recommended_model_for_memory(None) == DEFAULT_MODEL_SMALL


def test_build_provider_config_uses_stable_provider_and_model_alias() -> None:
    config = build_provider_config(
        provider_id=DEFAULT_PROVIDER_ID,
        model_id=DEFAULT_MODEL_ID,
        proxy_port=30000,
    )
    assert config["baseUrl"] == "http://host.docker.internal:30000/v1"
    assert config["models"][0]["id"] == DEFAULT_MODEL_ID
    assert config["models"][0]["name"] == "MLX Tinker Local"


def test_build_gateway_remote_url_targets_loopback_port() -> None:
    assert build_gateway_remote_url(18859) == "ws://127.0.0.1:18859"


def test_merge_string_list_item_appends_once_and_preserves_existing_entries() -> None:
    assert merge_string_list_item(["exec", "message"], "message") == ["exec", "message"]
    assert merge_string_list_item(["exec", " process "], "message") == ["exec", "process", "message"]


def test_state_round_trip(tmp_path: Path) -> None:
    paths = build_paths(tmp_path)
    paths.managed_dir.mkdir(parents=True, exist_ok=True)
    state = ManagedState(
        model="Qwen/Qwen3.5-0.8B",
        backend_port=8010,
        proxy_port=30000,
        gateway_port=18789,
        bridge_port=18790,
        gateway_bind="loopback",
        gateway_token="token",
        docker_image="openclaw:local",
        config_backup="backup.json",
        previous_primary_model="openai/gpt-5.2",
    )
    save_state(paths, state)
    loaded = load_state(paths)
    assert loaded == state
    assert json.loads(paths.state_path.read_text(encoding="utf-8"))["model"] == state.model


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def make_origin_repo(tmp_path: Path) -> tuple[Path, str]:
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-b", "main")
    git(origin, "config", "user.name", "Test User")
    git(origin, "config", "user.email", "test@example.com")
    (origin / "README.md").write_text("base\n", encoding="utf-8")
    git(origin, "add", "README.md")
    git(origin, "commit", "-m", "initial")
    git(origin, "checkout", "-b", "feature")
    (origin / "README.md").write_text("feature\n", encoding="utf-8")
    git(origin, "commit", "-am", "feature update")
    return origin, git(origin, "rev-parse", "HEAD")


def test_ensure_git_checkout_clones_missing_repo(tmp_path: Path) -> None:
    origin, head_sha = make_origin_repo(tmp_path)
    checkout = tmp_path / "checkout"

    resolved_sha = ensure_git_checkout(
        repo_dir=checkout,
        repo_url=str(origin),
        ref="feature",
        label="OpenClaw-RL",
    )

    assert checkout.joinpath(".git").exists()
    assert resolved_sha == head_sha
    assert git(checkout, "rev-parse", "HEAD") == head_sha
    assert git(checkout, "branch", "--show-current") == "feature"


def test_ensure_git_checkout_fast_forwards_clean_repo(tmp_path: Path) -> None:
    origin, first_sha = make_origin_repo(tmp_path)
    checkout = tmp_path / "checkout"
    ensure_git_checkout(
        repo_dir=checkout,
        repo_url=str(origin),
        ref="feature",
        label="OpenClaw-RL",
    )

    git(origin, "checkout", "feature")
    (origin / "README.md").write_text("feature v2\n", encoding="utf-8")
    git(origin, "commit", "-am", "feature update 2")
    second_sha = git(origin, "rev-parse", "HEAD")

    resolved_sha = ensure_git_checkout(
        repo_dir=checkout,
        repo_url=str(origin),
        ref="feature",
        label="OpenClaw-RL",
    )

    assert first_sha != second_sha
    assert resolved_sha == second_sha
    assert git(checkout, "rev-parse", "HEAD") == second_sha


def test_ensure_git_checkout_rejects_dirty_repo(tmp_path: Path) -> None:
    origin, _ = make_origin_repo(tmp_path)
    checkout = tmp_path / "checkout"
    ensure_git_checkout(
        repo_dir=checkout,
        repo_url=str(origin),
        ref="feature",
        label="OpenClaw-RL",
    )
    (checkout / "README.md").write_text("dirty\n", encoding="utf-8")

    try:
        ensure_git_checkout(
            repo_dir=checkout,
            repo_url=str(origin),
            ref="feature",
            label="OpenClaw-RL",
        )
    except RuntimeError as exc:
        assert "dirty" in str(exc)
    else:
        raise AssertionError("expected dirty checkout to be rejected")
