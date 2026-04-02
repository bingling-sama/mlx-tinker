from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

PROJECT_DIR = Path(__file__).resolve().parents[2]
OPENCLAW_REPO_URL = "https://github.com/openclaw/openclaw"
DEFAULT_OPENCLAW_RL_REPO_URL = "https://github.com/ojus1/OpenClaw-RL.git"
DEFAULT_OPENCLAW_RL_REF = "codex/qwen35-openclaw-tinker"
DEFAULT_PROVIDER_ID = "mlx-tinker-local"
DEFAULT_MODEL_ID = "local-primary"
DEFAULT_GATEWAY_PORT = 18789
DEFAULT_PROXY_PORT = 30000
DEFAULT_BACKEND_PORT = 8010
DEFAULT_GATEWAY_BIND = "lan"
DEFAULT_GATEWAY_IMAGE = "openclaw:local"
DEFAULT_PROXY_API_KEY = "tml-local"
DEFAULT_MAX_CONTEXT_TOKENS = 8192
DEFAULT_MAX_STEPS = 1000000
DEFAULT_BATCH_SIZE = 1
DEFAULT_MODEL_SMALL = "Qwen/Qwen3.5-0.8B"
DEFAULT_MODEL_LARGE = "Qwen/Qwen3.5-4B"
DEFAULT_WORKSPACE_CONFIG_PATH = "~/.openclaw/workspace"


@dataclass
class OpenClawPaths:
    home_dir: Path
    config_dir: Path
    config_path: Path
    workspace_dir: Path
    managed_dir: Path
    runtime_dir: Path
    logs_dir: Path
    records_dir: Path
    checkpoints_dir: Path
    db_path: Path
    state_path: Path
    extensions_dir: Path
    plugin_dst_dir: Path
    openclaw_dir: Path
    openclaw_rl_dir: Path
    openclaw_plugin_src_dir: Path
    local_chat_plugin_src_dir: Path
    local_chat_plugin_dst_dir: Path
    openclaw_compose_env_path: Path


@dataclass
class ManagedState:
    model: str
    backend_port: int
    proxy_port: int
    gateway_port: int
    bridge_port: int
    gateway_bind: str
    gateway_token: str
    docker_image: str
    provider_id: str = DEFAULT_PROVIDER_ID
    model_id: str = DEFAULT_MODEL_ID
    backend_pid: int | None = None
    proxy_pid: int | None = None
    setup_mode: str = "new"
    config_backup: str | None = None
    previous_primary_model: str | None = None

    @property
    def model_ref(self) -> str:
        return f"{self.provider_id}/{self.model_id}"


def recommended_model_for_memory(total_memory_bytes: int | None) -> str:
    if total_memory_bytes is not None and total_memory_bytes >= 24 * 1024**3:
        return DEFAULT_MODEL_LARGE
    return DEFAULT_MODEL_SMALL


def build_provider_config(*, provider_id: str, model_id: str, proxy_port: int) -> dict[str, Any]:
    return {
        "baseUrl": f"http://host.docker.internal:{proxy_port}/v1",
        "apiKey": DEFAULT_PROXY_API_KEY,
        "api": "openai-completions",
        "models": [
            {
                "id": model_id,
                "name": "MLX Tinker Local",
                "reasoning": False,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 32768,
                "maxTokens": 4096,
            }
        ],
    }


def build_gateway_remote_url(port: int) -> str:
    return f"ws://127.0.0.1:{port}"


def merge_string_list_item(existing: list[str] | None, item: str) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in existing or []:
        normalized = value.strip()
        if normalized and normalized not in seen:
            merged.append(normalized)
            seen.add(normalized)
    normalized_item = item.strip()
    if normalized_item and normalized_item not in seen:
        merged.append(normalized_item)
    return merged


def build_paths(home_dir: Path | None = None) -> OpenClawPaths:
    resolved_home = (home_dir or Path.home()).expanduser().resolve()
    config_dir = resolved_home / ".openclaw"
    managed_dir = config_dir / "mlx-tinker"
    return OpenClawPaths(
        home_dir=resolved_home,
        config_dir=config_dir,
        config_path=config_dir / "openclaw.json",
        workspace_dir=config_dir / "workspace",
        managed_dir=managed_dir,
        runtime_dir=managed_dir / "runtime",
        logs_dir=managed_dir / "logs",
        records_dir=managed_dir / "records",
        checkpoints_dir=managed_dir / "checkpoints",
        db_path=managed_dir / "run.db",
        state_path=managed_dir / "state.json",
        extensions_dir=config_dir / "extensions",
        plugin_dst_dir=config_dir / "extensions" / "rl-training-headers",
        openclaw_dir=PROJECT_DIR / ".external" / "openclaw",
        openclaw_rl_dir=PROJECT_DIR / ".external" / "openclaw-rl",
        openclaw_plugin_src_dir=PROJECT_DIR
        / ".external"
        / "openclaw-rl"
        / "extensions"
        / "rl-training-headers",
        local_chat_plugin_src_dir=PROJECT_DIR / "extensions" / "local-chat-guardrails",
        local_chat_plugin_dst_dir=config_dir / "extensions" / "local-chat-guardrails",
        openclaw_compose_env_path=managed_dir / "openclaw-docker.env",
    )


def ensure_directories(paths: OpenClawPaths) -> None:
    for path in [
        paths.config_dir,
        paths.workspace_dir,
        paths.managed_dir,
        paths.runtime_dir,
        paths.logs_dir,
        paths.records_dir,
        paths.checkpoints_dir,
        paths.extensions_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_state(paths: OpenClawPaths) -> ManagedState | None:
    if not paths.state_path.exists():
        return None
    return ManagedState(**json.loads(paths.state_path.read_text(encoding="utf-8")))


def save_state(paths: OpenClawPaths, state: ManagedState) -> None:
    paths.state_path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def run_command(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=capture_output,
    )
    if check and result.returncode != 0:
        output = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Command failed ({' '.join(cmd)}):\n{output}")
    return result


def require_command(name: str) -> None:
    if shutil.which(name):
        return
    raise RuntimeError(f"Missing required dependency: {name}")


def resolve_openclaw_entry(paths: OpenClawPaths) -> Path:
    for relative in ("dist/index.js", "dist/index.mjs"):
        candidate = paths.openclaw_dir / relative
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"OpenClaw CLI entrypoint not found under {paths.openclaw_dir}/dist. "
        "Expected dist/index.js or dist/index.mjs."
    )


def openclaw_cli_env(paths: OpenClawPaths) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(paths.home_dir)
    env["OPENCLAW_HOME"] = str(paths.home_dir)
    env["OPENCLAW_CONFIG_PATH"] = str(paths.config_path)
    return env


def get_config_value(paths: OpenClawPaths, path: str) -> str | None:
    result = run_command(
        ["node", str(resolve_openclaw_entry(paths)), "config", "get", path],
        env=openclaw_cli_env(paths),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    if value in {"", "undefined", "null"}:
        return None
    return value


def set_config_value(paths: OpenClawPaths, path: str, value: Any, *, strict_json: bool = True) -> None:
    serialized = value if isinstance(value, str) and not strict_json else json.dumps(value)
    cmd = ["node", str(resolve_openclaw_entry(paths)), "config", "set", path, serialized]
    if strict_json:
        cmd.append("--strict-json")
    run_command(cmd, env=openclaw_cli_env(paths))


def get_string_list_config_value(paths: OpenClawPaths, path: str) -> list[str]:
    raw = get_config_value(paths, path)
    if raw is None:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, str)]


def ensure_string_list_config_value(paths: OpenClawPaths, path: str, item: str) -> None:
    merged = merge_string_list_item(get_string_list_config_value(paths, path), item)
    set_config_value(paths, path, merged)


def detect_total_memory_bytes() -> int | None:
    if sys.platform != "darwin":
        return None
    result = run_command(["sysctl", "-n", "hw.memsize"], capture_output=True, check=False)
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    try:
        return int(raw)
    except ValueError:
        return None


def git_stdout(args: list[str], *, cwd: Path | None = None) -> str:
    result = run_command(["git", *args], cwd=cwd, capture_output=True)
    return result.stdout.strip()


def ensure_git_checkout(*, repo_dir: Path, repo_url: str, ref: str, label: str) -> str:
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if not repo_dir.joinpath(".git").exists():
        run_command(
            ["git", "clone", "--branch", ref, "--single-branch", repo_url, str(repo_dir)],
        )
        return git_stdout(["rev-parse", "HEAD"], cwd=repo_dir)

    if git_stdout(["status", "--porcelain"], cwd=repo_dir):
        raise RuntimeError(
            f"Existing {label} checkout is dirty: {repo_dir}\n"
            "Commit, stash, or reset changes before continuing."
        )

    run_command(["git", "fetch", repo_url, ref], cwd=repo_dir)
    fetched_sha = git_stdout(["rev-parse", "FETCH_HEAD"], cwd=repo_dir)
    current_sha = git_stdout(["rev-parse", "HEAD"], cwd=repo_dir)

    if current_sha == fetched_sha:
        return current_sha

    ancestry = run_command(
        ["git", "merge-base", "--is-ancestor", current_sha, fetched_sha],
        cwd=repo_dir,
        check=False,
    )
    if ancestry.returncode == 0:
        run_command(["git", "checkout", "-B", ref, fetched_sha], cwd=repo_dir)
        return fetched_sha

    raise RuntimeError(
        f"Existing {label} checkout diverges from {repo_url}#{ref}\n"
        "Refusing to overwrite clean but divergent history."
    )


def bootstrap_openclaw_checkout(paths: OpenClawPaths) -> None:
    external_dir = paths.openclaw_dir.parent
    external_dir.mkdir(parents=True, exist_ok=True)
    if paths.openclaw_dir.joinpath(".git").exists():
        return
    run_command(["git", "clone", "--depth=1", OPENCLAW_REPO_URL, str(paths.openclaw_dir)])


def bootstrap_openclaw_rl_checkout(paths: OpenClawPaths) -> None:
    ensure_git_checkout(
        repo_dir=paths.openclaw_rl_dir,
        repo_url=DEFAULT_OPENCLAW_RL_REPO_URL,
        ref=DEFAULT_OPENCLAW_RL_REF,
        label="OpenClaw-RL",
    )


def ensure_openclaw_image(image: str, *, openclaw_dir: Path) -> None:
    inspect = run_command(["docker", "image", "inspect", image], capture_output=True, check=False)
    if inspect.returncode == 0:
        return
    if image == DEFAULT_GATEWAY_IMAGE:
        run_command(["docker", "build", "-t", image, "-f", "Dockerfile", "."], cwd=openclaw_dir)
        return
    run_command(["docker", "pull", image])


def create_config_backup(paths: OpenClawPaths) -> str | None:
    if not paths.config_path.exists():
        return None
    backup_dir = paths.managed_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"openclaw.{stamp}.json"
    shutil.copy2(paths.config_path, backup_path)
    return str(backup_path)


def copy_plugin(src_dir: Path, dst_dir: Path, *, missing_message: str) -> None:
    if not src_dir.exists():
        raise RuntimeError(missing_message)
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def copy_managed_plugins(paths: OpenClawPaths) -> None:
    copy_plugin(
        paths.openclaw_plugin_src_dir,
        paths.plugin_dst_dir,
        missing_message=(
            f"RL training headers plugin not found at {paths.openclaw_plugin_src_dir}. "
            "OpenClaw-RL bootstrap may have failed."
        ),
    )
    copy_plugin(
        paths.local_chat_plugin_src_dir,
        paths.local_chat_plugin_dst_dir,
        missing_message=(
            f"Local chat guardrails plugin not found at {paths.local_chat_plugin_src_dir}. "
            "mlx-tinker install may be incomplete."
        ),
    )


def write_compose_env(paths: OpenClawPaths, state: ManagedState) -> None:
    values = {
        "OPENCLAW_CONFIG_DIR": str(paths.config_dir),
        "OPENCLAW_WORKSPACE_DIR": str(paths.workspace_dir),
        "OPENCLAW_GATEWAY_PORT": str(state.gateway_port),
        "OPENCLAW_BRIDGE_PORT": str(state.bridge_port),
        "OPENCLAW_GATEWAY_BIND": state.gateway_bind,
        "OPENCLAW_GATEWAY_TOKEN": state.gateway_token,
        "OPENCLAW_IMAGE": state.docker_image,
        "MLX_TINKER_API_KEY": DEFAULT_PROXY_API_KEY,
    }
    lines = [f"{key}={value}" for key, value in values.items()]
    paths.openclaw_compose_env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_state_from_args(
    paths: OpenClawPaths,
    args: argparse.Namespace,
    existing_state: ManagedState | None,
) -> ManagedState:
    current_primary = get_config_value(paths, "agents.defaults.model.primary")
    current_token = get_config_value(paths, "gateway.auth.token")
    current_bind = get_config_value(paths, "gateway.bind")
    config_exists = paths.config_path.exists()
    selected_model = args.model or (existing_state.model if existing_state else None)
    if not selected_model:
        selected_model = recommended_model_for_memory(detect_total_memory_bytes())

    if args.gateway_bind:
        gateway_bind = args.gateway_bind
    else:
        gateway_bind = existing_state.gateway_bind if existing_state else None
        if gateway_bind not in {"loopback", "lan"}:
            gateway_bind = None
        if gateway_bind is None or gateway_bind == "loopback":
            gateway_bind = DEFAULT_GATEWAY_BIND

    token = current_token or (existing_state.gateway_token if existing_state else None) or secrets.token_hex(32)
    gateway_port = args.gateway_port or (existing_state.gateway_port if existing_state else DEFAULT_GATEWAY_PORT)
    bridge_port = args.bridge_port or (existing_state.bridge_port if existing_state else gateway_port + 1)
    backend_port = args.backend_port or (existing_state.backend_port if existing_state else DEFAULT_BACKEND_PORT)
    proxy_port = args.proxy_port or (existing_state.proxy_port if existing_state else DEFAULT_PROXY_PORT)
    docker_image = args.docker_image or (existing_state.docker_image if existing_state else DEFAULT_GATEWAY_IMAGE)

    previous_primary = existing_state.previous_primary_model if existing_state else None
    if current_primary and current_primary != f"{DEFAULT_PROVIDER_ID}/{DEFAULT_MODEL_ID}":
        previous_primary = current_primary

    return ManagedState(
        model=selected_model,
        backend_port=backend_port,
        proxy_port=proxy_port,
        gateway_port=gateway_port,
        bridge_port=bridge_port,
        gateway_bind=gateway_bind,
        gateway_token=token,
        docker_image=docker_image,
        setup_mode="existing" if config_exists else "new",
        config_backup=existing_state.config_backup if existing_state else None,
        previous_primary_model=previous_primary,
        backend_pid=existing_state.backend_pid if existing_state else None,
        proxy_pid=existing_state.proxy_pid if existing_state else None,
    )


def configure_openclaw(paths: OpenClawPaths, state: ManagedState) -> None:
    provider_config = build_provider_config(
        provider_id=state.provider_id,
        model_id=state.model_id,
        proxy_port=state.proxy_port,
    )

    set_config_value(paths, "gateway.mode", "local")
    set_config_value(paths, "gateway.port", state.gateway_port)
    set_config_value(paths, "gateway.bind", state.gateway_bind)
    set_config_value(paths, "gateway.auth.mode", "token")
    set_config_value(paths, "gateway.auth.token", state.gateway_token)
    set_config_value(paths, "gateway.remote.url", build_gateway_remote_url(state.gateway_port))
    set_config_value(paths, "gateway.remote.token", state.gateway_token)

    if state.setup_mode == "new":
        set_config_value(paths, "gateway.controlUi.allowInsecureAuth", True)
        set_config_value(paths, "gateway.controlUi.dangerouslyDisableDeviceAuth", True)
        set_config_value(paths, "gateway.controlUi.dangerouslyAllowHostHeaderOriginFallback", True)

    set_config_value(paths, "agents.defaults.workspace", DEFAULT_WORKSPACE_CONFIG_PATH)

    set_config_value(paths, "models.mode", "merge")
    set_config_value(paths, f"models.providers.{state.provider_id}", provider_config)
    set_config_value(paths, "agents.defaults.model.primary", state.model_ref)
    set_config_value(paths, "plugins.entries.rl-training-headers.enabled", True)
    set_config_value(paths, "plugins.entries.local-chat-guardrails.enabled", True)
    # Local WebChat/CLI sessions are the primary onboarding path. Disabling the
    # proactive message tool keeps smaller local models from trying to send
    # outbound channel messages instead of answering inline.
    ensure_string_list_config_value(paths, "tools.deny", "message")


def is_process_alive(pid: int | None, expected_fragment: str | None = None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    if not expected_fragment:
        return True
    result = run_command(["ps", "-p", str(pid), "-o", "command="], capture_output=True, check=False)
    if result.returncode != 0:
        return False
    return expected_fragment in result.stdout


def stop_pid(pid: int | None, *, expected_fragment: str) -> None:
    if not is_process_alive(pid, expected_fragment):
        return
    assert pid is not None
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not is_process_alive(pid, expected_fragment):
            return
        time.sleep(1)
    os.kill(pid, signal.SIGKILL)


def wait_for_http(url: str, *, headers: dict[str, str] | None = None, timeout_seconds: int = 120) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        request = Request(url, headers=headers or {})
        try:
            with urlopen(request, timeout=5) as response:
                if 200 <= response.status < 300:
                    return True
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(2)
    return False


def backend_log_path(paths: OpenClawPaths) -> Path:
    return paths.logs_dir / "mlx-tinker.log"


def proxy_log_path(paths: OpenClawPaths) -> Path:
    return paths.logs_dir / "openclaw-rl.log"


def start_backend(paths: OpenClawPaths, state: ManagedState) -> int:
    if is_process_alive(state.backend_pid, "mlx_tinker"):
        return state.backend_pid or 0
    log_file = backend_log_path(paths).open("a", encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "mlx_tinker",
        "--model",
        state.model,
        "--host",
        "0.0.0.0",
        "--port",
        str(state.backend_port),
        "--db",
        str(paths.db_path),
        "--checkpoints",
        str(paths.checkpoints_dir),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_DIR),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    log_file.close()
    if not wait_for_http(f"http://127.0.0.1:{state.backend_port}/api/v1/healthz"):
        raise RuntimeError(
            f"mlx-tinker failed to start on port {state.backend_port}. "
            f"See {backend_log_path(paths)}"
        )
    return proc.pid


def start_proxy(paths: OpenClawPaths, state: ManagedState) -> int:
    if is_process_alive(state.proxy_pid, "openclaw-tinker/run.py"):
        return state.proxy_pid or 0
    log_file = proxy_log_path(paths).open("a", encoding="utf-8")
    env = os.environ.copy()
    env["TINKER_BASE_URL"] = f"http://127.0.0.1:{state.backend_port}"
    env["TINKER_API_KEY"] = DEFAULT_PROXY_API_KEY
    env["WANDB_DISABLED"] = "true"
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "uv",
        "run",
        "python",
        "openclaw-tinker/run.py",
        "--method",
        "rl",
        "--model-name",
        state.model,
        "--teacher-model-name",
        state.model,
        "--proxy-host",
        "127.0.0.1",
        "--proxy-port",
        str(state.proxy_port),
        "--served-model-name",
        state.model_id,
        "--api-key",
        DEFAULT_PROXY_API_KEY,
        "--record-dir",
        str(paths.records_dir),
        "--batch-size",
        str(DEFAULT_BATCH_SIZE),
        "--max-steps",
        str(DEFAULT_MAX_STEPS),
        "--loss-fn",
        "ppo",
        "--save-interval",
        "20",
        "--max-context-tokens",
        str(DEFAULT_MAX_CONTEXT_TOKENS),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(paths.openclaw_rl_dir),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    log_file.close()
    if not wait_for_http(f"http://127.0.0.1:{state.proxy_port}/healthz"):
        raise RuntimeError(
            f"OpenClaw-RL proxy failed to start on port {state.proxy_port}. "
            f"See {proxy_log_path(paths)}"
        )
    return proc.pid


def docker_compose(paths: OpenClawPaths, *args: str) -> subprocess.CompletedProcess[str]:
    return run_command(
        ["docker", "compose", "--env-file", str(paths.openclaw_compose_env_path), *args],
        cwd=paths.openclaw_dir,
        capture_output=True,
    )


def start_gateway(paths: OpenClawPaths, state: ManagedState) -> None:
    docker_compose(paths, "up", "-d", "openclaw-gateway")
    headers = {"Authorization": f"Bearer {state.gateway_token}"}
    if not wait_for_http(
        f"http://127.0.0.1:{state.gateway_port}/healthz",
        headers=headers,
        timeout_seconds=90,
    ):
        raise RuntimeError(
            f"OpenClaw gateway failed to start on port {state.gateway_port}. "
            "Run `mlx-tinker openclaw logs --service gateway` for details."
        )


def stop_gateway(paths: OpenClawPaths) -> None:
    run_command(
        ["docker", "compose", "--env-file", str(paths.openclaw_compose_env_path), "stop", "openclaw-gateway"],
        cwd=paths.openclaw_dir,
        capture_output=True,
        check=False,
    )


def render_status(paths: OpenClawPaths, state: ManagedState) -> str:
    backend_running = is_process_alive(state.backend_pid, "mlx_tinker")
    proxy_running = is_process_alive(state.proxy_pid, "openclaw-tinker/run.py")
    backend_ready = wait_for_http(
        f"http://127.0.0.1:{state.backend_port}/api/v1/healthz",
        timeout_seconds=1,
    )
    proxy_ready = wait_for_http(f"http://127.0.0.1:{state.proxy_port}/healthz", timeout_seconds=1)
    gateway_ready = wait_for_http(
        f"http://127.0.0.1:{state.gateway_port}/healthz",
        headers={"Authorization": f"Bearer {state.gateway_token}"},
        timeout_seconds=1,
    )
    lines = [
        f"Mode: {state.setup_mode}",
        f"Model: {state.model}",
        f"Primary model ref: {state.model_ref}",
        f"Gateway: http://127.0.0.1:{state.gateway_port} ({'ready' if gateway_ready else 'down'})",
        f"Proxy: http://127.0.0.1:{state.proxy_port}/v1 ({'ready' if proxy_ready else 'down'})",
        f"Backend: http://127.0.0.1:{state.backend_port} ({'ready' if backend_ready else 'down'})",
        f"RL batch size: {DEFAULT_BATCH_SIZE}",
        f"RL max context tokens: {DEFAULT_MAX_CONTEXT_TOKENS}",
        f"Gateway token: {state.gateway_token}",
        f"Backend PID: {state.backend_pid or '-'} ({'alive' if backend_running else 'stopped'})",
        f"Proxy PID: {state.proxy_pid or '-'} ({'alive' if proxy_running else 'stopped'})",
        f"Config: {paths.config_path}",
        f"Workspace: {paths.workspace_dir}",
        f"Logs: {paths.logs_dir}",
    ]
    return "\n".join(lines)


def tail_file(path: Path, lines: int) -> str:
    if not path.exists():
        return f"{path} does not exist yet."
    content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(content[-lines:])


def handle_setup(args: argparse.Namespace) -> None:
    paths = build_paths(Path(args.home_dir) if args.home_dir else None)
    ensure_directories(paths)

    require_command("git")
    require_command("docker")
    require_command("node")
    require_command("uv")
    bootstrap_openclaw_checkout(paths)
    bootstrap_openclaw_rl_checkout(paths)

    existing_state = load_state(paths)
    state = build_state_from_args(paths, args, existing_state)
    if paths.config_path.exists() and not state.config_backup:
        state.config_backup = create_config_backup(paths)

    copy_managed_plugins(paths)
    configure_openclaw(paths, state)
    write_compose_env(paths, state)

    if not args.no_start:
        ensure_openclaw_image(state.docker_image, openclaw_dir=paths.openclaw_dir)
        state.backend_pid = start_backend(paths, state)
        state.proxy_pid = start_proxy(paths, state)
        start_gateway(paths, state)

    save_state(paths, state)

    mode_msg = "Migrated existing OpenClaw config" if state.setup_mode == "existing" else "Created new OpenClaw config"
    print(mode_msg)
    print(render_status(paths, state))
    if state.config_backup:
        print(f"Backup: {state.config_backup}")


def handle_start(args: argparse.Namespace) -> None:
    paths = build_paths(Path(args.home_dir) if args.home_dir else None)
    state = load_state(paths)
    if state is None:
        raise RuntimeError("No mlx-tinker OpenClaw state found. Run `mlx-tinker openclaw setup` first.")
    ensure_openclaw_image(state.docker_image, openclaw_dir=paths.openclaw_dir)
    state.backend_pid = start_backend(paths, state)
    state.proxy_pid = start_proxy(paths, state)
    start_gateway(paths, state)
    save_state(paths, state)
    print(render_status(paths, state))


def handle_stop(args: argparse.Namespace) -> None:
    paths = build_paths(Path(args.home_dir) if args.home_dir else None)
    state = load_state(paths)
    if state is None:
        print("No mlx-tinker OpenClaw state found.")
        return
    stop_pid(state.proxy_pid, expected_fragment="openclaw-tinker/run.py")
    stop_pid(state.backend_pid, expected_fragment="mlx_tinker")
    stop_gateway(paths)
    state.backend_pid = None
    state.proxy_pid = None
    save_state(paths, state)
    print("Stopped mlx-tinker backend, OpenClaw-RL proxy, and Docker gateway.")


def handle_status(args: argparse.Namespace) -> None:
    paths = build_paths(Path(args.home_dir) if args.home_dir else None)
    state = load_state(paths)
    if state is None:
        print("No mlx-tinker OpenClaw state found.")
        return
    print(render_status(paths, state))


def handle_logs(args: argparse.Namespace) -> None:
    paths = build_paths(Path(args.home_dir) if args.home_dir else None)
    service = args.service
    if service in {"backend", "all"}:
        print(f"== backend ({backend_log_path(paths)}) ==")
        print(tail_file(backend_log_path(paths), args.lines))
    if service in {"proxy", "all"}:
        print(f"== proxy ({proxy_log_path(paths)}) ==")
        print(tail_file(proxy_log_path(paths), args.lines))
    if service in {"gateway", "all"}:
        result = run_command(
            [
                "docker",
                "compose",
                "--env-file",
                str(paths.openclaw_compose_env_path),
                "logs",
                f"--tail={args.lines}",
                "openclaw-gateway",
            ],
            cwd=paths.openclaw_dir,
            capture_output=True,
            check=False,
        )
        print("== gateway (docker compose logs) ==")
        print((result.stdout or result.stderr or "").strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-tinker openclaw",
        description="Docker-first OpenClaw onboarding and service management for mlx-tinker.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_home_arg(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--home-dir",
            default=None,
            help="Home directory to manage (defaults to the current user's home). "
            "Useful for smoke-testing against a temp home.",
        )

    setup = subparsers.add_parser(
        "setup",
        help="Bootstrap OpenClaw + OpenClaw-RL, install the RL header plugin, patch config, and start the stack.",
    )
    add_home_arg(setup)
    setup.add_argument("--model", default=None, help="HF model name to run locally.")
    setup.add_argument("--backend-port", type=int, default=None, help="mlx-tinker backend port.")
    setup.add_argument("--proxy-port", type=int, default=None, help="OpenClaw-RL proxy port.")
    setup.add_argument("--gateway-port", type=int, default=None, help="OpenClaw gateway port.")
    setup.add_argument("--bridge-port", type=int, default=None, help="OpenClaw bridge port.")
    setup.add_argument("--gateway-bind", default=None, choices=["loopback", "lan"], help="Gateway bind mode.")
    setup.add_argument("--docker-image", default=None, help="Docker image for OpenClaw gateway.")
    setup.add_argument("--no-start", action="store_true", help="Only prepare config and files.")

    start = subparsers.add_parser("start", help="Start the local backend, RL proxy, and Docker gateway.")
    add_home_arg(start)

    stop = subparsers.add_parser("stop", help="Stop the local backend, RL proxy, and Docker gateway.")
    add_home_arg(stop)

    status = subparsers.add_parser("status", help="Show current service and config status.")
    add_home_arg(status)

    logs = subparsers.add_parser("logs", help="Show recent backend/proxy/gateway logs.")
    add_home_arg(logs)
    logs.add_argument("--service", choices=["all", "backend", "proxy", "gateway"], default="all")
    logs.add_argument("--lines", type=int, default=40)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "setup":
        handle_setup(args)
        return
    if args.command == "start":
        handle_start(args)
        return
    if args.command == "stop":
        handle_stop(args)
        return
    if args.command == "status":
        handle_status(args)
        return
    if args.command == "logs":
        handle_logs(args)
        return
    parser.error(f"unknown command: {args.command}")
