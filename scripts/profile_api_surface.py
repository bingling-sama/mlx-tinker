from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_TIMEOUT = 120.0


@dataclass
class RunContext:
    base_url: str
    client: httpx.Client
    server_proc: subprocess.Popen[str]
    run_dir: Path
    checkpoints_dir: Path
    model: str
    port: int
    session_id: str | None = None
    base_sampling_session_id: str | None = None
    model_id: str | None = None
    explicit_sampler_path: str | None = None
    explicit_sampling_session_id: str | None = None
    ephemeral_sampling_session_id: str | None = None


def _now_utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((DEFAULT_HOST, 0))
        return int(sock.getsockname()[1])


def _rss_mb(pid: int) -> float | None:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        if not out:
            return None
        return round(int(out) / 1024.0, 2)
    except Exception:
        return None


def _dir_size_mb(path: Path) -> float:
    total = 0
    if path.exists():
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    return round(total / (1024.0 * 1024.0), 4)


def _make_tokens(length: int, *, offset: int = 0) -> list[int]:
    return [((idx + offset) % 97) + 10 for idx in range(length)]


def _make_model_input(length: int, *, offset: int = 0) -> dict[str, Any]:
    return {
        "chunks": [
            {
                "type": "encoded_text",
                "tokens": _make_tokens(length, offset=offset),
            }
        ]
    }


def _make_ce_datum(seq_len: int, *, offset: int = 0) -> dict[str, Any]:
    tokens = _make_tokens(seq_len, offset=offset)
    targets = tokens[1:] + [tokens[-1]]
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": targets, "dtype": "int64"},
            "weights": {"data": [1.0] * seq_len, "dtype": "float32"},
        },
    }


def _make_is_datum(seq_len: int, *, offset: int = 0) -> dict[str, Any]:
    tokens = _make_tokens(seq_len, offset=offset)
    targets = tokens[1:] + [tokens[-1]]
    advantages = [((idx % 8) - 3.5) / 3.5 for idx in range(seq_len)]
    logprobs = [-1.5 - ((idx % 5) * 0.05) for idx in range(seq_len)]
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
        "loss_fn_inputs": {
            "target_tokens": {"data": targets, "dtype": "int64"},
            "advantages": {"data": advantages, "dtype": "float32"},
            "logprobs": {"data": logprobs, "dtype": "float32"},
        },
    }


def _summarize_response_json(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"response_type": type(data).__name__}

    summary: dict[str, Any] = {"keys": sorted(data.keys())}
    if "session_id" in data:
        summary["session_id"] = data["session_id"]
    if "sampling_session_id" in data:
        summary["sampling_session_id"] = data["sampling_session_id"]
    if "model_id" in data:
        summary["model_id"] = data["model_id"]
    if "status" in data:
        summary["status"] = data["status"]
    if "supported_models" in data:
        summary["supported_models"] = len(data["supported_models"])
    if "sequences" in data:
        sequences = data["sequences"]
        summary["num_sequences"] = len(sequences)
        summary["generated_tokens_total"] = sum(len(seq.get("tokens", [])) for seq in sequences)
        summary["generated_tokens_per_sequence"] = [len(seq.get("tokens", [])) for seq in sequences]
    if "logprobs" in data:
        summary["forward_rows"] = len(data["logprobs"])
        summary["forward_tokens_total"] = sum(len(row) for row in data["logprobs"])
    if "loss_fn_outputs" in data:
        summary["loss_fn_outputs"] = len(data["loss_fn_outputs"])
    if "metrics" in data and isinstance(data["metrics"], dict):
        metrics = data["metrics"]
        interesting = {}
        for key in (
            "loss:sum",
            "num_tokens:sum",
            "num_sequences:sum",
            "grad_norm:mean",
            "grad_norm_clipped:mean",
            "learning_rate:unique",
            "skipped:sum",
        ):
            if key in metrics:
                interesting[key] = metrics[key]
        summary["metrics"] = interesting
    if "path" in data and data["path"] is not None:
        summary["path"] = data["path"]
    return summary


def _request_bytes(payload: Any) -> int:
    return len(json.dumps(payload).encode("utf-8")) if payload is not None else 0


def _start_server(run_dir: Path, model: str, port: int) -> subprocess.Popen[str]:
    db_path = run_dir / "profile.db"
    checkpoints_dir = run_dir / "checkpoints"
    log_path = run_dir / "server.log"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_file = log_path.open("w")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlx_tinker",
            "--model",
            model,
            "--host",
            DEFAULT_HOST,
            "--port",
            str(port),
            "--db",
            str(db_path),
            "--checkpoints",
            str(checkpoints_dir),
            "--log-level",
            "INFO",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_for_health(base_url: str, timeout_sec: float = 30.0) -> None:
    deadline = time.perf_counter() + timeout_sec
    with httpx.Client(timeout=5.0) as client:
        while time.perf_counter() < deadline:
            try:
                resp = client.get(f"{base_url}/api/v1/healthz")
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.2)
    raise RuntimeError(f"Server at {base_url} did not become healthy within {timeout_sec}s")


def _stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _sync_case(
    ctx: RunContext,
    name: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    repeat: int = 1,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    statuses: list[int] = []
    response_summary: dict[str, Any] = {}
    rss_before = _rss_mb(ctx.server_proc.pid)
    started_at = time.time()

    for _ in range(repeat):
        t0 = time.perf_counter()
        if method == "GET":
            resp = ctx.client.get(path)
        else:
            resp = ctx.client.post(path, json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        statuses.append(resp.status_code)
        response_summary = _summarize_response_json(resp.json())
        resp.raise_for_status()

    result = {
        "name": name,
        "endpoint": path,
        "method": method,
        "case_type": "sync",
        "repeat": repeat,
        "request_bytes": _request_bytes(payload),
        "status_codes": statuses,
        "started_at_epoch_sec": started_at,
        "latency_ms": {
            "min": round(min(latencies_ms), 3),
            "median": round(statistics.median(latencies_ms), 3),
            "max": round(max(latencies_ms), 3),
            "p95": round(sorted(latencies_ms)[max(0, int(len(latencies_ms) * 0.95) - 1)], 3),
        },
        "response_summary": response_summary,
        "rss_before_mb": rss_before,
        "rss_after_mb": _rss_mb(ctx.server_proc.pid),
    }
    if extra:
        result["extra"] = extra
    return result


def _future_case(
    ctx: RunContext,
    name: str,
    path: str,
    payload: dict[str, Any],
    *,
    max_wait_sec: float = 900.0,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rss_before = _rss_mb(ctx.server_proc.pid)
    started_at = time.time()

    submit_start = time.perf_counter()
    submit_resp = ctx.client.post(path, json=payload)
    submit_latency_ms = (time.perf_counter() - submit_start) * 1000.0
    submit_resp.raise_for_status()
    submit_json = submit_resp.json()
    request_id = submit_json["request_id"]

    retrieve_calls = 0
    retrieve_wait_ms_total = 0.0
    retrieve_latencies: list[float] = []
    poll_errors: list[str] = []
    result_json: dict[str, Any] | None = None
    future_status = "completed"
    deadline = time.perf_counter() + max_wait_sec

    while time.perf_counter() < deadline:
        retrieve_calls += 1
        retrieve_start = time.perf_counter()
        retrieve_resp = ctx.client.post(
            "/api/v1/retrieve_future",
            json={"request_id": request_id, "allow_metadata_only": False},
        )
        retrieve_elapsed_ms = (time.perf_counter() - retrieve_start) * 1000.0
        retrieve_wait_ms_total += retrieve_elapsed_ms
        retrieve_latencies.append(retrieve_elapsed_ms)
        retrieve_resp.raise_for_status()
        retrieve_json = retrieve_resp.json()
        if retrieve_json.get("type") == "try_again":
            continue
        if "error" in retrieve_json:
            future_status = "failed"
            poll_errors.append(retrieve_json.get("error", "Unknown error"))
            result_json = retrieve_json
            break
        result_json = retrieve_json
        break

    if result_json is None:
        future_status = "timeout"
        result_json = {"error": f"Timed out after {max_wait_sec}s"}

    ready_latency_ms = (time.perf_counter() - submit_start) * 1000.0
    result = {
        "name": name,
        "endpoint": path,
        "method": "POST",
        "case_type": "future",
        "request_bytes": _request_bytes(payload),
        "started_at_epoch_sec": started_at,
        "future_status": future_status,
        "submit_latency_ms": round(submit_latency_ms, 3),
        "ready_latency_ms": round(ready_latency_ms, 3),
        "retrieve_calls": retrieve_calls,
        "retrieve_wait_ms_total": round(retrieve_wait_ms_total, 3),
        "retrieve_latency_ms": {
            "min": round(min(retrieve_latencies), 3) if retrieve_latencies else None,
            "median": round(statistics.median(retrieve_latencies), 3) if retrieve_latencies else None,
            "max": round(max(retrieve_latencies), 3) if retrieve_latencies else None,
        },
        "submit_response_summary": _summarize_response_json(submit_json),
        "result_summary": _summarize_response_json(result_json),
        "errors": poll_errors,
        "rss_before_mb": rss_before,
        "rss_after_mb": _rss_mb(ctx.server_proc.pid),
    }
    if extra:
        result["extra"] = extra
    return result, result_json


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False))


def _make_markdown(run_summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 0.8B API Surface Profiling Report")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- Model: `{run_summary['model']}`")
    lines.append(f"- Base URL: `{run_summary['base_url']}`")
    lines.append(f"- Timestamp (UTC): `{run_summary['timestamp_utc']}`")
    lines.append(f"- Total cases: `{len(run_summary['results'])}`")
    lines.append(f"- Raw JSON: `{run_summary['raw_results_path']}`")
    lines.append(f"- Server log: `{run_summary['server_log_path']}`")
    lines.append("")

    failures = [r for r in run_summary["results"] if r.get("future_status") in {"failed", "timeout"}]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for failure in failures:
            lines.append(
                f"- `{failure['name']}`: `{failure.get('future_status')}` "
                f"({'; '.join(failure.get('errors', [])) or 'see raw results'})"
            )
        lines.append("")

    lines.append("## Endpoint Summary")
    lines.append("")
    lines.append("| Case | Endpoint | Type | Key Latency | Polls | RSS After (MB) |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: |")
    for result in run_summary["results"]:
        if result["case_type"] == "sync":
            latency = result["latency_ms"]["median"]
            polls = "-"
        else:
            latency = result["ready_latency_ms"]
            polls = result["retrieve_calls"]
        rss_after = result.get("rss_after_mb")
        rss_after_str = "-" if rss_after is None else f"{rss_after:.2f}"
        lines.append(
            f"| `{result['name']}` | `{result['endpoint']}` | `{result['case_type']}` | "
            f"{latency} | {polls} | {rss_after_str} |"
        )
    lines.append("")

    lines.append("## Slowest Future Cases")
    lines.append("")
    future_cases = [r for r in run_summary["results"] if r["case_type"] == "future"]
    future_cases.sort(key=lambda item: item["ready_latency_ms"], reverse=True)
    for result in future_cases[:10]:
        lines.append(
            f"- `{result['name']}`: ready `{result['ready_latency_ms']:.1f} ms`, "
            f"submit `{result['submit_latency_ms']:.1f} ms`, polls `{result['retrieve_calls']}`, "
            f"request bytes `{result['request_bytes']}`"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Future-backed routes include both submission cost and completion cost.")
    lines.append("- `retrieve_future` polling counts reflect the current server behavior with bounded waits inside the endpoint.")
    lines.append("- RSS numbers come from `ps` snapshots around each case, so they are approximate rather than allocator-accurate.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the local MLX-Tinker API surface sequentially.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="workspace_reports")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_root = (repo_root / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    timestamp = _now_utc_stamp()
    run_dir = out_root / f"api_profile_0_8b_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    port = args.port or _find_free_port()
    base_url = f"http://{DEFAULT_HOST}:{port}"

    server_proc = _start_server(run_dir, args.model, port)
    try:
        _wait_for_health(base_url)
        client = httpx.Client(base_url=base_url, timeout=DEFAULT_TIMEOUT)
        ctx = RunContext(
            base_url=base_url,
            client=client,
            server_proc=server_proc,
            run_dir=run_dir,
            checkpoints_dir=run_dir / "checkpoints",
            model=args.model,
            port=port,
        )

        results: list[dict[str, Any]] = []

        # Cheap synchronous endpoints.
        results.append(_sync_case(ctx, "root", "GET", "/", repeat=10))
        results.append(_sync_case(ctx, "healthz", "GET", "/api/v1/healthz", repeat=10))
        results.append(
            _sync_case(ctx, "get_server_capabilities", "GET", "/api/v1/get_server_capabilities", repeat=10)
        )
        results.append(
            _sync_case(
                ctx,
                "telemetry_single",
                "POST",
                "/api/v1/telemetry",
                payload={"event": "profile_single", "platform": "local", "sdk_version": "profiling"},
                repeat=5,
            )
        )
        results.append(
            _sync_case(
                ctx,
                "telemetry_batch",
                "POST",
                "/api/v1/telemetry",
                payload={
                    "events": [
                        {"event": "case_start", "ts": timestamp},
                        {"event": "case_end", "ts": timestamp},
                    ],
                    "platform": "local",
                    "sdk_version": "profiling",
                },
                repeat=5,
            )
        )

        # Session lifecycle.
        create_session_payload = {
            "tags": ["profiling", "0.8b"],
            "user_metadata": {"purpose": "api_surface_profile"},
            "sdk_version": "0.16.1",
            "project_id": "api-profile",
            "type": "create_session",
        }
        session_result = _sync_case(
            ctx,
            "create_session",
            "POST",
            "/api/v1/create_session",
            payload=create_session_payload,
            repeat=5,
        )
        results.append(session_result)
        session_resp = ctx.client.post("/api/v1/create_session", json=create_session_payload)
        session_resp.raise_for_status()
        ctx.session_id = session_resp.json()["session_id"]

        results.append(
            _sync_case(
                ctx,
                "session_heartbeat",
                "POST",
                "/api/v1/session_heartbeat",
                payload={"session_id": ctx.session_id},
                repeat=5,
            )
        )

        base_sampling_session_result = _sync_case(
            ctx,
            "create_sampling_session_base_model",
            "POST",
            "/api/v1/create_sampling_session",
            payload={
                "session_id": ctx.session_id,
                "sampling_session_seq_id": 0,
                "base_model": args.model,
                "type": "create_sampling_session",
            },
            repeat=3,
        )
        results.append(base_sampling_session_result)
        base_sampling_session_resp = ctx.client.post(
            "/api/v1/create_sampling_session",
            json={
                "session_id": ctx.session_id,
                "sampling_session_seq_id": 1,
                "base_model": args.model,
                "type": "create_sampling_session",
            },
        )
        base_sampling_session_resp.raise_for_status()
        ctx.base_sampling_session_id = base_sampling_session_resp.json()["sampling_session_id"]

        # Base-model sampling cases.
        sample_cases = [
            (
                "asample_base_greedy_ns1_pl32_mt16",
                {
                    "prompt": _make_model_input(32),
                    "sampling_params": {"temperature": 0.0, "max_tokens": 16, "seed": 11, "top_p": 1.0},
                    "num_samples": 1,
                    "base_model": args.model,
                    "type": "sample",
                },
            ),
            (
                "asample_base_temp_ns1_pl32_mt64",
                {
                    "prompt": _make_model_input(32, offset=10),
                    "sampling_params": {"temperature": 0.8, "max_tokens": 64, "seed": 12, "top_p": 0.95},
                    "num_samples": 1,
                    "base_model": args.model,
                    "type": "sample",
                },
            ),
            (
                "asample_base_temp_ns4_pl64_mt32",
                {
                    "prompt": _make_model_input(64, offset=20),
                    "sampling_params": {"temperature": 0.8, "max_tokens": 32, "seed": 13, "top_p": 0.95},
                    "num_samples": 4,
                    "base_model": args.model,
                    "type": "sample",
                },
            ),
            (
                "asample_base_temp_ns8_pl64_mt24",
                {
                    "prompt": _make_model_input(64, offset=30),
                    "sampling_params": {"temperature": 0.8, "max_tokens": 24, "seed": 14, "top_p": 0.95},
                    "num_samples": 8,
                    "base_model": args.model,
                    "type": "sample",
                },
            ),
            (
                "asample_base_prompt_logprobs_ns1_pl128_mt8",
                {
                    "prompt": _make_model_input(128, offset=40),
                    "sampling_params": {"temperature": 0.0, "max_tokens": 8, "seed": 15, "top_p": 1.0},
                    "num_samples": 1,
                    "base_model": args.model,
                    "prompt_logprobs": True,
                    "topk_prompt_logprobs": 0,
                    "type": "sample",
                },
            ),
            (
                "asample_base_sampling_session_ns1_pl32_mt24",
                {
                    "prompt": _make_model_input(32, offset=50),
                    "sampling_params": {"temperature": 0.7, "max_tokens": 24, "seed": 16, "top_p": 0.95},
                    "num_samples": 1,
                    "sampling_session_id": ctx.base_sampling_session_id,
                    "type": "sample",
                },
            ),
        ]
        for name, payload in sample_cases:
            result, _ = _future_case(ctx, name, "/api/v1/asample", payload, max_wait_sec=900.0)
            results.append(result)

        # Create main training model.
        create_model_payload = {
            "session_id": ctx.session_id,
            "model_seq_id": 0,
            "base_model": args.model,
            "lora_config": {
                "rank": 16,
                "alpha": 32.0,
                "seed": 42,
                "train_attn": True,
                "train_mlp": True,
                "train_unembed": False,
            },
            "user_metadata": {"case": "main_profile_model"},
            "type": "create_model",
        }
        create_model_result, create_model_json = _future_case(
            ctx,
            "create_model_rank16",
            "/api/v1/create_model",
            create_model_payload,
            max_wait_sec=1800.0,
        )
        results.append(create_model_result)
        ctx.model_id = create_model_json.get("model_id") or create_model_result["submit_response_summary"].get("model_id")
        if ctx.model_id is None:
            raise RuntimeError("Failed to capture model_id from create_model")

        results.append(
            _sync_case(
                ctx,
                "get_info_main_model",
                "POST",
                "/api/v1/get_info",
                payload={"model_id": ctx.model_id},
                repeat=3,
            )
        )

        # In-memory sampling on the trainable model.
        for name, payload in [
            (
                "asample_model_id_ns1_pl32_mt24",
                {
                    "model_id": ctx.model_id,
                    "prompt": _make_model_input(32, offset=60),
                    "sampling_params": {"temperature": 0.7, "max_tokens": 24, "seed": 17, "top_p": 0.95},
                    "num_samples": 1,
                    "type": "sample",
                },
            ),
            (
                "asample_model_id_ns8_pl64_mt24",
                {
                    "model_id": ctx.model_id,
                    "prompt": _make_model_input(64, offset=70),
                    "sampling_params": {"temperature": 0.7, "max_tokens": 24, "seed": 18, "top_p": 0.95},
                    "num_samples": 8,
                    "type": "sample",
                },
            ),
        ]:
            result, _ = _future_case(ctx, name, "/api/v1/asample", payload, max_wait_sec=900.0)
            results.append(result)

        # Forward / forward_backward / optim step.
        training_cases = [
            (
                "forward_b1_s32",
                "/api/v1/forward",
                {
                    "forward_input": {"data": [_make_ce_datum(32)], "loss_fn": "cross_entropy"},
                    "model_id": ctx.model_id,
                    "seq_id": 1,
                },
            ),
            (
                "forward_b4_s128",
                "/api/v1/forward",
                {
                    "forward_input": {"data": [_make_ce_datum(128, offset=i * 7) for i in range(4)], "loss_fn": "cross_entropy"},
                    "model_id": ctx.model_id,
                    "seq_id": 2,
                },
            ),
            (
                "forward_backward_ce_b1_s32",
                "/api/v1/forward_backward",
                {
                    "forward_backward_input": {
                        "data": [_make_ce_datum(32, offset=100)],
                        "loss_fn": "cross_entropy",
                    },
                    "model_id": ctx.model_id,
                    "seq_id": 3,
                },
            ),
            (
                "optim_step_after_ce_b1",
                "/api/v1/optim_step",
                {
                    "adam_params": {
                        "learning_rate": 1e-4,
                        "beta1": 0.9,
                        "beta2": 0.999,
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                        "grad_clip_norm": 0.0,
                    },
                    "model_id": ctx.model_id,
                    "seq_id": 4,
                    "type": "optim_step",
                },
            ),
            (
                "forward_backward_ce_b4_s128",
                "/api/v1/forward_backward",
                {
                    "forward_backward_input": {
                        "data": [_make_ce_datum(128, offset=110 + i * 9) for i in range(4)],
                        "loss_fn": "cross_entropy",
                    },
                    "model_id": ctx.model_id,
                    "seq_id": 5,
                },
            ),
            (
                "optim_step_after_ce_b4_clip1",
                "/api/v1/optim_step",
                {
                    "adam_params": {
                        "learning_rate": 1e-4,
                        "beta1": 0.9,
                        "beta2": 0.999,
                        "eps": 1e-8,
                        "weight_decay": 0.01,
                        "grad_clip_norm": 1.0,
                    },
                    "model_id": ctx.model_id,
                    "seq_id": 6,
                    "type": "optim_step",
                },
            ),
            (
                "forward_backward_is_b8_s96",
                "/api/v1/forward_backward",
                {
                    "forward_backward_input": {
                        "data": [_make_is_datum(96, offset=200 + i * 11) for i in range(8)],
                        "loss_fn": "importance_sampling",
                    },
                    "model_id": ctx.model_id,
                    "seq_id": 7,
                },
            ),
            (
                "optim_step_after_is_b8",
                "/api/v1/optim_step",
                {
                    "adam_params": {
                        "learning_rate": 5e-5,
                        "beta1": 0.9,
                        "beta2": 0.999,
                        "eps": 1e-8,
                        "weight_decay": 0.0,
                        "grad_clip_norm": 1.0,
                    },
                    "model_id": ctx.model_id,
                    "seq_id": 8,
                    "type": "optim_step",
                },
            ),
        ]
        for name, path, payload in training_cases:
            result, _ = _future_case(ctx, name, path, payload, max_wait_sec=1200.0)
            results.append(result)

        # Checkpointing and sampler export.
        checkpoint_id = "profile_ckpt"
        checkpoint_path = ctx.checkpoints_dir / ctx.model_id / checkpoint_id
        save_weights_result, save_weights_json = _future_case(
            ctx,
            "save_weights_profile_ckpt",
            "/api/v1/save_weights",
            {
                "model_id": ctx.model_id,
                "path": str(checkpoint_path),
                "seq_id": 9,
                "type": "save_weights",
            },
            max_wait_sec=1200.0,
        )
        save_weights_result["extra"] = {"checkpoint_size_mb": _dir_size_mb(checkpoint_path)}
        save_weights_result["result_summary"]["path"] = save_weights_json.get("path")
        results.append(save_weights_result)

        load_weights_result, _ = _future_case(
            ctx,
            "load_weights_profile_ckpt",
            "/api/v1/load_weights",
            {
                "model_id": ctx.model_id,
                "source_model_id": ctx.model_id,
                "checkpoint_id": checkpoint_id,
                "optimizer": True,
                "seq_id": 10,
                "type": "load_weights",
            },
            max_wait_sec=1200.0,
        )
        results.append(load_weights_result)

        explicit_sampler_path = ctx.checkpoints_dir / ctx.model_id / "sampler" / "explicit_profile"
        ctx.explicit_sampler_path = str(explicit_sampler_path)
        save_sampler_explicit_result, _ = _future_case(
            ctx,
            "save_weights_for_sampler_explicit",
            "/api/v1/save_weights_for_sampler",
            {
                "model_id": ctx.model_id,
                "path": ctx.explicit_sampler_path,
                "seq_id": 11,
                "type": "save_weights_for_sampler",
            },
            max_wait_sec=1200.0,
        )
        save_sampler_explicit_result["extra"] = {"checkpoint_size_mb": _dir_size_mb(explicit_sampler_path)}
        results.append(save_sampler_explicit_result)

        explicit_sampling_session_result = _sync_case(
            ctx,
            "create_sampling_session_model_path",
            "POST",
            "/api/v1/create_sampling_session",
            payload={
                "session_id": ctx.session_id,
                "sampling_session_seq_id": 2,
                "model_path": ctx.explicit_sampler_path,
                "base_model": args.model,
                "type": "create_sampling_session",
            },
            repeat=3,
        )
        results.append(explicit_sampling_session_result)
        explicit_sampling_session_resp = ctx.client.post(
            "/api/v1/create_sampling_session",
            json={
                "session_id": ctx.session_id,
                "sampling_session_seq_id": 3,
                "model_path": ctx.explicit_sampler_path,
                "base_model": args.model,
                "type": "create_sampling_session",
            },
        )
        explicit_sampling_session_resp.raise_for_status()
        ctx.explicit_sampling_session_id = explicit_sampling_session_resp.json()["sampling_session_id"]

        for name, payload in [
            (
                "asample_sampler_session_ns1_pl32_mt24",
                {
                    "prompt": _make_model_input(32, offset=300),
                    "sampling_params": {"temperature": 0.7, "max_tokens": 24, "seed": 19, "top_p": 0.95},
                    "num_samples": 1,
                    "sampling_session_id": ctx.explicit_sampling_session_id,
                    "type": "sample",
                },
            ),
            (
                "asample_sampler_session_ns8_pl64_mt24",
                {
                    "prompt": _make_model_input(64, offset=310),
                    "sampling_params": {"temperature": 0.7, "max_tokens": 24, "seed": 20, "top_p": 0.95},
                    "num_samples": 8,
                    "sampling_session_id": ctx.explicit_sampling_session_id,
                    "type": "sample",
                },
            ),
        ]:
            result, _ = _future_case(ctx, name, "/api/v1/asample", payload, max_wait_sec=900.0)
            results.append(result)

        save_sampler_ephemeral_result, save_sampler_ephemeral_json = _future_case(
            ctx,
            "save_weights_for_sampler_ephemeral",
            "/api/v1/save_weights_for_sampler",
            {
                "model_id": ctx.model_id,
                "sampling_session_seq_id": 4,
                "seq_id": 12,
                "ttl_seconds": 3600,
                "type": "save_weights_for_sampler",
            },
            max_wait_sec=1200.0,
        )
        ctx.ephemeral_sampling_session_id = save_sampler_ephemeral_json.get("sampling_session_id")
        if ctx.ephemeral_sampling_session_id:
            ephemeral_dir = ctx.checkpoints_dir / ctx.model_id / "sampler" / ctx.ephemeral_sampling_session_id
            save_sampler_ephemeral_result["extra"] = {"checkpoint_size_mb": _dir_size_mb(ephemeral_dir)}
        results.append(save_sampler_ephemeral_result)

        for name, payload in [
            (
                "asample_ephemeral_sampler_ns1_pl32_mt24",
                {
                    "prompt": _make_model_input(32, offset=320),
                    "sampling_params": {"temperature": 0.7, "max_tokens": 24, "seed": 21, "top_p": 0.95},
                    "num_samples": 1,
                    "sampling_session_id": ctx.ephemeral_sampling_session_id,
                    "type": "sample",
                },
            ),
            (
                "asample_ephemeral_sampler_ns8_pl64_mt24",
                {
                    "prompt": _make_model_input(64, offset=330),
                    "sampling_params": {"temperature": 0.7, "max_tokens": 24, "seed": 22, "top_p": 0.95},
                    "num_samples": 8,
                    "sampling_session_id": ctx.ephemeral_sampling_session_id,
                    "type": "sample",
                },
            ),
        ]:
            result, _ = _future_case(ctx, name, "/api/v1/asample", payload, max_wait_sec=900.0)
            results.append(result)

        unload_main_result, _ = _future_case(
            ctx,
            "unload_model_rank16",
            "/api/v1/unload_model",
            {"model_id": ctx.model_id},
            max_wait_sec=300.0,
        )
        results.append(unload_main_result)

        # Alternate create_model params to capture rank sensitivity.
        create_model_rank8_result, rank8_json = _future_case(
            ctx,
            "create_model_rank8",
            "/api/v1/create_model",
            {
                "session_id": ctx.session_id,
                "model_seq_id": 1,
                "base_model": args.model,
                "lora_config": {
                    "rank": 8,
                    "alpha": 16.0,
                    "seed": 43,
                    "train_attn": True,
                    "train_mlp": True,
                    "train_unembed": False,
                },
                "user_metadata": {"case": "rank8_compare"},
                "type": "create_model",
            },
            max_wait_sec=1800.0,
        )
        results.append(create_model_rank8_result)
        rank8_model_id = rank8_json.get("model_id") or create_model_rank8_result["submit_response_summary"].get("model_id")
        if rank8_model_id:
            unload_rank8_result, _ = _future_case(
                ctx,
                "unload_model_rank8",
                "/api/v1/unload_model",
                {"model_id": rank8_model_id},
                max_wait_sec=300.0,
            )
            results.append(unload_rank8_result)

        client.close()

        run_summary = {
            "timestamp_utc": timestamp,
            "model": args.model,
            "port": port,
            "base_url": base_url,
            "raw_results_path": str(run_dir / "results.json"),
            "server_log_path": str(run_dir / "server.log"),
            "results": results,
        }
        _write_json(run_dir / "results.json", run_summary)
        (run_dir / "report.md").write_text(_make_markdown(run_summary))
        print(json.dumps({
            "run_dir": str(run_dir),
            "report_path": str(run_dir / "report.md"),
            "results_path": str(run_dir / "results.json"),
            "cases": len(results),
        }, indent=2))
    finally:
        _stop_server(server_proc)


if __name__ == "__main__":
    main()
