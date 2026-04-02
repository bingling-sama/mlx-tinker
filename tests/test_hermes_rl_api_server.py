from __future__ import annotations

import json
import queue
import threading
import time

from fastapi.testclient import TestClient

from integrations.hermes_rl.api_server import HermesCombineServer, HermesOPDServer, HermesRLServer
from integrations.hermes_rl.config import HermesRLConfig


class _FakeSequence:
    def __init__(self, tokens, logprobs, stop_reason="stop"):
        self.tokens = tokens
        self.logprobs = logprobs
        self.stop_reason = stop_reason


class _FakeSampleResponse:
    def __init__(self, sequence):
        self.sequences = [sequence]


class _FakeSamplingClient:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls = []

    async def sample_async(self, **kwargs):
        self.calls.append(kwargs)
        text = self.outputs.pop(0)
        tokens = [ord(char) for char in text]
        logprobs = [-0.1] * len(tokens)
        return _FakeSampleResponse(_FakeSequence(tokens=tokens, logprobs=logprobs))


class _FakeTokenizer:
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token) for token in tokens)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        del tokenize, add_generation_prompt, kwargs
        return "\n".join(f"{message['role']}: {message.get('content', '')}" for message in messages)

    def __call__(self, text: str, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": self.encode(text)}


class _FakePRMScorer:
    def __init__(self, score: float = 1.0):
        self.score = score
        self.calls = []

    async def evaluate(self, response_text, next_state_text, next_state_role, session_id, turn_num):
        self.calls.append(
            {
                "response_text": response_text,
                "next_state_text": next_state_text,
                "next_state_role": next_state_role,
                "session_id": session_id,
                "turn_num": turn_num,
            }
        )
        return {"score": self.score, "votes": [self.score], "representative": response_text}


class _FakeOPDScorer:
    def __init__(self, *, accepted: bool = True, eval_score: float | None = 1.0):
        self.accepted = accepted
        self.eval_score = eval_score
        self.calls = []

    async def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        response_len = len(kwargs["turn_data"]["response_ids"])
        return {
            "accepted": self.accepted,
            "hint": "use the tool result",
            "hint_raw": "raw hint",
            "eval_score": self.eval_score,
            "eval_raw": "raw eval",
            "teacher_log_probs": [-0.2] * response_len,
        }


def _wait_for(predicate, timeout: float = 1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for condition")


def _build_headers(
    *,
    session_id: str,
    outer_turn_id: str,
    step_index: int,
    request_id: str,
    turn_type: str = "main",
):
    return {
        "X-Session-Id": session_id,
        "X-Hermes-Outer-Turn-Id": outer_turn_id,
        "X-Hermes-Step-Index": str(step_index),
        "X-Hermes-Request-Id": request_id,
        "X-Turn-Type": turn_type,
    }


def _make_config(tmp_path) -> HermesRLConfig:
    return HermesRLConfig(
        model_name="Qwen/Qwen3.5-4B",
        served_model_name="hermes-local",
        record_dir=str(tmp_path),
        proxy_api_key="",
    )


def _make_messages(role: str, content: str):
    return [{"role": role, "content": content}]


def test_rl_server_flushes_previous_step_on_next_step(tmp_path):
    output_queue: queue.Queue = queue.Queue()
    enabled = threading.Event()
    enabled.set()
    scorer = _FakePRMScorer(score=1.0)
    server = HermesRLServer(
        config=_make_config(tmp_path),
        output_queue=output_queue,
        submission_enabled=enabled,
        sampling_client=_FakeSamplingClient(["tool step", "final answer"]),
        tokenizer=_FakeTokenizer(),
        prm_scorer=scorer,
    )
    client = TestClient(server.app)

    response = client.post(
        "/v1/chat/completions",
        headers=_build_headers(
            session_id="sess-1",
            outer_turn_id="1",
            step_index=1,
            request_id="req-1",
        ),
        json={"model": "hermes-local", "messages": _make_messages("user", "start")},
    )
    assert response.status_code == 200

    response = client.post(
        "/v1/chat/completions",
        headers=_build_headers(
            session_id="sess-1",
            outer_turn_id="1",
            step_index=2,
            request_id="req-2",
        ),
        json={
            "model": "hermes-local",
            "messages": [
                {"role": "user", "content": "start"},
                {"role": "tool", "content": "tool returned success"},
            ],
        },
    )
    assert response.status_code == 200

    _wait_for(lambda: output_queue.qsize() == 1)
    _, group = output_queue.get_nowait()
    sample = group[0]
    assert sample.turn_num == 1
    assert sample.reward == 1.0
    assert sample.response_text == "tool step"

    assert scorer.calls[0]["next_state_role"] == "tool"
    assert scorer.calls[0]["next_state_text"] == "tool returned success"

    records = (tmp_path / "conversations.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert record["outer_turn_id"] == "1"
    assert record["step_index"] == 1
    assert record["next_state"]["role"] == "tool"


def test_rl_server_dedupes_duplicate_request_id(tmp_path):
    output_queue: queue.Queue = queue.Queue()
    enabled = threading.Event()
    enabled.set()
    server = HermesRLServer(
        config=_make_config(tmp_path),
        output_queue=output_queue,
        submission_enabled=enabled,
        sampling_client=_FakeSamplingClient(["first answer", "duplicate answer", "next answer"]),
        tokenizer=_FakeTokenizer(),
        prm_scorer=_FakePRMScorer(score=1.0),
    )
    client = TestClient(server.app)

    headers = _build_headers(
        session_id="sess-2",
        outer_turn_id="1",
        step_index=1,
        request_id="req-1",
    )
    payload = {"model": "hermes-local", "messages": _make_messages("user", "start")}
    assert client.post("/v1/chat/completions", headers=headers, json=payload).status_code == 200
    assert client.post("/v1/chat/completions", headers=headers, json=payload).status_code == 200
    assert (
        client.post(
            "/v1/chat/completions",
            headers=_build_headers(
                session_id="sess-2",
                outer_turn_id="1",
                step_index=2,
                request_id="req-2",
            ),
            json={
                "model": "hermes-local",
                "messages": [
                    {"role": "user", "content": "start"},
                    {"role": "tool", "content": "tool ok"},
                ],
            },
        ).status_code
        == 200
    )

    _wait_for(lambda: output_queue.qsize() == 1)
    _, group = output_queue.get_nowait()
    assert group[0].response_text == "first answer"
    assert len(server._seen_request_ids["sess-2"]) == 2


def test_rl_server_overwrites_same_logical_step_on_new_request_id(tmp_path):
    output_queue: queue.Queue = queue.Queue()
    enabled = threading.Event()
    enabled.set()
    server = HermesRLServer(
        config=_make_config(tmp_path),
        output_queue=output_queue,
        submission_enabled=enabled,
        sampling_client=_FakeSamplingClient(["draft answer", "better answer", "final answer"]),
        tokenizer=_FakeTokenizer(),
        prm_scorer=_FakePRMScorer(score=1.0),
    )
    client = TestClient(server.app)

    assert (
        client.post(
            "/v1/chat/completions",
            headers=_build_headers(
                session_id="sess-3",
                outer_turn_id="1",
                step_index=1,
                request_id="req-1",
            ),
            json={"model": "hermes-local", "messages": _make_messages("user", "start")},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/chat/completions",
            headers=_build_headers(
                session_id="sess-3",
                outer_turn_id="1",
                step_index=1,
                request_id="req-1b",
            ),
            json={"model": "hermes-local", "messages": _make_messages("user", "start")},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/chat/completions",
            headers=_build_headers(
                session_id="sess-3",
                outer_turn_id="1",
                step_index=2,
                request_id="req-2",
            ),
            json={
                "model": "hermes-local",
                "messages": [
                    {"role": "user", "content": "start"},
                    {"role": "tool", "content": "tool ok"},
                ],
            },
        ).status_code
        == 200
    )

    _wait_for(lambda: output_queue.qsize() == 1)
    _, group = output_queue.get_nowait()
    assert group[0].response_text == "better answer"
    assert group[0].turn_num == 1


def test_non_main_requests_are_ignored_for_training(tmp_path):
    output_queue: queue.Queue = queue.Queue()
    enabled = threading.Event()
    enabled.set()
    server = HermesRLServer(
        config=_make_config(tmp_path),
        output_queue=output_queue,
        submission_enabled=enabled,
        sampling_client=_FakeSamplingClient(["helper answer"]),
        tokenizer=_FakeTokenizer(),
        prm_scorer=_FakePRMScorer(score=1.0),
    )
    client = TestClient(server.app)

    response = client.post(
        "/v1/chat/completions",
        headers=_build_headers(
            session_id="sess-4",
            outer_turn_id="1",
            step_index=1,
            request_id="req-1",
            turn_type="ignore",
        ),
        json={"model": "hermes-local", "messages": _make_messages("user", "summarize this")},
    )
    assert response.status_code == 200
    time.sleep(0.05)
    assert output_queue.qsize() == 0


def test_opd_server_emits_teacher_logprob_sample(tmp_path):
    output_queue: queue.Queue = queue.Queue()
    enabled = threading.Event()
    enabled.set()
    server = HermesOPDServer(
        config=_make_config(tmp_path),
        output_queue=output_queue,
        submission_enabled=enabled,
        sampling_client=_FakeSamplingClient(["step one", "step two"]),
        tokenizer=_FakeTokenizer(),
        opd_scorer=_FakeOPDScorer(accepted=True, eval_score=1.0),
    )
    client = TestClient(server.app)

    assert (
        client.post(
            "/v1/chat/completions",
            headers=_build_headers(
                session_id="sess-opd",
                outer_turn_id="1",
                step_index=1,
                request_id="req-1",
            ),
            json={"model": "hermes-local", "messages": _make_messages("user", "start")},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/chat/completions",
            headers=_build_headers(
                session_id="sess-opd",
                outer_turn_id="1",
                step_index=2,
                request_id="req-2",
            ),
            json={
                "model": "hermes-local",
                "messages": [
                    {"role": "user", "content": "start"},
                    {"role": "tool", "content": "tool ok"},
                ],
            },
        ).status_code
        == 200
    )

    _wait_for(lambda: output_queue.qsize() == 1)
    _, group = output_queue.get_nowait()
    sample = group[0]
    assert sample.reward == 1.0
    assert sample.teacher_logprobs == [-0.2] * len(sample.response_tokens)


def test_combine_server_routes_rl_only_sample(tmp_path):
    output_queue: queue.Queue = queue.Queue()
    enabled = threading.Event()
    enabled.set()
    server = HermesCombineServer(
        config=_make_config(tmp_path),
        output_queue=output_queue,
        submission_enabled=enabled,
        sampling_client=_FakeSamplingClient(["step one", "step two"]),
        tokenizer=_FakeTokenizer(),
        scorer=_FakeOPDScorer(accepted=False, eval_score=-1.0),
    )
    client = TestClient(server.app)

    assert (
        client.post(
            "/v1/chat/completions",
            headers=_build_headers(
                session_id="sess-combine",
                outer_turn_id="1",
                step_index=1,
                request_id="req-1",
            ),
            json={"model": "hermes-local", "messages": _make_messages("user", "start")},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/chat/completions",
            headers=_build_headers(
                session_id="sess-combine",
                outer_turn_id="1",
                step_index=2,
                request_id="req-2",
            ),
            json={
                "model": "hermes-local",
                "messages": [
                    {"role": "user", "content": "start"},
                    {"role": "tool", "content": "tool ok"},
                ],
            },
        ).status_code
        == 200
    )

    _wait_for(lambda: output_queue.qsize() == 1)
    _, group = output_queue.get_nowait()
    sample = group[0]
    assert sample.sample_type == "rl"
    assert sample.reward == -1.0
    assert sample.teacher_logprobs == sample.response_logprobs
