from __future__ import annotations

import json

from fastapi.testclient import TestClient

from integrations.openclaw_rl_local.api_server import OpenClawLocalProxy
from integrations.openclaw_rl_local.curriculum import evaluate_task_output, load_curriculum
from integrations.openclaw_rl_local.data_formatter import TrainingSample, batch_to_datums
from integrations.openclaw_rl_local.runtime import (
    ContainerRuntime,
    build_runtime_urls,
    resolve_provider_base_url,
)
from tests.helpers import FakeTokenizer


class _FakeSequence:
    def __init__(self, tokens, logprobs, stop_reason="stop"):
        self.tokens = tokens
        self.logprobs = logprobs
        self.stop_reason = stop_reason


class _FakeSampleResponse:
    def __init__(self, sequence):
        self.sequences = [sequence]


class _FakeSamplingClient:
    def __init__(self, sequence):
        self.sequence = sequence
        self.calls = []

    async def sample_async(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeSampleResponse(self.sequence)


def test_curriculum_loads_and_splits():
    tasks = load_curriculum(
        __import__("pathlib").Path(
            "integrations/openclaw_rl_local/curriculum/wildclaw_text_v1.yaml"
        )
    )
    assert len(tasks) == 8
    assert any(task.trainable for task in tasks)
    assert any(not task.trainable for task in tasks)


def test_evaluate_task_output_binary_checks():
    task = load_curriculum(
        __import__("pathlib").Path(
            "integrations/openclaw_rl_local/curriculum/wildclaw_text_v1.yaml"
        )
    )[0]
    passing = evaluate_task_output(task, "Themes: ... Risks: ... Next steps: memory risk")
    failing = evaluate_task_output(task, "short answer")
    assert passing.passed is True
    assert passing.reward == 1.0
    assert failing.passed is False
    assert failing.reward == -1.0


def test_batch_to_datums_preserves_prompt_mask():
    sample = TrainingSample(
        task_id="task",
        prompt_tokens=[1, 2, 3],
        response_tokens=[4, 5],
        response_logprobs=[-0.1, -0.2],
        reward=1.0,
        prompt_text="prompt",
        response_text="response",
    )
    datum = batch_to_datums([sample])[0]
    assert datum.loss_fn_inputs["target_tokens"].data == [2, 3, 4, 5]
    assert datum.loss_fn_inputs["advantages"].data[:2] == [0.0, 0.0]
    assert datum.loss_fn_inputs["advantages"].data[2:] == [1.0, 1.0]


def test_proxy_records_prompt_and_logprobs():
    tokenizer = FakeTokenizer()
    sampling_client = _FakeSamplingClient(_FakeSequence(tokens=[1, 2], logprobs=[-0.3, -0.4]))
    proxy = OpenClawLocalProxy(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        served_model_name="qwen3.5-local",
        api_key=None,
    )
    client = TestClient(proxy.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3.5-local",
            "messages": [{"role": "developer", "content": "be helpful"}, {"role": "user", "content": "hello"}],
            "temperature": 0.0,
            "max_tokens": 8,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["message"]["role"] == "assistant"
    records = proxy.records_since(0)
    assert len(records) == 1
    assert records[0].response_logprobs == (-0.3, -0.4)
    assert sampling_client.calls[0]["num_samples"] == 1


def test_runtime_provider_base_url_defaults():
    assert resolve_provider_base_url(ContainerRuntime.PODMAN, 30000) == "http://host.containers.internal:30000/v1"
    assert resolve_provider_base_url(ContainerRuntime.DOCKER, 30000) == "http://host.docker.internal:30000/v1"
    urls = build_runtime_urls(
        runtime=ContainerRuntime.PODMAN,
        mlx_tinker_base_url="http://127.0.0.1:8010",
        proxy_host="127.0.0.1",
        proxy_port=30000,
    )
    assert urls.proxy_base_url == "http://127.0.0.1:30000/v1"


def test_rendered_provider_config_shape():
    from subprocess import check_output

    payload = json.loads(
        check_output(
            ["python3", "scripts/render_openclaw_provider_config.py", "--runtime", "podman"],
            text=True,
        )
    )
    provider = payload["providers"]["mlx-tinker-local"]
    assert provider["api"] == "openai-completions"
    assert provider["baseUrl"] == "http://host.containers.internal:30000/v1"
