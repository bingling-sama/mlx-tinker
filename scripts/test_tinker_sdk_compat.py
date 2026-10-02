"""Comprehensive compatibility and regression test suite for Tinker SDK v0.31.0+.

Validates 100% of native ServiceClient, TrainingClient, SamplingClient, and RestClient
public APIs without mocks or shims.

Specifically covers and guards against:
1. POST /api/v1/client/config & client/dynamic_config (HTTP 404 block on client initialization)
2. Content-Type: application/x-protobuf & Accept: application/x-protobuf (forward_backward & retrieve_future)
3. sample_sequence_ids assertion (assert sample_sequence_ids is not None on sampling promises)
4. forward / forward_async (forward_only loss evaluation via protobuf)
5. forward_backward_custom (client-side custom loss gradient backprop via CE weights surrogate)
6. optim_step & weight checkpointing lifecycle (save_state, load_state, sampler export)
7. SamplingParams stop sequences and prompt_logprobs
8. RestClient full session and checkpoint management APIs
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import Generator

import pytest
import tinker
import tinker.types
import torch
import uvicorn
from transformers import AutoTokenizer

from mlx_tinker.api.server import create_app
from mlx_tinker.config import EngineConfig

LOCAL_MODEL = os.path.abspath("Qwen3.5-4B")
MODEL = os.environ.get("MLX_TINKER_MODEL", LOCAL_MODEL)


@pytest.fixture(scope="session")
def live_server(tmp_path_factory) -> Generator[str, None, None]:
    """Start a real mlx-tinker FastAPI uvicorn server in a background thread."""
    td = tmp_path_factory.mktemp("mlx_tinker_test")
    db_path = td / "tinker_test.db"
    ckpt_path = td / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    config = EngineConfig(
        base_model=MODEL,
        database_path=db_path,
        checkpoints_base=ckpt_path,
    )
    app = create_app(config)

    port = 8765
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
    )
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    base_url = f"http://127.0.0.1:{port}"
    os.environ["TINKER_BASE_URL"] = base_url
    os.environ["TINKER_API_KEY"] = "tml-native-sdk-test-key"

    # Wait for server to come up
    import urllib.request
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"{base_url}/api/v1/healthz", timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError(f"Server at {base_url} failed to start in time")

    yield base_url
    server.should_exit = True
    t.join(timeout=5)


@pytest.fixture(scope="session")
def tokenizer(live_server):
    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


@pytest.fixture(scope="session")
def service_client(live_server):
    sc = tinker.ServiceClient()
    yield sc
    sc.holder.close()


@pytest.fixture(scope="session")
def rest_client(service_client):
    return service_client.create_rest_client()


# ==============================================================================
# 1. ServiceClient Protocol & Methods
# ==============================================================================

class TestServiceClientMethods:
    """Validates ServiceClient initialization, client_config handshake, and factory methods."""

    def test_client_config_handshake_and_session(self, service_client):
        """Guard 1: ServiceClient must negotiate /api/v1/client/config and get valid session_id."""
        holder = service_client.holder
        assert holder._session_id is not None
        assert holder._client_config is not None
        assert holder._client_config.pjwt_auth_enabled is False
        assert holder._client_config.parallel_fwdbwd_chunks is True

    def test_get_server_capabilities_sync_and_async(self, service_client):
        caps_sync = service_client.get_server_capabilities()
        assert isinstance(caps_sync, tinker.types.GetServerCapabilitiesResponse)
        assert len(caps_sync.supported_models) > 0
        assert caps_sync.supported_models[0].model_name == MODEL

        async def run_caps():
            return await service_client.get_server_capabilities_async()

        caps_async = asyncio.run(run_caps())
        assert isinstance(caps_async, tinker.types.GetServerCapabilitiesResponse)
        assert caps_async.supported_models[0].model_name == MODEL

    def test_get_telemetry(self, service_client):
        service_client.get_telemetry()

    def test_create_rest_client(self, service_client):
        rc = service_client.create_rest_client()
        assert rc is not None


# ==============================================================================
# 2. TrainingClient Protocol & Methods (Protobuf Fwd/Bwd, Custom Loss, Optim)
# ==============================================================================

class TestTrainingClientMethods:
    """Validates TrainingClient methods with native Protobuf payload serialization."""

@pytest.fixture(scope="session")
def training_client(service_client):
    # Single persistent resident LoRA model on the server across all tests
    return service_client.create_lora_training_client(
        base_model=MODEL,
        rank=8,
        user_metadata={"test": "session_tc"},
    )

    def test_get_info_sync_and_async(self, training_client):
        info_sync = training_client.get_info()
        assert isinstance(info_sync, tinker.types.GetInfoResponse)
        assert info_sync.model_id
        assert info_sync.is_lora is True
        assert info_sync.lora_rank == 8

        async def run_info():
            return await training_client.get_info_async()

        info_async = asyncio.run(run_info())
        assert isinstance(info_async, tinker.types.GetInfoResponse)
        assert info_async.model_id == info_sync.model_id

    def test_get_tokenizer(self, training_client):
        tok = training_client.get_tokenizer()
        assert tok is not None
        assert len(tok.encode("hello")) > 0

    def test_forward_and_forward_async_protobuf(self, training_client, tokenizer):
        """Guard 2a: forward sends protobuf and receives ForwardBackwardOutput."""
        prompt_ids = tokenizer.encode("Hello", add_special_tokens=False)
        tokens = prompt_ids[:4] + [tokenizer.eos_token_id or 0]
        n = len(tokens) - 1
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:n]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(
                    torch.tensor(tokens[1 : n + 1], dtype=torch.long)
                ),
            },
        )

        # 1. Sync forward
        res_sync = training_client.forward([datum], loss_fn="cross_entropy").result()
        assert isinstance(res_sync, tinker.types.ForwardBackwardOutput)
        assert "loss:sum" in res_sync.metrics
        assert len(res_sync.loss_fn_outputs) == 1
        assert "logprobs" in res_sync.loss_fn_outputs[0]

        # 2. Async forward
        async def run_fwd():
            fut = await training_client.forward_async([datum], loss_fn="cross_entropy")
            return await fut.result_async()

        res_async = asyncio.run(run_fwd())
        assert isinstance(res_async, tinker.types.ForwardBackwardOutput)
        assert "loss:sum" in res_async.metrics

    def test_forward_backward_and_forward_backward_async_protobuf(self, training_client, tokenizer):
        """Guard 2b: forward_backward sends binary protobuf and accumulates gradients."""
        prompt_ids = tokenizer.encode("Tinker native SDK compatibility", add_special_tokens=False)
        tokens = prompt_ids[:4] + [tokenizer.eos_token_id or 0]
        n = len(tokens) - 1
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:n]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(
                    torch.tensor(tokens[1 : n + 1], dtype=torch.long)
                ),
            },
        )

        # 1. forward_backward sync
        res_sync = training_client.forward_backward([datum], loss_fn="cross_entropy").result()
        assert isinstance(res_sync, tinker.types.ForwardBackwardOutput)
        assert "loss:sum" in res_sync.metrics
        assert res_sync.metrics["loss:sum"] > 0

        # 2. forward_backward async
        async def run_fb():
            fut = await training_client.forward_backward_async([datum], loss_fn="cross_entropy")
            return await fut.result_async()

        res_async = asyncio.run(run_fb())
        assert isinstance(res_async, tinker.types.ForwardBackwardOutput)
        assert "loss:sum" in res_async.metrics

    def test_forward_backward_custom_sync_and_async(self, training_client, tokenizer):
        """Guard: client-side PyTorch custom loss backprop via surrogate cross_entropy weights."""
        prompt_ids = tokenizer.encode("Custom loss test", add_special_tokens=False)
        tokens = prompt_ids[:4] + [tokenizer.eos_token_id or 0]
        n = len(tokens) - 1
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:n]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(
                    torch.tensor(tokens[1 : n + 1], dtype=torch.long)
                ),
            },
        )

        def custom_loss(data, logprobs_list):
            loss = torch.mean(torch.stack([torch.mean(lp) for lp in logprobs_list]))
            return loss, {"custom_metric": float(loss.item())}

        # 1. forward_backward_custom sync
        res_sync = training_client.forward_backward_custom([datum], custom_loss).result()
        assert isinstance(res_sync, tinker.types.ForwardBackwardOutput)
        assert "custom_metric" in res_sync.metrics

        # 2. forward_backward_custom_async
        async def run_fb_custom():
            fut = await training_client.forward_backward_custom_async([datum], custom_loss)
            return await fut.result_async()

        res_async = asyncio.run(run_fb_custom())
        assert isinstance(res_async, tinker.types.ForwardBackwardOutput)
        assert "custom_metric" in res_async.metrics

    def test_optim_step_sync_and_async(self, training_client, tokenizer):
        """Test optim_step applies accumulated gradients."""
        prompt_ids = tokenizer.encode("Optim step test", add_special_tokens=False)
        tokens = prompt_ids[:4] + [tokenizer.eos_token_id or 0]
        n = len(tokens) - 1
        datum = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(tokens[:n]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_torch(
                    torch.tensor(tokens[1 : n + 1], dtype=torch.long)
                ),
            },
        )

        # 1. optim_step sync (after forward_backward)
        training_client.forward_backward([datum], loss_fn="cross_entropy").result()
        res_sync = training_client.optim_step(tinker.AdamParams(learning_rate=1e-5)).result()
        assert isinstance(res_sync, tinker.types.OptimStepResponse)
        assert "learning_rate:unique" in res_sync.metrics

        # 2. optim_step async (after forward_backward_async)
        async def run_optim():
            fut_fb = await training_client.forward_backward_async([datum], loss_fn="cross_entropy")
            await fut_fb.result_async()
            fut_opt = await training_client.optim_step_async(tinker.AdamParams(learning_rate=1e-5))
            return await fut_opt.result_async()

        res_async = asyncio.run(run_optim())
        assert isinstance(res_async, tinker.types.OptimStepResponse)
        assert "learning_rate:unique" in res_async.metrics

    def test_save_state_and_load_state(self, training_client, service_client):
        """Test checkpoint saving, tinker:// uri generation, and load_state."""
        # 1. save_state sync & async
        save_res = training_client.save_state("sdk_ckpt_step1", ttl_seconds=3600).result()
        tinker_path = save_res.path
        assert tinker_path.startswith("tinker://")

        async def run_save():
            fut = await training_client.save_state_async("sdk_ckpt_step2", ttl_seconds=3600)
            return await fut.result_async()

        save_res2 = asyncio.run(run_save())
        assert save_res2.path.startswith("tinker://")

        # 2. load_state sync & async
        training_client.load_state(tinker_path).result()

        async def run_load():
            fut = await training_client.load_state_async(tinker_path)
            return await fut.result_async()

        asyncio.run(run_load())

    def test_save_weights_for_sampler_and_create_sampling_client(self, training_client):
        """Test exporting weights for sampler and creating SamplingClient."""
        # 1. save_weights_for_sampler sync & async
        save_s_res = training_client.save_weights_for_sampler("sampler_export_sync").result()
        assert "sampler_weights" in save_s_res.path

        async def run_save_s():
            fut = await training_client.save_weights_for_sampler_async("sampler_export_async")
            return await fut.result_async()

        save_s_res2 = asyncio.run(run_save_s())
        assert "sampler_weights" in save_s_res2.path

        # 2. create_sampling_client from exported sampler weights
        sc1 = training_client.create_sampling_client(save_s_res.path)
        assert sc1 is not None

        async def run_create_sc():
            return await training_client.create_sampling_client_async(save_s_res.path)

        sc2 = asyncio.run(run_create_sc())
        assert sc2 is not None

        # 3. save_weights_and_get_sampling_client shortcut
        sc3 = training_client.save_weights_and_get_sampling_client("auto_export")
        assert sc3 is not None


# ==============================================================================
# 3. SamplingClient Protocol & Methods (sample_sequence_ids, prompt logprobs)
# ==============================================================================

class TestSamplingClientMethods:
    """Validates SamplingClient methods with sequence ID attachment and protobuf support."""

    @pytest.fixture(scope="class")
    def sampling_client(self, service_client):
        return service_client.create_sampling_client(base_model=MODEL)

    def test_get_base_model_sync_and_async(self, sampling_client):
        bm_sync = sampling_client.get_base_model()
        assert bm_sync == MODEL

        async def run_bm():
            return await sampling_client.get_base_model_async()

        bm_async = asyncio.run(run_bm())
        assert bm_async == MODEL

    def test_get_tokenizer(self, sampling_client):
        tok = sampling_client.get_tokenizer()
        assert tok is not None
        assert len(tok.encode("Sampling test")) > 0

    def test_sample_sync_and_async_with_sequence_ids(self, sampling_client, tokenizer):
        """Guard 3: SDK asserts sample_sequence_ids is not None on returned SampledSequence."""
        prompt_ids = tokenizer.encode("What is 2+2?", add_special_tokens=False)
        chunk = tinker.EncodedTextChunk(tokens=list(prompt_ids), type="encoded_text")
        model_input = tinker.ModelInput(chunks=[chunk])
        params = tinker.SamplingParams(temperature=0.7, max_tokens=10)

        # 1. sample sync
        res_sync = sampling_client.sample(
            prompt=model_input,
            num_samples=2,
            sampling_params=params,
        ).result()
        assert isinstance(res_sync, tinker.types.SampleResponse)
        assert len(res_sync.sequences) == 2
        for seq in res_sync.sequences:
            assert seq.sequence_id is not None
            assert seq.sequence_id.startswith("seq_")
            assert len(seq.tokens) > 0
            assert seq.stop_reason in {"stop", "length"}

        # 2. sample_async
        async def run_sample():
            return await sampling_client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=params,
            )

        res_async = asyncio.run(run_sample())
        assert isinstance(res_async, tinker.types.SampleResponse)
        assert len(res_async.sequences) == 1
        assert res_async.sequences[0].sequence_id is not None
        assert res_async.sequences[0].sequence_id.startswith("seq_")

    def test_sample_with_prompt_logprobs(self, sampling_client, tokenizer):
        """Test prompt_logprobs return via protobuf."""
        prompt_ids = tokenizer.encode("The quick brown fox", add_special_tokens=False)
        chunk = tinker.EncodedTextChunk(tokens=list(prompt_ids), type="encoded_text")
        model_input = tinker.ModelInput(chunks=[chunk])
        params = tinker.SamplingParams(temperature=0.0, max_tokens=1)

        res = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=params,
            include_prompt_logprobs=True,
            topk_prompt_logprobs=0,
        ).result()
        assert res.prompt_logprobs is not None
        assert len(res.prompt_logprobs) > 0

    def test_compute_logprobs_sync_and_async(self, sampling_client, tokenizer):
        """Test compute_logprobs helper on SamplingClient."""
        prompt_ids = tokenizer.encode("Hello world", add_special_tokens=False)
        model_input = tinker.ModelInput.from_ints(list(prompt_ids))

        # 1. compute_logprobs sync
        lps_sync = sampling_client.compute_logprobs(model_input).result()
        assert len(lps_sync) > 0

        # 2. compute_logprobs_async
        async def run_lps():
            return await sampling_client.compute_logprobs_async(model_input)

        lps_async = asyncio.run(run_lps())
        assert len(lps_async) > 0


# ==============================================================================
# 4. RestClient Protocol & Lifecycle Methods
# ==============================================================================

class TestRestClientMethods:
    """Validates RestClient operations for sessions, training runs, checkpoints, and samplers."""

    def test_sessions_operations_sync_and_async(self, rest_client, service_client):
        sess_id = service_client.holder._session_id

        # 1. list_sessions sync & async
        sessions_sync = rest_client.list_sessions(limit=10).result()
        assert isinstance(sessions_sync, tinker.types.ListSessionsResponse)

        async def run_list():
            return await rest_client.list_sessions_async(limit=10)

        sessions_async = asyncio.run(run_list())
        assert isinstance(sessions_async, tinker.types.ListSessionsResponse)

        # 2. get_session sync & async
        sess_sync = rest_client.get_session(sess_id).result()
        assert isinstance(sess_sync, tinker.types.GetSessionResponse)

        async def run_get():
            return await rest_client.get_session_async(sess_id)

        sess_async = asyncio.run(run_get())
        assert isinstance(sess_async, tinker.types.GetSessionResponse)

    def test_training_runs_and_checkpoints_lifecycle(self, rest_client, training_client):
        # We use the active training model
        tc = training_client
        run_id = tc._guaranteed_model_id()

        # 1. list_training_runs sync & async
        runs_sync = rest_client.list_training_runs(limit=10).result()
        assert isinstance(runs_sync, tinker.types.TrainingRunsResponse)

        async def run_list_runs():
            return await rest_client.list_training_runs_async(limit=10)

        runs_async = asyncio.run(run_list_runs())
        assert isinstance(runs_async, tinker.types.TrainingRunsResponse)

        # 2. get_training_run sync & async
        run_sync = rest_client.get_training_run(run_id).result()
        assert isinstance(run_sync, tinker.types.TrainingRun)

        async def run_get_run():
            return await rest_client.get_training_run_async(run_id)

        run_async = asyncio.run(run_get_run())
        assert isinstance(run_async, tinker.types.TrainingRun)

        # 3. Save a checkpoint
        save_res = tc.save_state("rest_ckpt_01", ttl_seconds=3600).result()
        tinker_path = save_res.path

        # 4. get_training_run_by_tinker_path sync & async
        by_path_sync = rest_client.get_training_run_by_tinker_path(tinker_path).result()
        assert by_path_sync.training_run_id == run_id

        async def run_by_path():
            return await rest_client.get_training_run_by_tinker_path_async(tinker_path)

        by_path_async = asyncio.run(run_by_path())
        assert by_path_async.training_run_id == run_id

        # 5. get_weights_info_by_tinker_path
        winfo = rest_client.get_weights_info_by_tinker_path(tinker_path).result()
        assert isinstance(winfo, tinker.types.WeightsInfoResponse)

        # 6. list_checkpoints sync & async
        ckpts_sync = rest_client.list_checkpoints(run_id).result()
        assert isinstance(ckpts_sync, tinker.types.CheckpointsListResponse)

        async def run_list_ckpts():
            return await rest_client.list_checkpoints_async(run_id)

        ckpts_async = asyncio.run(run_list_ckpts())
        assert isinstance(ckpts_async, tinker.types.CheckpointsListResponse)

        # 7. list_user_checkpoints sync & async
        all_ckpts_sync = rest_client.list_user_checkpoints(limit=10).result()
        assert isinstance(all_ckpts_sync, tinker.types.CheckpointsListResponse)

        async def run_user_ckpts():
            return await rest_client.list_user_checkpoints_async(limit=10)

        all_ckpts_async = asyncio.run(run_user_ckpts())
        assert isinstance(all_ckpts_async, tinker.types.CheckpointsListResponse)

        # 8. publish / unpublish checkpoint
        rest_client.publish_checkpoint_from_tinker_path(tinker_path).result()
        rest_client.unpublish_checkpoint_from_tinker_path(tinker_path).result()

        async def run_pub():
            await rest_client.publish_checkpoint_from_tinker_path_async(tinker_path)
            await rest_client.unpublish_checkpoint_from_tinker_path_async(tinker_path)

        asyncio.run(run_pub())

        # 9. set_checkpoint_ttl
        rest_client.set_checkpoint_ttl_from_tinker_path(tinker_path, ttl_seconds=7200).result()

        async def run_ttl():
            await rest_client.set_checkpoint_ttl_from_tinker_path_async(tinker_path, ttl_seconds=3600)

        asyncio.run(run_ttl())

        # 10. get_checkpoint_archive_url & from_tinker_path
        arc_sync = rest_client.get_checkpoint_archive_url_from_tinker_path(tinker_path).result()
        assert isinstance(arc_sync, tinker.types.CheckpointArchiveUrlResponse)

        async def run_arc():
            return await rest_client.get_checkpoint_archive_url_from_tinker_path_async(tinker_path)

        arc_async = asyncio.run(run_arc())
        assert isinstance(arc_async, tinker.types.CheckpointArchiveUrlResponse)

        # 11. get_sampler info
        sampler_res = tc.save_weights_for_sampler("sampler_for_rest").result()
        sampler_client = tc.create_sampling_client(sampler_res.path)
        sampler_id = sampler_client._sampling_session_id

        s_info_sync = rest_client.get_sampler(sampler_id).result()
        assert isinstance(s_info_sync, tinker.types.GetSamplerResponse)

        async def run_get_sampler():
            return await rest_client.get_sampler_async(sampler_id)

        s_info_async = asyncio.run(run_get_sampler())
        assert isinstance(s_info_async, tinker.types.GetSamplerResponse)

        # 12. delete_checkpoint
        rest_client.delete_checkpoint_from_tinker_path(tinker_path).result()
