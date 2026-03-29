# mlx-tinker

**Drop-in Tinker replacement for Apple Silicon.** Train and run Qwen3.5 models locally with QLoRA, gradient checkpointing, and full RL support — same SDK, same code, no cloud required.

mlx-tinker implements the [Tinker API](https://docs.tinker.ai) on top of Apple's [MLX](https://github.com/ml-explore/mlx) framework. Point your `tinker.ServiceClient` at `localhost` and everything just works — SFT, PPO, CISPO, DRO, importance sampling, checkpointing, and inference.

## Quick Start

```bash
# Install (requires Python 3.12+, macOS with Apple Silicon)
pip install uv && git clone https://github.com/ojus1/mlx-tinker.git && cd mlx-tinker && uv sync
```

```bash
# Start the server
uv run python -m mlx_tinker --model Qwen/Qwen3.5-0.8B
```

```bash
# That's it. Use the Tinker SDK exactly as you would with cloud:
python -c "
import tinker
client = tinker.ServiceClient(base_url='http://localhost:8080', api_key='local')
print(client.healthz())
"
```

## Tinker Compatibility is Real

The only change is `base_url`. Everything else — training loops, RL pipelines, checkpointing, sampling — is identical:

```python
import tinker

# Tinker (official):  client = tinker.ServiceClient()
# mlx-tinker:
client = tinker.ServiceClient(base_url="http://localhost:8080", api_key="local")

# Create a QLoRA training client
training = await client.create_lora_training_client_async(
    base_model="Qwen/Qwen3.5-4B", rank=8
)

# SFT training loop
for batch in train_data:
    await training.forward_backward_async(batch, loss_fn="cross_entropy")
    await training.optim_step_async(tinker.AdamParams(learning_rate=1e-4))

# Get a sampling client from the trained model
sampler = await training.save_weights_and_get_sampling_client_async()
response = await sampler.sample_async(
    prompt=tinker.ModelInput.from_ints(prompt_tokens),
    num_samples=1,
    sampling_params=tinker.SamplingParams(temperature=0.0, max_tokens=128),
)
print(tokenizer.decode(response.sequences[0].tokens))
```

**RL works too** — importance sampling, PPO, the full loop:

```python
# RL: sample rollouts, compute rewards, train with advantages
sampler = await training.save_weights_and_get_sampling_client_async()
rollouts = await sampler.sample_async(prompt, num_samples=8,
    sampling_params=tinker.SamplingParams(temperature=0.8, max_tokens=128))

rewards = [compute_reward(seq) for seq in rollouts.sequences]
advantages = [r - sum(rewards) / len(rewards) for r in rewards]

rl_batch = [build_rl_datum(seq, advantage) for seq, advantage in zip(rollouts.sequences, advantages)]
await training.forward_backward_async(rl_batch, loss_fn="importance_sampling")
await training.optim_step_async(tinker.AdamParams(learning_rate=5e-5))
```

## Benchmark: Tinker (official) vs mlx-tinker

50-step SFT on WikiSQL with QLoRA (rank-8, 4-bit quantization, batch_size=2):

![SFT Benchmark](assets/benchmark_sft_comparison.png)

| Metric | Tinker (official) 4B | mlx-tinker 4B (M4 MBP 24GB) | mlx-tinker 9B (M4 MBP 24GB) |
|--------|:----------------------:|:-----------------------------:|:-----------------------------:|
| Initial loss | 22.34 | 24.23 | 17.90 |
| Final loss | 0.07 | 0.43 | 0.01 |
| Avg step time | 3.7s | 5.3s | 8.7s |
| Total train time | 184s | 267s | 436s |
| Post-train eval accuracy | 33% | 30% | 29% |

Both backends converge on WikiSQL SFT with comparable accuracy. mlx-tinker trades some per-step speed for running entirely on your Mac — no cloud costs, no network latency, your data stays local. And Tinker (official) doesn't even support 9B — mlx-tinker lets you train larger models that the cloud can't.

## Features

### QLoRA with Gradient Checkpointing

4-bit quantized base model with LoRA adapters. Gradient checkpointing recomputes activations during the backward pass, dramatically reducing memory usage. All testing and development was done on a M4 MacBook Pro 24GB.

### Five Loss Functions

| Loss | Use Case | Formula |
|------|----------|---------|
| `cross_entropy` | SFT | `(-logp * w).sum()` |
| `importance_sampling` | Off-policy RL | `-(ratio * adv).sum()` |
| `ppo` | PPO-clip | `-min(ratio*adv, clip(ratio)*adv).sum()` |
| `cispo` | Conservative IS | `-(sg(clip(ratio)) * logp * adv).sum()` |
| `dro` | Direct Reward Opt | `-(logp*adv - 0.5*beta*(logp-old_lp)^2).sum()` |

All losses use **sum reduction** to match Tinker's official formulas.

### Supported Models

Non-MoE Qwen3.5 family:

| Model | Status |
|-------|:------:|
| Qwen/Qwen3.5-0.8B | Tested |
| Qwen/Qwen3.5-4B | Tested |
| Qwen/Qwen3.5-9B | Tested |
| Tesslate/OmniCoder-9B | Tested |

### OpenAI-Compatible Inference

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "default", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Architecture

```
tinker.ServiceClient
        |
   FastAPI Server (api/)         19 endpoints, full Tinker wire protocol
        |
   Async Engine (engine/)        100ms polling cycles, barrier-aware batching
        |
   MLX Backend (backend/)        QLoRA, training, inference, checkpointing
        |
   Apple Silicon (Metal GPU)     Unified memory, no CUDA
```

- **API layer** — FastAPI with Tinker-compatible request/response models, session management, and OpenAI-compat endpoints
- **Engine** — Async polling loop that batches compatible requests (forward_backward, sample) and respects barriers (optim_step must wait for all pending forward_backward)
- **Backend** — MLX compute: `nn.value_and_grad` for training, KV-cache inference, chunked cross-entropy, 8-bit Adam optimizer

## Run the Tests

```bash
# Unit tests (fast, no model download needed for most)
uv run pytest tests/ -k "not stress and not cookbook"

# SFT + RL cookbook tests (downloads Qwen3.5-0.8B, ~5 min)
uv run pytest tests/cookbook/ -m cookbook -v

# Full benchmark
uv run python scripts/run_benchmark.py --mlx-only --sft-steps 50
```

All 10 cookbook tests pass, covering SFT convergence, RL with importance sampling, PPO, tool-use RL, and capability proofs (SFT + RL progression on exact-match tasks).

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- Developed and tested on a M4 MacBook Pro 24GB

## Project Structure

```
mlx_tinker/
  api/        FastAPI server + Tinker endpoints + OpenAI compat
  engine/     Async request scheduler + dispatcher
  backend/    MLX training, inference, QLoRA, loss functions, checkpointing
  db/         SQLModel ORM with async SQLite (WAL mode)
  types.py    Tinker-compatible enums and data types
  config.py   Pydantic configuration
tests/
  cookbook/    SFT + RL end-to-end workflow tests
  stress/     Cross-framework parity tests (MLX vs PyTorch)
scripts/
  run_benchmark.py           Cloud vs local benchmark
  generate_readme_plot.py    Generate the comparison plot above
```
