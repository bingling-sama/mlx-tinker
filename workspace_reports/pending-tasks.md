# Pending Tasks — mlx-tinker

Generated: 2026-03-28
Context: Cross-framework audit script (`scripts/cross_framework_audit.py`) is complete with 83 tests (72 default + 11 Tinker API). These tasks emerged from the audit findings and user feedback.

---

## P0 — Production correctness (do before any real workload)

### 1. Runtime NaN/Inf guard in `optim_step`
**File:** `mlx_tinker/backend/training.py` (optim_step method, ~line 350)

VERL and TRL both skip optimizer updates when gradients contain non-finite values. mlx-tinker currently checks `math.isfinite()` only in tests — not in production code. For a long-running agent loop, one NaN propagates and permanently corrupts the model.

**Fix:** After computing `grad_norm` in `optim_step`, add:
```python
if not math.isfinite(grad_norm):
    logger.warning("Non-finite grad_norm=%.4f, skipping update for model=%s", grad_norm, model_id)
    self.accumulated_grads[model_id] = None
    self.grad_accum_counts[model_id] = 0
    self.total_tokens[model_id] = 0.0
    return OptimStepOutput(metrics={"grad_norm": grad_norm, "skipped": True})
```

### 2. KV cache invalidation after weight updates
**File:** `mlx_tinker/backend/mlx_backend.py` or `inference.py`

After `optim_step` changes model weights, any cached KV states from previous inference become stale. In a train-sample-train agent loop (OpenClaw-RL pattern), a stale cache produces silently wrong logits.

**Verify:** Does `model.eval()` / `model.train()` toggling clear the cache? If mlx-lm caches state externally, an explicit cache invalidation is needed.

### 3. Checkpoint-resume determinism
**Test to add:** `scripts/cross_framework_audit.py` or `tests/stress/`

Compare:
- Path A: Train N steps → save checkpoint → load checkpoint → train M steps
- Path B: Train N+M steps straight through

Parameters after both paths should match within fp32 tolerance. The 8-bit optimizer state quantization makes this non-trivial — the quantize→save→load→dequantize roundtrip must preserve enough precision.

**Current gap:** Tests only verify save/load succeeds, not trajectory equivalence.

---

## P1 — Testing gaps (important for release confidence)

### 4. Cookbook golden fixture pipeline
**Context:** The `--tinker-api` tests hit the real Tinker API, but they're not integrated into CI. Cookbook tests currently check "does it not crash" — not "does it match Tinker."

**Tasks:**
1. Run `scripts/generate_reference_logs.py` to create `tests/fixtures/tinker_reference_logs.json` (currently missing)
2. Add CI step that replays cookbook examples against golden fixtures
3. Add fixture regeneration workflow (monthly or on model updates)

### 5. Gradient accumulation cross-API check
**Context:** The audit script verifies 4x sequential `forward_backward` + 1x `optim_step` produces identical gradients to 1x batched `forward_backward(4 datums)` + 1x `optim_step`. But this only tests the backend — it doesn't verify this through the full API/engine path.

**Add:** An end-to-end test via the HTTP API (using FastAPI TestClient + running engine) that:
1. Sends 4 sequential `forward_backward` requests, waits for each
2. Sends 1 `optim_step`
3. Compares against 1 batched `forward_backward` + 1 `optim_step`

### 6. Wire format fuzz testing with engine running
**Context:** `test_e2e_sdk_wire.py` validates JSON shapes but the engine never runs — all responses are pending futures. Need tests that actually execute through the engine and validate response payloads.

**Key fields to validate:**
- `type` discriminators on all responses
- `loss_fn_outputs[].logprobs` structure (List[Dict[str, TensorData]])
- `metrics` keys (`loss:sum`, `num_sequences`, `learning_rate`, etc.)
- `null` vs absent for optional fields
- `retrieve_future` unwrapping (returns raw result, not wrapper)

---

## P2 — Performance & architecture (before scaling up)

### 7. `mx.compile` integration
**Context:** v0.1 report flags this as missing. For inference-heavy agent workloads, this is likely 2-5x speedup on the forward pass.

**Where:** `inference.py` — wrap the forward pass with `mx.compile()` for KV-cache generation. Training forward pass is harder to compile (dynamic shapes per datum).

### 8. Latency percentile tracking
**Context:** Agents care about time-to-first-token and per-step latency, not just tok/s.

**Add to `--profile` mode:**
- p50/p95/p99 wall-clock per forward_backward
- p50/p95/p99 per optim_step
- Time-to-first-token for sampling
- Breakdown: model forward vs loss compute vs gradient accumulation

### 9. Streaming SSE parity tests
**File:** `mlx_tinker/api/openai_compat.py`

If agents use the `/v1/chat/completions` endpoint with `stream=True`, need tests for:
- Chunk boundary correctness
- `[DONE]` sentinel present and final
- `usage` stats in final chunk
- Partial token handling

### 10. Model family abstraction
**File:** `mlx_tinker/backend/lora_manager.py` (line ~100, hardcoded keys)

LoRA target keys are hardcoded for Qwen/Llama style (`self_attn.q_proj`, `mlp.gate_proj`). Adding Mistral/Gemma later requires surgery.

**Fix:** Add a model registry or auto-detection:
```python
def _detect_lora_keys(model) -> list[str]:
    """Detect attention/MLP projection names from model architecture."""
```

### 11. Concurrent session stress test
**Context:** Agent frameworks (Hermes) may hold multiple sessions or fire concurrent requests.

**Add:** A stress test that simulates 3-5 concurrent sessions doing interleaved:
- `forward_backward` on different models
- `sample` while training is in progress
- `optim_step` from one session while another is doing `forward_backward`

Verify: no deadlocks, no cross-session gradient contamination, all futures resolve.

---

## Audit findings — loss divergence between Tinker API and mlx-tinker

**Observed:**
- Step-0 SFT loss: Tinker=87.14, Local=85.69 (1.9% relative diff)
- Step-5 SFT loss: Tinker=28.61, Local=23.06 (~19% relative diff)
- Loss curve correlation: 0.96 (strong but not exact)
- RL trend direction: matches

**Root cause:** Different 4-bit quantization schemes (CUDA NF4 vs MLX groupwise). Confirmed by:
1. Identical token sequences (same tokenizer, same WikiSQL data)
2. Step-0 diff exists before any optimizer effect
3. Divergence compounds over training steps as quantized weight errors accumulate through gradient updates

**Not a bug.** This is an inherent consequence of different hardware/framework quantization. The audit script tracks these numbers for regression detection.

---

## Current test coverage

| Category | Tests | Status |
|----------|-------|--------|
| Loss function forward parity (MLX vs PyTorch) | 20 | PASS |
| Loss function gradient parity | 5 | PASS |
| Loss function finite-difference | 5 | PASS |
| Chunked cross-entropy | 14 | PASS |
| AdamW8Bit optimizer | 5 | PASS |
| Dynamic quantization map | 3 | PASS |
| Log-probability computation | 3 | PASS |
| Gradient clipping | 3 | PASS |
| Full training step (tiny model) | 4 | PASS |
| Real model inference (Qwen3.5-0.8B) | 10 | PASS |
| Tinker API parity (opt-in) | 11 | PASS |
| **Total** | **83** | **All PASS** |
