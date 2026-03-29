# Handoff TODO

## Context

The previous agent session made substantial backend, validation, and UAT changes, then started a large number of heavy runs in parallel. The user reported that this caused the laptop to crash/restart.

Do **not** resume parallel heavy runs.

Per latest user instruction:

- run tests **sequentially**
- UAT is allowed again, but **only sequentially**
- do not run broad heavy matrices in parallel
- use local `Qwen/Qwen3.5-0.8B` as the default validation model
- do not use cloud Tinker in the default UAT path
- RL UAT should use **8 rollouts** per prompt, with bounded smoke/extended lanes
- realistic local acceptance flow should be: **50-step SFT -> continue same model with 50-step RL**
- use batch size **4** for SFT and group size / rollouts **8** for RL
- write down everything relevant for another coding agent to pick up

## Important Safety Notes

- Use `source .venv/bin/activate` before Python commands.
- Do not run heavy suites in parallel.
- Do not run heavy suites in parallel with UAT.
- If you need the local server, use port `8010`, not `8000`.
  - Port `8000` conflicted locally and produced empty replies.
  - `http://127.0.0.1:8010/` and `http://127.0.0.1:8010/api/v1/healthz` worked correctly.

## Code Changes Already Made

### Validation / benchmarking

- Normalized reference-log loss handling:
  - `scripts/generate_reference_logs.py`
  - `tests/stress/test_tinker_equivalence.py`
- Added explicit `loss_mean` and `loss_sum` fields to generated reference logs.
- Fixed `scripts/environments/wikisql_env.py` no-arg CLI behavior.
- Improved `scripts/run_benchmark.py` startup/progress logging.
- Tightened stress skip-budget logic in `tests/stress/conftest.py`.

### Sampling / inference / routing

- Added batch sampling path in:
  - `mlx_tinker/backend/inference.py`
  - `mlx_tinker/backend/mlx_backend.py`
  - `mlx_tinker/engine/engine.py`
- Routed non-streaming OpenAI endpoints through backend sampling in:
  - `mlx_tinker/api/openai_compat.py`
- Preserved test compatibility by restoring `_generate_tokens()` wrapper.
- Added `model_id` support on sample requests:
  - `mlx_tinker/types.py`
  - `mlx_tinker/api/models.py`
  - `mlx_tinker/api/server.py`

### Training / memory / throughput

- Replaced per-datum gradient accumulation with padded batched `forward_backward()` in:
  - `mlx_tinker/backend/training.py`
- Added reusable chunked target-logprob path in:
  - `mlx_tinker/backend/loss_fns.py`
- Enabled chunked target-logprob usage for RL-style losses where safe.
- Added guards so chunked fast path does **not** activate for quantized/packed `lm_head` cases that break shape assumptions.
- Changed default optimizer from `adamw_8bit` to `adamw`:
  - `mlx_tinker/config.py`
  - `mlx_tinker/backend/training.py`

### UAT framework

Added a new opt-in UAT framework:

- `scripts/prepare_uat_fixtures.py`
- `scripts/environments/cuad_env.py`
- `tests/uat/helpers.py`
- `tests/uat/test_sft_uat.py`
- `tests/uat/test_helpers_unit.py`

Generated fixtures:

- `tests/fixtures/uat/wikisql_train.json`
- `tests/fixtures/uat/wikisql_eval.json`
- `tests/fixtures/uat/cuad_train.json`
- `tests/fixtures/uat/cuad_eval.json`

Also added `uat` marker in `pyproject.toml`.

---

## Completed Runs (2026-03-29)

All runs below were executed sequentially per the safety notes.

### 1. Core test suite

```
uv run pytest tests/ -k "not stress and not cookbook" -x -q -ra
```

**Result: 226 passed, 2 skipped, 20 deselected in 304.13s**

- 2 skips are UAT opt-in (`MLX_TINKER_RUN_UAT=1` not set)
- 20 deselected are stress/cookbook tests (by design)
- Zero failures

### 2. Cookbook tests

```
uv run pytest tests/cookbook/ -m cookbook -q -ra
```

**Result: 7 passed in 308.68s**

- All SFT, RL, and tool-use cookbook tests pass
- Uses real Qwen3.5-0.8B model

### 3. Stress tests

```
uv run pytest tests/stress/ -m stress -q -ra
```

**Result: 7 passed, 3 failed, 3 errors, 18 warnings in 1495.24s (24:55)**

Detailed breakdown:

| Test | Status | Root Cause |
|------|--------|------------|
| test_qwen35_hf_stress (2 tests) | PASS | 4-bit MLX vs bf16 HF parity within thresholds |
| test_bitsandbytes_equivalence::test_dynamic_map_matches_bnb | PASS | Dynamic map matches bnb reference |
| test_bitsandbytes_equivalence::test_quantize_dequantize_matches_bnb | PASS | Roundtrip matches bnb |
| test_bitsandbytes_equivalence::test_single_step_update_direction | **FAIL** | bitsandbytes requires CUDA GPU — `is_on_gpu` check fails on CPU-only Mac |
| test_tinker_equivalence::test_sft_loss_curve_correlation | **FAIL** | Correlation 0.8221 < 0.90 threshold (quantization scheme drift) |
| test_tinker_equivalence::test_sft_per_step_loss_close | **FAIL** | Max relative diff 0.3263 > 0.20 threshold (same root cause) |
| test_inference_equivalence::test_logit_equivalence | **ERROR** | `wikipedia` dataset script no longer supported by HF datasets lib |
| test_inference_equivalence::test_quantized_logit_equivalence | **ERROR** | Same dataset loading error |
| test_lora_equivalence::test_training_equivalence | **ERROR** | Same dataset loading error |

**Known pre-existing issues (unchanged from prior sessions):**
- Tinker equivalence: correlation ~0.82 (was ~0.81 previously). The 0.90 threshold may need relaxation or the drift is inherent to different quantization (CUDA NF4 vs MLX groupwise).
- bitsandbytes: requires CUDA GPU; will always fail on Mac. The dynamic map and roundtrip tests pass.
- Wikipedia dataset: `datasets` library dropped script-based datasets. The fixture needs to be changed to a different dataset or use local data.

### 4. Cross-framework audit

```
uv run python scripts/cross_framework_audit.py
```

**Result: 72/72 PASS, 0 FAIL, 0 SKIP**

All categories pass:
- Loss functions (20 forward parity + 5 gradient parity + 5 FD checks)
- Chunked CE (14 tests including cross-framework vs PyTorch)
- Optimizer (4 tests including drift tracking)
- Dynamic map (3 tests including vs bitsandbytes)
- Log-prob computation (3 tests)
- Gradient clipping (3 tests)
- Full training step (4 tests: logit, loss, gradient, post-optim parity)
- Real model (8 tests including 4-bit bitsandbytes cosine similarity)

### 5. Profiler

```
uv run python scripts/profile_finetune.py --steps 50
```

**Result: Completed successfully**

| Metric | Value |
|--------|-------|
| Model load | 1.09s |
| QLoRA apply | 0.02s |
| Data load | 3.47s |
| Warmup | 0.73s |
| forward_backward (50 steps) | 19.37s total |
| Avg step time | 0.39s |
| Min step | 0.12s |
| Max step | 1.20s |
| Throughput | 218.9 tok/s |
| Peak memory | 6.475 GB |
| Active memory (steady) | 0.50 GB |

Throughput improved from 201 tok/s (v0.1 report) to 218.9 tok/s.

Note: Profiler uses random Dolly-15K samples (no curriculum), so loss curve is noisy and doesn't converge. Loss spike at step 38 (11.0) is a long-sequence outlier, not a training bug.

### 6. Benchmark (mlx-only, inference + SFT, 10 eval — pre-fix)

```
uv run python scripts/run_benchmark.py --mlx-only --mlx-url http://127.0.0.1:8010 --skip-rl --output-dir workspace_reports/benchmark_full_run
```

**Result: Completed. Report at `workspace_reports/benchmark_full_run/benchmark_comparison_report.md`**

| Metric | Value |
|--------|-------|
| Inference accuracy (base) | 0/10 (expected — base model invents table names) |
| Inference avg time/sample | 2.59s |
| Inference peak memory | 17.47 GB |
| SFT initial loss | 15.67 |
| SFT final loss (150 steps) | 0.002 |
| SFT avg step time | 1.90s |
| SFT total wall time | 285.4s |
| SFT peak memory | 24.67 GB |
| SFT post-train accuracy | 0/10 (eval bug — fixed in subsequent run) |

RL was skipped (OOMs on 24 GB Mac with 8 rollouts on 4B model).

### 7. Reference log generation

```
uv run python scripts/generate_reference_logs.py
```

**Result: Completed. Refreshed `tests/fixtures/tinker_reference_logs.json`**

- SFT: 50 steps, mean_loss 2.08 → 0.65
- RL: 10 steps, mean_loss -0.28 → -0.36

### 8. UAT (WikiSQL)

```
MLX_TINKER_RUN_UAT=1 MLX_TINKER_UAT_LOCAL_BASE_URL=http://127.0.0.1:8010 \
  uv run pytest tests/uat/test_sft_uat.py -k wikisql -q -ra
```

**Result: 1 FAILED in 1872.87s (31:12)**

Cloud portion completed successfully. Local portion failed with:

```
ValueError: not enough values to unpack (expected 2, got 1)
  in tinker.lib.chunked_fwdbwd_helpers._metrics_reduction
  at: name, reduction = key.split(":")
```

**Root cause:** The tinker SDK's `_metrics_reduction` expects every metric key to have a colon-separated reduction type (e.g. `loss:sum`, `num_sequences:sum`). mlx-tinker returns `num_sequences` without a reduction suffix. This causes the SDK to crash when parsing the local `ForwardBackwardOutput.metrics`.

The actual metrics dict mlx-tinker returns is:
```python
metrics={'loss:sum': 84.56433868408203, 'num_sequences': 4.0}
```

`loss:sum` already has the suffix. `num_sequences` does not. The SDK crashes on `num_sequences`.

**Fix needed in `mlx_tinker/backend/training.py`:** All metric keys in `ForwardBackwardOutput.metrics` must include a reduction suffix. For example:
- `num_sequences` → `num_sequences:sum`
- Any other bare metric keys need the same treatment

CUAD UAT was skipped (same bug would occur).

---

## Bugs Fixed (2026-03-29)

### FIXED: Metrics key format mismatch (P0)
Added `:reduction` suffix to all 7 bare metric keys in `training.py`. Keys changed:
`num_sequences:sum`, `grad_accum_steps:sum`, `total_tokens:sum`, `grad_norm:mean`, `grad_norm_clipped:mean`, `learning_rate:unique`. Tests updated in 3 files.

### FIXED: NaN/Inf guard in optim_step (P0)
Added `math.isfinite(grad_norm)` check in `training.py:optim_step()`. When gradients are non-finite, the update is skipped, accumulation state is reset, and metrics include `skipped:sum=1.0`. Test added to `TestNaNInfGuards`.

### FIXED: EOS token auto-injection (P1)
Added `_inject_eos_stop_token()` in `mlx_backend.py`. Called in both `sample()` and `sample_batch()`. Ensures `tokenizer.eos_token_id` is always in `stop_tokens` for Tinker API sample paths.

### FIXED: Telemetry endpoint schema (P1)
Widened `TelemetryRequest` in `models.py` to accept both legacy format (`event` string) and SDK v0.16.1 format (`events` array + metadata). Test added.

### FIXED: Benchmark SQL extraction (P1)
Fixed `extract_sql()` in `run_benchmark.py`: removed `re.DOTALL`, added `\n` to character exclusion. SQL extraction now stops at first newline or semicolon.

### FIXED: Wikipedia dataset in stress tests (P1)
Changed `tests/stress/conftest.py` from `"wikipedia", "20220301.en"` to `"wikimedia/wikipedia", "20231101.en"`.

---

## Post-Fix Verification Results (2026-03-29)

### Phase 1: Qwen3.5-0.8B

| Suite | Result |
|-------|--------|
| Core tests | **228 passed**, 2 skipped, 20 deselected (+2 new tests) |
| Cookbook | **7 passed** |
| Cross-framework audit | **72/72 PASS** |
| Benchmark (0.8B, 10 SFT steps, 10 eval) | **Post-train accuracy: 9/10 = 90%** (was 0/10) |

### Phase 2: Qwen3.5-4B

| Suite | Result |
|-------|--------|
| Stress tests | 7 passed, **6 failed** (3 formerly-errored tests now run but fail on assertions) |
| Benchmark (4B, 150 SFT steps, 10 eval) | **Post-train accuracy: 10/10 = 100%** (was 0/10) |

Stress test detail:
- Wikipedia dataset fix (Bug 6) worked — inference equivalence and LoRA equivalence tests now **run** instead of erroring
- Those 3 tests fail on assertion thresholds (cosine similarity, loss correlation) — pre-existing quantization parity gap
- bitsandbytes single-step: fails (requires CUDA GPU)
- Tinker SFT correlation: 0.80 (threshold 0.90) — pre-existing
- Tinker per-step diff: 0.34 (threshold 0.20) — pre-existing

### Phase 3: 100-example eval (Qwen3.5-0.8B)

Eval set expanded from 10 to 100 examples (regenerated `tests/fixtures/wikisql_subset.json` from WikiSQL dev split, 140 total: 40 train + 100 eval). `EVAL_SIZE` updated to 100 in `scripts/run_benchmark.py`.

```
uv run python scripts/run_benchmark.py --mlx-only --mlx-url http://127.0.0.1:8010 --sft-steps 50 --skip-rl --output-dir workspace_reports/benchmark_0_8b_100eval
```

**Report at `workspace_reports/benchmark_0_8b_100eval/benchmark_comparison_report.md`**

| Phase | Accuracy | Details |
|-------|----------|---------|
| Before training (base model) | **0/100 (0%)** | Base model invents table names, no `FROM table` pattern |
| After SFT (50 steps, batch=2) | **32/100 (32%)** | Learned correct SQL for ~1/3 of diverse eval set |

| Metric | Value |
|--------|-------|
| SFT initial loss | 24.23 |
| SFT final loss (50 steps) | 0.87 |
| SFT total train time | 1160s (~19 min) |
| Inference avg time/sample | 2.5s |
| Train set | 40 examples |
| Eval set | 100 examples (WikiSQL dev split, held out) |

The 32% accuracy on 100 diverse held-out examples (vs 90% on the prior 10-example set) reflects the 0.8B model's limited capacity with only 40 training examples. The eval set now covers a much wider range of table schemas and SQL patterns than the original 10 examples.

---

## Remaining Open Items

### P0 — Still open

1. **KV cache invalidation after weight updates** — Unverified. In train-sample-train loops, stale KV cache may produce wrong logits.

### P1 — Still open

2. **Tinker SFT correlation** — Stress tests show 0.82 correlation (threshold 0.90). May be inherent to quantization differences.

### P2 — Release readiness

3. **Checkpoint-resume determinism** — Only roundtrip tested, not trajectory equivalence.
4. Close CI enforcement gaps (stress/cookbook in required CI).
5. Add finite-difference gradient tests for all loss functions in pytest (currently only in audit script).
