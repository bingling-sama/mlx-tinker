# MLX-Tinker Local UAT and Optimization Timeline

Date: 2026-03-29
Workspace: `/Users/surya/Documents/personal_projects/mlx-tinker`

## Scope

This report captures the timeline of decisions, fixes, and verification work taken to make the local-only `0.8B` acceptance flow viable, while also landing the requested first-pass training/inference optimizations.

The guiding constraints for this pass were:

- local-only testing and UAT
- no cloud Tinker in the default path
- no parallel heavy runs
- `Qwen/Qwen3.5-0.8B` as the standard verification model
- realistic continuation flow: `50` SFT steps followed by `50` RL steps
- `batch_size=4`
- GRPO group size / rollout count `= 8`

## Timeline of Decisions

### 1. Converge on a realistic local acceptance target

Initial audit work surfaced that the previous UAT structure was not aligned with the intended day-to-day local workflow. The decision was made to standardize the default verification target on:

- one local server on `127.0.0.1:8010`
- one local model: `Qwen/Qwen3.5-0.8B`
- sequential runs only
- realistic SFT followed by continued RL on SQL data

This replaced earlier ideas around cloud comparison and smaller rollout counts. The explicit user guidance was that `2` rollouts were not meaningful and RL UAT needed `8` rollouts to produce non-degenerate group statistics.

### 2. Reframe the RL UAT as continuation, not scratch RL

The next decision was to make the RL UAT mirror the common real workflow:

1. run SFT first
2. keep training state alive
3. continue with RL on the same model

That led to the UAT helper defaults being aligned around:

- `SFT_STEPS=50`
- `RL_STEPS=50`
- `BATCH_SIZE=4`
- `RL_NUM_ROLLOUTS=8`

The RL UAT was updated to assert this exact contract.

### 3. Investigate the `retrieve_future` storm instead of treating it as model slowness

The first major runtime issue observed during local UAT was a flood of `retrieve_future` requests. Investigation showed this behavior was normal for the current server and SDK interaction:

- the server returned `try_again` immediately for incomplete work
- the client retried immediately with no useful backoff

The decision here was not to rewrite the public protocol, but to make the local implementation less pathological:

- add bounded wait behavior in the server before returning `try_again`
- keep the wire format unchanged

This reduced busy-poll intensity without requiring SDK schema changes.

### 4. Treat sampler export behavior as a correctness bug, not just a performance issue

The next investigation focused on RL sampling. The intended design was that sampler refreshes should export LoRA-only state, but the code was actually saving full exposed model parameters for the sampler path.

That was treated as an implementation bug with correctness and performance implications.

The fix path was:

- make sampler exports adapter-only
- write `adapters.safetensors` plus a small config describing the base model and LoRA setup
- ensure ephemeral sampler saves create a real `sampling_session_id`
- make sampling requests resolve the `sampling_session_id` to the correct exported sampler path

This was an important turning point because it fixed both wire correctness and the “sample from wrong weights” risk.

### 5. Fix the SDK response-contract mismatch for ephemeral sampler saves

Once the realistic local UAT was rerun, the first hard failure was not in model training but in the sampler-save response shape. The SDK expected:

- `sampling_session_id` present
- `path` absent for the ephemeral sampler-save flow

The backend was not satisfying that contract. The decision here was to fix our server/backend response semantics instead of weakening the SDK assertion.

That change let the local UAT progress through SFT and into RL.

### 6. Solve the sampler-side RL crash before touching algorithm semantics

The next hard blocker appeared in RL sampling with `8` rollouts. The system hit Metal/GPU instability when using the batched multi-sample path on exported sampler checkpoints.

The decision was deliberately conservative:

- keep the public API unchanged
- keep `8` rollouts
- avoid the batched rollout path only for sampler-checkpoint-backed requests
- continue to use the regular batched path where it remained safe

This preserved the intended RL behavior while routing the fragile checkpoint-backed sampler path onto a safer sequential generation path.

### 7. Fix RL training request shape after sampling became stable

After sampler-side crashes were fixed, the next observed failure moved later in the RL loop: a single `forward_backward` request over all `8` rollouts could exceed the request timeout budget.

The decision here was to change only the helper-side batching, not the algorithm:

- still sample `8` rollouts per prompt
- convert them into sequential microbatches of `4`
- run two `forward_backward` calls
- perform one `optim_step` after both microbatches

This preserved:

- rollout group size `8`
- one policy update per full group
- the realistic local memory budget

It also aligned the RL training request shape with the user-requested `batch_size=4`.

### 8. Land the requested low-risk optimization pass in parallel with the UAT fixes

In addition to the UAT/runtime fixes, the requested first-pass backend optimizations were implemented. The decisions here followed the revised optimization plan closely:

- replace repeated per-leaf grad norm syncs with one local reduction helper
- return both unclipped and clipped grad norm values from the shared helper so `optim_step()` does not recompute them
- evaluate `loss`, token count, and captured logprobs together in `forward_backward()`
- materialize accumulated grads without deep-copying every leaf
- update optimizer hyperparameters in place instead of recreating the optimizer and losing state
- plumb existing `max_kv_cache_size` into inference paths
- honor `SamplingParams.seed`
- replace engine-side JSON serialization grouping with the existing tuple-style sampling key

The explicit decision was to avoid speculative or riskier work in this pass:

- no live LoRA fuse/unfuse
- no new KV-cache quantization config
- no speculative decoding
- no `mx.compile` attempt on the optimizer path

### 9. Treat zero-advantage RL as a real acceptance failure

After the infrastructure and runtime issues were cleared, the first full realistic local UAT still failed. This time the failure was meaningful:

- SFT completed
- RL completed all configured steps
- but every rollout in every RL step received the same reward, so `non_zero_advantage_steps == 0`

This was not treated as a flaky test problem. The decision was to keep the assertion and investigate why the supposedly realistic RL flow still had degenerate gradients.

### 10. Diagnose reward sparsity versus rollout diversity

Focused probing showed that the sampler was not collapsed to a single string. The rollout texts were diverse, but they were all malformed SQL continuations, which meant the original WikiSQL reward still collapsed to one flat value for the entire group.

Two useful observations came out of this:

- the problem was not the `8`-rollout group size
- the problem was not a dead trainer
- the problem was sparse reward for this small-model local setup

That led to the next explicit decision: keep exact execution-match as the evaluation metric, but use a denser SQL-shaped reward during RL training.

### 11. Add dense SQL reward shaping for RL while keeping exact-match evaluation

The final RL reward decision was:

- exact execution-match remains `1.0`
- missing SQL remains a strong negative signal
- invalid or non-executable SQL is graded by structural similarity to the gold SQL instead of collapsing to one constant penalty
- valid but incorrect SQL receives partial reward based on structural similarity and limited execution/result bonuses

This keeps evaluation strict while making the RL signal informative enough for local UAT on `0.8B`.

### 12. Raise timeout budgets only for the heavy local continuation path

Once the dense reward shaping was in place, the next full realistic rerun still showed that the local continuation flow was healthy but slow. The practical late-stage failure mode was no longer “bad RL,” it was that some heavy training futures could exceed the generic request timeout budget.

The decision here was intentionally narrow:

- keep ordinary request timeout behavior at `90s`
- add a separate heavy-request timeout of `180s` for training and sampler-heavy UAT operations
- raise the overall UAT wall-clock budget to `5400s` so a realistic laptop run can finish without being killed by the harness

This preserved fail-fast behavior for lightweight requests while making the realistic continuation run feasible.

### 13. Accept operational RL success even though exact-match accuracy stayed flat

The final successful UAT did not improve exact WikiSQL accuracy on the tiny local setup. The accepted interpretation was:

- exact-match WikiSQL accuracy remains a strict evaluation metric
- the `0.8B`, `50/50`, `40-train/10-eval` local profile is still too small and noisy to expect a reliable accuracy lift
- the important acceptance property for this pass is that the end-to-end continuation flow is operationally correct

The RL UAT now demonstrates:

- non-zero advantages on every RL step
- finite losses and rewards throughout
- stable sequential `8`-rollout continuation on a local laptop-safe model
- no late-step request timeout failure under the revised timeout budget

## Files Changed

The core implementation work landed in:

- `mlx_tinker/api/server.py`
- `mlx_tinker/backend/checkpointing.py`
- `mlx_tinker/backend/inference.py`
- `mlx_tinker/backend/mlx_backend.py`
- `mlx_tinker/backend/training.py`
- `mlx_tinker/engine/engine.py`
- `mlx_tinker/types.py`
- `tests/uat/helpers.py`

Targeted regression coverage was added or updated in:

- `tests/test_api.py`
- `tests/test_backend_inference.py`
- `tests/test_checkpointing.py`
- `tests/test_engine.py`
- `tests/test_optimizers.py`
- `tests/uat/test_helpers_unit.py`

## Verification Timeline

### Fast targeted suites

The following sequential validations passed after the code changes:

- `tests/test_backend_training.py`
- `tests/test_optimizers.py`
- `tests/test_backend_inference.py`
- `tests/test_engine.py`
- `tests/test_api.py`
- `tests/test_numerical_stability.py`
- `tests/test_pytorch_parity.py`
- `tests/test_e2e_sdk_wire.py`
- `tests/uat/test_helpers_unit.py`

Aggregate passing result captured during this pass:

- `166 passed, 2 warnings`

Additional final targeted verification after the last UAT-timeout helper change:

- `uv run pytest tests/test_backend_training.py tests/test_optimizers.py tests/test_backend_inference.py tests/test_engine.py tests/test_api.py tests/test_checkpointing.py tests/test_e2e_sdk_wire.py tests/uat/test_helpers_unit.py -q -ra`
  - `127 passed, 2 warnings in 10.36s`
- `uv run pytest tests/test_numerical_stability.py tests/test_pytorch_parity.py -q -ra`
  - `53 passed in 0.74s`
- `uv run pytest tests/uat/test_helpers_unit.py -q -ra`
  - `7 passed in 1.16s`

### End-to-end local UAT progression

The realistic local UAT was not made to pass in one jump. It progressed through several concrete stages:

1. initial runs failed on the sampler-save response contract
2. after sampler contract fixes, SFT passed and RL got farther
3. RL then failed on sampler-side batched-rollout instability
4. after sampler-path adjustment, RL progressed to training but hit timeout on the 8-rollout `forward_backward` request
5. RL helper batching was updated to use sequential microbatches of `4` while preserving `8` rollouts per step
6. the full realistic local UAT was rerun and exposed a real algorithmic failure: all RL rewards collapsed, so advantages stayed zero
7. focused debugging showed the rollouts were diverse but uniformly malformed SQL
8. dense WikiSQL reward shaping was added for RL while exact-match execution accuracy remained the evaluation metric
9. a shorter replay (`10` SFT / `5` RL) passed with non-zero-advantage behavior
10. the next full realistic rerun progressed much farther but still failed late on per-request timeout pressure
11. heavy-request timeout handling was split from ordinary request timeout handling
12. the full realistic local UAT was rerun sequentially and passed end-to-end

### Final realistic local UAT

The final sequential realistic local verification completed successfully.

Final pytest run:

- `MLX_TINKER_RUN_UAT=1 MLX_TINKER_UAT_LOCAL_BASE_URL=http://127.0.0.1:8010 MLX_TINKER_UAT_MODEL=Qwen/Qwen3.5-0.8B uv run pytest tests/uat/test_sft_uat.py tests/uat/test_rl_uat.py -q -ra`
- result: `2 passed, 2 warnings in 4081.38s (1:08:01)`

Relevant generated artifacts:

- SFT smoke report: `workspace_reports/uat/wikisql_sft_smoke_20260329T110422Z.json`
- combined continuation report: `workspace_reports/uat/wikisql_sft_then_rl_20260329T120157Z.json`

Key final metrics from the combined continuation run:

- base exact-match accuracy: `0.0`
- post-SFT exact-match accuracy: `0.0`
- post-RL exact-match accuracy: `0.0`
- SFT steps: `50`
- RL steps: `50`
- rollout/group size: `8`
- RL non-zero-advantage steps: `50 / 50`
- helper-measured total continuation runtime: `3452.215s`
- request timeout: `90s`
- heavy request timeout: `180s`

Interpretation:

- The realistic local continuation flow now works end-to-end under the required constraints.
- The RL loop is no longer degenerate: all steps produced non-zero group advantages.
- The remaining limitation is model quality on this very small local setup, not operational correctness of the UAT flow.

## Current Runtime Observations

Even after the correctness fixes, the dominant remaining cost in the RL path is now operational rather than obviously algorithmic:

- every RL step refreshes sampler weights
- every sampler refresh currently causes a fresh checkpoint-backed sampler load
- repeated `retrieve_future` traffic still exists, though it is less pathological than before

This does not currently block correctness, but it is the clearest next performance target after acceptance is stable.

## Summary of What Was Actually Fixed

- sampler saves now export LoRA-only weights
- ephemeral sampler saves now have correct response semantics
- sampling requests now correctly resolve `sampling_session_id`
- pending-future retrieval now waits briefly server-side before returning `try_again`
- grad norm and clipping no longer incur repeated per-leaf host syncs
- optimizer hyperparameter changes preserve optimizer state
- gradient accumulation materializes safely without unnecessary deep copies
- inference honors `max_kv_cache_size`
- inference honors `SamplingParams.seed`
- engine sample grouping now uses the structural tuple key
- realistic local RL UAT uses `8` rollouts with sequential microbatches of `4`

## Remaining Risks and Follow-ups

The main remaining risks are performance and ergonomics rather than basic correctness:

- checkpoint-backed sampler refresh is still expensive
- RL UAT wall time remains high for a laptop-only workflow
- repeated sampler reloads are likely the next major optimization target
- exact WikiSQL accuracy remains a hard problem for the small local profile even after the RL loop is operational
- the realistic run is now stable enough to profile, which was not true earlier in the day

Once the final UAT result is recorded, the most sensible next follow-up is to reduce sampler refresh overhead without making the RL path meaningfully off-policy.
