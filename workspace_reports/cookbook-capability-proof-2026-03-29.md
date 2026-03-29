# Cookbook Capability Proof

Date: 2026-03-29
Workspace: `/Users/surya/Documents/personal_projects/mlx-tinker`

## Goal

Verify against the official `tinker-cookbook` patterns that:

- SFT improves task capability, not just loss
- GRPO-style / grouped-rollout RL improves task reward in a measurable way

This report does **not** claim that the current WikiSQL local UAT shows task-level accuracy gains. It establishes that the training loops themselves can improve behavior on a fast local proof task.

## Cookbook References Checked

The following upstream examples were reviewed from `thinking-machines-lab/tinker-cookbook`:

- `tinker_cookbook/recipes/sl_basic.py`
- `tinker_cookbook/recipes/sl_loop.py`
- `tinker_cookbook/recipes/rl_basic.py`
- `tinker_cookbook/recipes/rl_loop.py`

The key cookbook behaviors mirrored locally were:

- SFT judged by post-training behavioral improvement, not just decreasing loss
- RL judged through grouped rollouts, per-group reward centering, and importance-sampling updates
- one rollout group produces one centered-advantage training batch
- no cloud dependency required for the proof itself

## Local Proof Design

A tiny arbitrary codeword-to-label task was used:

- model: `Qwen/Qwen3.5-0.8B`
- backend: direct local `MLXBackend`
- task: map one of four synthetic codewords to `YES` or `NO`
- reason for task choice: small enough for fast local proof, arbitrary enough that success requires learning rather than pure pretraining recall

Two proofs were added in:

- `tests/cookbook/test_capability_proofs.py`

### SFT proof

- Train on exact prompt/completion pairs
- Evaluate exact-match accuracy before and after SFT

### RL proof

- Warm-start with a few SFT steps to avoid a fully sparse reward regime
- Sample `8` rollouts per prompt
- Compute exact-match reward `1/0`
- Center rewards within the rollout group
- Train with `importance_sampling`
- Evaluate pre/post sampled reward and greedy exact-match accuracy

## Verification Run

### Capability proof file

Command:

```bash
uv run pytest tests/cookbook/test_capability_proofs.py -s -q -ra
```

Result:

- `2 passed, 2 warnings in 33.62s`

Observed proof metrics:

### SFT Capability Proof

- base accuracy: `0.000`
- tuned accuracy: `1.000`
- loss: `4.9694 -> 0.0000`

Interpretation:

- SFT clearly improved exact-match task capability on the local proof task.

### RL Capability Proof

- pre-RL greedy accuracy: `0.875`
- post-RL greedy accuracy: `0.875`
- pre-RL sampled reward: `0.562`
- post-RL sampled reward: `0.828`
- train-time mean reward: `0.617`
- non-zero advantage steps: `27/32`

Interpretation:

- RL clearly improved **policy reward under sampling** on the proof task.
- Greedy exact-match accuracy did not regress.
- This is consistent with RL improving the policy distribution even when greedy decoding is already near saturation on a tiny task.

### Existing cookbook regression

Command:

```bash
uv run pytest tests/cookbook/test_sft.py tests/cookbook/test_rl.py -q -ra
```

Result:

- `5 passed, 2 warnings in 64.80s`

## What This Proves

- The local SFT loop can improve exact-match capability on a controlled task.
- The local grouped-rollout RL loop can improve reward on a controlled task.
- The cookbook-style training logic is functioning locally on `0.8B`.

## What This Does Not Prove

- It does not prove that the current WikiSQL RL UAT improves exact-match SQL accuracy.
- It does not prove broad generalization.
- It does not replace a larger benchmark or a real downstream eval.

## Conclusion

The right conclusion is:

- **SFT works** in the sense of measurable exact-match capability gain on a local proof task.
- **RL works** in the sense of measurable reward improvement with grouped rollouts and non-zero centered advantages on a local proof task.
- The remaining open question is task transfer and benchmark lift on more realistic datasets like WikiSQL, not whether the basic SFT / GRPO-style loops function at all.
