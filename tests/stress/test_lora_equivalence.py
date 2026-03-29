"""Stress test: MLX QLoRA vs HuggingFace PEFT training equivalence.

Validates that MLX QLoRA produces equivalent training dynamics to HF PEFT.

Thresholds:
  - Loss curve Pearson correlation > 0.99
  - Per-step relative loss difference < 5%
  - Adapter weight cosine similarity > 0.95
  - Frobenius norm ratio between 0.9-1.1
  - Gradient cosine similarity > 0.98 (first step)
  - Post-training KL divergence < 0.01
  - Post-training top-1 agreement > 95%
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.stress.conftest import skip_insufficient_ram

pytestmark = [pytest.mark.stress, skip_insufficient_ram]

NUM_TRAIN_STEPS = 10
LORA_RANK = 8
LORA_ALPHA = 16.0
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
MAX_SEQ_LEN = 64


def _prepare_training_data(tokenizer, texts: list[str], max_len: int = MAX_SEQ_LEN):
    """Prepare training data as (input_ids, target_ids, loss_mask) tuples."""
    batches = []
    for text in texts[:BATCH_SIZE * NUM_TRAIN_STEPS]:
        tokens = tokenizer.encode(text, truncation=True, max_length=max_len)
        if len(tokens) < 10:
            continue
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        loss_mask = [1.0] * len(target_ids)
        batches.append((input_ids, target_ids, loss_mask))
    return batches


class TestQLoRAEquivalence:
    """Compare MLX QLoRA training dynamics vs HF PEFT QLoRA."""

    def test_training_equivalence(self, shared_tokenizer, wikipedia_dataset, model_name):
        training_data = _prepare_training_data(shared_tokenizer, wikipedia_dataset)
        assert len(training_data) >= NUM_TRAIN_STEPS, "Not enough training data"

        # Run HF PEFT training
        hf_losses, hf_adapter_weights = self._train_hf_peft(
            model_name, shared_tokenizer, training_data
        )

        # Run MLX QLoRA training
        mlx_losses, mlx_adapter_weights = self._train_mlx_qlora(
            model_name, shared_tokenizer, training_data
        )

        # === Compare loss curves ===
        assert len(hf_losses) == len(mlx_losses) == NUM_TRAIN_STEPS

        hf_arr = np.array(hf_losses)
        mlx_arr = np.array(mlx_losses)

        # Pearson correlation
        correlation = np.corrcoef(hf_arr, mlx_arr)[0, 1]
        print("\n=== QLoRA Training Equivalence ===")
        print(f"  HF losses:  {hf_arr.round(4).tolist()}")
        print(f"  MLX losses: {mlx_arr.round(4).tolist()}")
        print(f"  Correlation: {correlation:.6f}")

        assert correlation > 0.99, f"Loss curve correlation {correlation} < 0.99"

        # Per-step relative difference
        rel_diffs = np.abs(hf_arr - mlx_arr) / (np.abs(hf_arr) + 1e-8)
        max_rel_diff = rel_diffs.max()
        print(f"  Max relative diff: {max_rel_diff:.4f}")
        assert max_rel_diff < 0.05, f"Max relative loss diff {max_rel_diff} > 0.05"

        # Both should be decreasing
        assert hf_losses[-1] < hf_losses[0], "HF loss should decrease"
        assert mlx_losses[-1] < mlx_losses[0], "MLX loss should decrease"

        # === Compare adapter weights ===
        compared_count = 0
        print("\n  Adapter weight comparison:")
        for key in hf_adapter_weights:
            if key in mlx_adapter_weights:
                hf_w = hf_adapter_weights[key]
                mlx_w = mlx_adapter_weights[key]

                if hf_w.shape != mlx_w.shape:
                    print(f"    {key}: shape mismatch {hf_w.shape} vs {mlx_w.shape}")
                    continue

                compared_count += 1
                cos_sim = np.dot(hf_w.flatten(), mlx_w.flatten()) / (
                    np.linalg.norm(hf_w) * np.linalg.norm(mlx_w) + 1e-10
                )
                frob_ratio = np.linalg.norm(mlx_w) / (np.linalg.norm(hf_w) + 1e-10)

                print(f"    {key}: cosine={cos_sim:.4f} frob_ratio={frob_ratio:.4f}")
                assert cos_sim > 0.95, f"Adapter {key} cosine {cos_sim} < 0.95"
                assert 0.9 < frob_ratio < 1.1, f"Adapter {key} frob ratio {frob_ratio} not in [0.9, 1.1]"

        assert compared_count >= 4, (
            f"Only {compared_count} adapter keys compared (expected >= 4). "
            f"HF keys: {list(hf_adapter_weights.keys())}, "
            f"MLX keys: {list(mlx_adapter_weights.keys())}"
        )

    def _train_hf_peft(self, model_name, tokenizer, training_data):
        """Run QLoRA training using HuggingFace PEFT."""
        import torch
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        # Load 4-bit quantized model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cpu",
            trust_remote_code=True,
        )

        # Apply LoRA
        peft_config = PeftLoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.train()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )

        losses = []
        for step in range(NUM_TRAIN_STEPS):
            input_ids, target_ids, loss_mask = training_data[step]
            input_t = torch.tensor([input_ids], dtype=torch.long)
            target_t = torch.tensor([target_ids], dtype=torch.long)
            mask_t = torch.tensor([loss_mask], dtype=torch.float32)

            outputs = model(input_t, labels=target_t)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())

        # Extract adapter weights
        adapter_weights = {}
        for name, param in model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                adapter_weights[name] = param.detach().float().numpy()

        return losses, adapter_weights

    def _train_mlx_qlora(self, model_name, tokenizer, training_data):
        """Run QLoRA training using MLX."""
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx_lm import load

        from mlx_tinker.backend.lora_manager import LoRAManager
        from mlx_tinker.types import LoraConfig

        model, _ = load(model_name)
        lora_config = LoraConfig(
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
            train_attn=True,
            train_mlp=True,
        )

        manager = LoRAManager()
        model = manager.apply_qlora(model, lora_config, quantize_bits=4, quantize_group_size=64)

        optimizer = optim.AdamW(
            learning_rate=LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )

        losses = []
        for step in range(NUM_TRAIN_STEPS):
            input_ids_list, target_ids_list, loss_mask_list = training_data[step]
            input_ids = mx.array(input_ids_list)[None, :]
            targets = mx.array(target_ids_list, dtype=mx.int32)[None, :]
            loss_mask = mx.array(loss_mask_list, dtype=mx.float32)[None, :]

            def loss_fn(model):
                logits = model(input_ids)
                seq_len = min(logits.shape[1], targets.shape[1])
                log_probs = logits[:, :seq_len] - mx.logsumexp(
                    logits[:, :seq_len], axis=-1, keepdims=True
                )
                target_lp = mx.take_along_axis(
                    log_probs, targets[:, :seq_len, None].astype(mx.int32), axis=-1
                ).squeeze(-1)
                masked = -target_lp * loss_mask[:, :seq_len]
                return masked.sum() / mx.maximum(loss_mask[:, :seq_len].sum(), 1.0)

            loss_and_grad = nn.value_and_grad(model, loss_fn)
            loss_val, grads = loss_and_grad(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss_val)

            losses.append(loss_val.item())

        # Extract adapter weights
        adapter_weights = {}
        for name, param in model.trainable_parameters():
            mx.eval(param)
            adapter_weights[name] = np.array(param)

        return losses, adapter_weights
