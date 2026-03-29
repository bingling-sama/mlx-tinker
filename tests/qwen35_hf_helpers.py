from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx_lm import load as mlx_load
from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask
from transformers.models.qwen3_5.modeling_qwen3_5 import create_causal_mask
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

QWEN35_UNIT_MODEL = "Qwen/Qwen3.5-0.8B"
QWEN35_STRESS_MODEL = "Qwen/Qwen3.5-4B"

QWEN35_PROMPTS = [
    "Table columns: id, city, score\nRows: 1, Paris, 7\nRows: 2, Tokyo, 9\nQuestion: city with highest score\nAnswer:",
    "System: summarize the bug report in one sentence.\nUser: The app freezes after uploading an image and pressing retry twice.\nAssistant:",
    "Write a Python function named fibonacci that returns the nth Fibonacci number.\nCode:",
]


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def load_hf_model(model_name: str, *, quantized: bool):
    kwargs: dict[str, Any] = {
        "device_map": "cpu",
        "trust_remote_code": True,
    }
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model


def load_mlx_model(model_name: str, *, quantized: bool):
    model, _ = mlx_load(model_name)
    if quantized:
        nn.quantize(model, bits=4, group_size=64)
    return model


def tokenize_prompt(tokenizer, prompt: str, *, max_length: int) -> list[int]:
    return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids[
        0
    ].tolist()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def top_k_agreement(a_logits: np.ndarray, b_logits: np.ndarray, *, k: int = 5) -> float:
    matches = 0
    for row_a, row_b in zip(a_logits, b_logits, strict=True):
        if np.array_equal(np.argsort(row_a)[-k:], np.argsort(row_b)[-k:]):
            matches += 1
    return matches / max(len(a_logits), 1)


def target_logprob_rmse(logits_a: np.ndarray, logits_b: np.ndarray, token_ids: list[int]) -> float:
    labels = np.array(token_ids[1:], dtype=np.int64)
    lp_a = _log_softmax_np(logits_a[:-1])
    lp_b = _log_softmax_np(logits_b[:-1])
    tgt_a = np.array([lp_a[i, labels[i]] for i in range(len(labels))], dtype=np.float64)
    tgt_b = np.array([lp_b[i, labels[i]] for i in range(len(labels))], dtype=np.float64)
    return float(np.sqrt(np.mean((tgt_a - tgt_b) ** 2)))


def sum_cross_entropy_from_logits(logits: np.ndarray, token_ids: list[int]) -> float:
    labels = np.array(token_ids[1:], dtype=np.int64)
    lp = _log_softmax_np(logits[:-1])
    return float(-sum(lp[i, labels[i]] for i in range(len(labels))))


def relative_difference(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-8)


def compute_parity_metrics(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    token_ids: list[int],
    *,
    layer_ids: list[int],
) -> dict[str, Any]:
    hidden_cosines = {
        "embed": cosine_similarity(
            reference["hidden_states"]["embed"], candidate["hidden_states"]["embed"]
        ),
        **{
            f"layer_{idx}": cosine_similarity(
                reference["hidden_states"][f"layer_{idx}"],
                candidate["hidden_states"][f"layer_{idx}"],
            )
            for idx in layer_ids
        },
        "final_norm": cosine_similarity(
            reference["hidden_states"]["final_norm"],
            candidate["hidden_states"]["final_norm"],
        ),
    }
    return {
        "hidden_cosines": hidden_cosines,
        "logits_cosine": cosine_similarity(reference["logits"], candidate["logits"]),
        "grad_cosine": cosine_similarity(
            reference["final_hidden_grad"], candidate["final_hidden_grad"]
        ),
        "logprob_rmse": target_logprob_rmse(reference["logits"], candidate["logits"], token_ids),
        "ce_rel_diff": relative_difference(
            reference["loss_sum"], sum_cross_entropy_from_logits(candidate["logits"], token_ids)
        ),
        "top5_exact": top_k_agreement(reference["logits"][:-1], candidate["logits"][:-1], k=5),
    }


def selected_layer_ids(model) -> list[int]:
    num_layers = len(model.language_model.model.layers)
    return [1, max(2, math.ceil(num_layers / 2)), num_layers]


def run_hf_qwen_parity_pass(model, token_ids: list[int], *, layer_ids: list[int]) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    text_model = model.model
    inputs_embeds = text_model.embed_tokens(input_ids)
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    text_position_ids = position_ids[0]
    position_ids = position_ids[1:]

    causal_mask = create_causal_mask(
        config=text_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    linear_attn_mask = text_model._update_linear_attn_mask(None, None)
    hidden_states = inputs_embeds
    position_embeddings = text_model.rotary_emb(hidden_states, position_ids)

    selected = set(layer_ids)
    captured_hidden_states = {
        "embed": inputs_embeds[0].float().detach().cpu().numpy(),
    }
    for idx, decoder_layer in enumerate(text_model.layers[: text_model.config.num_hidden_layers], start=1):
        layer_mask = (
            linear_attn_mask if text_model.config.layer_types[idx - 1] == "linear_attention" else causal_mask
        )
        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
        )
        if idx in selected:
            captured_hidden_states[f"layer_{idx}"] = (
                hidden_states[0].float().detach().cpu().numpy()
            )

    final_hidden = text_model.norm(hidden_states)
    final_hidden.retain_grad()
    output_head = model.get_output_embeddings()
    logits = output_head(final_hidden)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="sum",
    )
    loss.backward()
    return {
        "hidden_states": {
            **captured_hidden_states,
            "final_norm": final_hidden[0].float().detach().cpu().numpy(),
        },
        "final_hidden_grad": final_hidden.grad[0, :-1].float().detach().cpu().numpy(),
        "logits": logits[0].float().detach().cpu().numpy(),
        "loss_sum": float(loss.item()),
    }


def run_mlx_qwen_parity_pass(model, token_ids: list[int], *, layer_ids: list[int]) -> dict[str, Any]:
    input_ids = mx.array(token_ids, dtype=mx.int32)[None, :]
    hidden, hidden_states = _run_mlx_hidden_states(model, input_ids, layer_ids=layer_ids)
    logits = _project_hidden(model, hidden).astype(mx.float32)

    def loss_of_hidden(h: mx.array) -> mx.array:
        projected = _project_hidden(model, h)[:, :-1, :].astype(mx.float32)
        targets = mx.array(token_ids[1:], dtype=mx.int32)[None, :]
        log_probs = projected - mx.logsumexp(projected, axis=-1, keepdims=True)
        target_lp = mx.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(-1)
        return (-target_lp).sum()

    grad = mx.grad(loss_of_hidden)(hidden)
    loss_sum = loss_of_hidden(hidden)
    mx.eval(logits, grad, loss_sum)

    return {
        "hidden_states": hidden_states,
        "final_hidden_grad": np.array(grad.astype(mx.float32)[0, :-1]),
        "logits": np.array(logits[0]),
        "loss_sum": float(loss_sum.item()),
    }


def _run_mlx_hidden_states(model, input_ids: mx.array, *, layer_ids: list[int]):
    text_model = model.language_model.model
    hidden = text_model.embed_tokens(input_ids)
    cache = [None] * len(text_model.layers)
    fa_mask = create_attention_mask(hidden, cache[text_model.fa_idx])
    ssm_mask = create_ssm_mask(hidden, cache[text_model.ssm_idx])

    hidden_states = {
        "embed": np.array(hidden.astype(mx.float32)[0]),
    }
    selected = set(layer_ids)
    for idx, (layer, layer_cache) in enumerate(zip(text_model.layers, cache), start=1):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=layer_cache)
        if idx in selected:
            mx.eval(hidden)
            hidden_states[f"layer_{idx}"] = np.array(hidden.astype(mx.float32)[0])

    hidden = text_model.norm(hidden)
    mx.eval(hidden)
    hidden_states["final_norm"] = np.array(hidden.astype(mx.float32)[0])
    return hidden, hidden_states


def _project_hidden(model, hidden: mx.array) -> mx.array:
    language_model = model.language_model
    if language_model.args.tie_word_embeddings:
        return language_model.model.embed_tokens.as_linear(hidden)
    return language_model.lm_head(hidden)


def _log_softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
