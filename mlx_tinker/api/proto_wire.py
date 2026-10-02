"""Protobuf serialization and deserialization helpers for wire compatibility with Tinker SDK v0.31.0+."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    from tinker.proto import tinker_public_pb2 as pb
    from tinker.proto.response_conv import _tensor_data_from_proto
    HAVE_TINKER_PROTO = True
except ImportError:
    HAVE_TINKER_PROTO = False

logger = logging.getLogger(__name__)


def decode_forward_backward_proto(body: bytes) -> tuple[str, dict[str, Any]]:
    """Decode a serialized tinker.proto.ForwardBackwardRequest protobuf message."""
    if not HAVE_TINKER_PROTO:
        raise RuntimeError("tinker package with proto support is not available")

    msg = pb.ForwardBackwardRequest()
    msg.ParseFromString(body)

    data_list: list[dict[str, Any]] = []
    for datum_msg in msg.data:
        chunks: list[dict[str, Any]] = []
        for c in datum_msg.model_input:
            field = c.WhichOneof("chunk")
            if field == "encoded_text":
                tokens = np.frombuffer(c.encoded_text.tokens, dtype=np.int32).tolist()
                chunks.append({"type": "encoded_text", "tokens": tokens})
            elif field == "image":
                chunks.append({"type": "image", "data": c.image.data, "format": c.image.format})

        loss_fn_inputs: dict[str, Any] = {}
        for k, v in datum_msg.loss_fn_inputs.items():
            t_obj = _tensor_data_from_proto(v)
            loss_fn_inputs[k] = {"data": t_obj.data, "dtype": t_obj.dtype, "shape": t_obj.shape}

        data_list.append({
            "model_input": {"chunks": chunks},
            "loss_fn_inputs": loss_fn_inputs,
        })

    loss_fn = msg.loss_fn or "cross_entropy"
    loss_fn_config: dict[str, float] | None = None
    if msg.loss_fn_config:
        loss_fn_config = dict(msg.loss_fn_config)
    elif msg.loss_fn_config_v2:
        loss_fn_config = {}
        for k, v in msg.loss_fn_config_v2.items():
            if v.HasField("number"):
                loss_fn_config[k] = v.number

    request_data = {
        "data": data_list,
        "loss_fn": loss_fn,
        "loss_fn_config": loss_fn_config,
        "forward_only": bool(msg.forward_only),
    }
    return msg.model_id, request_data


def serialize_forward_backward_output_proto(result_data: dict[str, Any]) -> bytes:
    """Serialize a ForwardBackwardOutput dict into Tinker public protobuf wire bytes."""
    if not HAVE_TINKER_PROTO:
        raise RuntimeError("tinker package with proto support is not available")

    proto = pb.ForwardBackwardOutput()
    proto.loss_fn_output_type = result_data.get("loss_fn_output_type", "cross_entropy")

    metrics = result_data.get("metrics", {})
    for k, v in metrics.items():
        if v is not None:
            proto.metrics[k] = float(v)

    loss_fn_outputs = result_data.get("loss_fn_outputs", [])
    if loss_fn_outputs:
        rec = proto.loss_fn_outputs.add()
        rec.type_tag = "CrossEntropyLossOutput"
        rec.num_datums = len(loss_fn_outputs)

        all_logprobs: list[np.ndarray] = []
        for datum_out in loss_fn_outputs:
            lp_info = datum_out.get("logprobs")
            if isinstance(lp_info, dict):
                all_logprobs.append(np.array(lp_info.get("data", []), dtype=np.float32))
            elif isinstance(lp_info, list):
                all_logprobs.append(np.array(lp_info, dtype=np.float32))
            elif hasattr(lp_info, "data"):
                all_logprobs.append(np.array(lp_info.data, dtype=np.float32))
            else:
                all_logprobs.append(np.array([], dtype=np.float32))

        bt = rec.fields["logprobs"]
        bt.dtype = pb.DTYPE_FLOAT32
        bt.trailing_shape.extend([])

        data_bytes = b"".join(arr.tobytes() for arr in all_logprobs)
        bt.data = data_bytes

        offsets = [0]
        cur = 0
        for arr in all_logprobs:
            cur += len(arr.tobytes())
            offsets.append(cur)
        bt.offsets = np.array(offsets, dtype=np.int64).tobytes()

    return proto.SerializeToString()


def serialize_sample_response_proto(result_data: dict[str, Any]) -> bytes:
    """Serialize a SampleResponse dict into Tinker public protobuf wire bytes."""
    if not HAVE_TINKER_PROTO:
        raise RuntimeError("tinker package with proto support is not available")

    proto = pb.SampleResponse()
    for seq_data in result_data.get("sequences", []):
        seq = proto.sequences.add()
        stop_reason_str = seq_data.get("stop_reason", "stop")
        seq.stop_reason = pb.STOP_REASON_STOP if stop_reason_str == "stop" else pb.STOP_REASON_LENGTH
        tokens = seq_data.get("tokens", [])
        seq.tokens = np.array(tokens, dtype=np.int32).tobytes()
        logprobs = seq_data.get("logprobs")
        if logprobs is not None:
            seq.logprobs = np.array(logprobs, dtype=np.float32).tobytes()

    prompt_logprobs = result_data.get("prompt_logprobs")
    if prompt_logprobs is not None:
        proto.prompt_logprobs = np.array(prompt_logprobs, dtype=np.float32).tobytes()

    return proto.SerializeToString()
