from __future__ import annotations

import json
import math
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

MAGIC = b"WLATZ1\n"
JSON_LEN_STRUCT = struct.Struct("<Q")
DEFAULT_ZSTD_LEVEL = 19


@dataclass(frozen=True)
class SchemeSpec:
    name: str
    uses_temporal_prediction: bool
    storage_dtype: str
    quantization: str


SCHEMES: dict[str, SchemeSpec] = {
    "intra_fp16_zstd": SchemeSpec(
        name="intra_fp16_zstd",
        uses_temporal_prediction=False,
        storage_dtype="float16",
        quantization="fp16",
    ),
    "inter_delta_fp16_zstd": SchemeSpec(
        name="inter_delta_fp16_zstd",
        uses_temporal_prediction=True,
        storage_dtype="float16",
        quantization="fp16",
    ),
    "intra_q8_zstd": SchemeSpec(
        name="intra_q8_zstd",
        uses_temporal_prediction=False,
        storage_dtype="int8",
        quantization="per_channel_int8",
    ),
    "inter_delta_q8_zstd": SchemeSpec(
        name="inter_delta_q8_zstd",
        uses_temporal_prediction=True,
        storage_dtype="int8",
        quantization="per_channel_int8",
    ),
}


def _zstd_compress(raw: bytes, level: int = DEFAULT_ZSTD_LEVEL) -> bytes:
    proc = subprocess.run(
        ["zstd", "-q", f"-{level}", "-c"],
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def _zstd_decompress(blob: bytes) -> bytes:
    proc = subprocess.run(
        ["zstd", "-q", "-d", "-c"],
        input=blob,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def _to_time_major(latents: torch.Tensor) -> torch.Tensor:
    return latents.permute(1, 0, 2, 3).contiguous()


def _from_time_major(latents_tchw: torch.Tensor) -> torch.Tensor:
    return latents_tchw.permute(1, 0, 2, 3).contiguous()


def _apply_temporal_prediction(latents_tchw: torch.Tensor) -> torch.Tensor:
    predicted = latents_tchw.clone()
    predicted[1:] = latents_tchw[1:] - latents_tchw[:-1]
    return predicted


def _undo_temporal_prediction(predicted_tchw: torch.Tensor) -> torch.Tensor:
    restored = predicted_tchw.clone()
    for idx in range(1, restored.shape[0]):
        restored[idx] = restored[idx - 1] + restored[idx]
    return restored


def _encode_fp16(latents_tchw: torch.Tensor) -> tuple[bytes, dict[str, Any]]:
    payload = latents_tchw.to(torch.float16).cpu().numpy()
    return payload.tobytes(order="C"), {}


def _decode_fp16(raw: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    array = np.frombuffer(raw, dtype=np.float16).reshape(shape)
    return torch.from_numpy(array.astype(np.float32, copy=True))


def _encode_q8(latents_tchw: torch.Tensor) -> tuple[bytes, dict[str, Any]]:
    working = latents_tchw.to(torch.float32).cpu()
    scales = working.abs().amax(dim=(0, 2, 3)) / 127.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = torch.round(working / scales[None, :, None, None]).clamp(-127, 127).to(torch.int8)
    meta = {
        "scales": [float(x) for x in scales.tolist()],
    }
    return quantized.numpy().tobytes(order="C"), meta


def _decode_q8(raw: bytes, shape: tuple[int, ...], meta: dict[str, Any]) -> torch.Tensor:
    array = np.frombuffer(raw, dtype=np.int8).reshape(shape)
    q = torch.from_numpy(array.astype(np.int16, copy=True)).to(torch.float32)
    scales = torch.tensor(meta["scales"], dtype=torch.float32)
    return q * scales[None, :, None, None]


def encode_latents(
    latents_cthw: torch.Tensor,
    output_path: str | Path,
    scheme_name: str,
    *,
    zstd_level: int = DEFAULT_ZSTD_LEVEL,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = SCHEMES[scheme_name]
    output_path = Path(output_path)
    latents_tchw = _to_time_major(latents_cthw.to(torch.float32).cpu())
    source_shape = tuple(int(x) for x in latents_tchw.shape)
    if spec.uses_temporal_prediction:
        coded_tchw = _apply_temporal_prediction(latents_tchw)
    else:
        coded_tchw = latents_tchw

    if spec.quantization == "fp16":
        raw_payload, quant_meta = _encode_fp16(coded_tchw)
    elif spec.quantization == "per_channel_int8":
        raw_payload, quant_meta = _encode_q8(coded_tchw)
    else:
        raise ValueError(f"Unsupported quantization: {spec.quantization}")

    compressed_payload = _zstd_compress(raw_payload, level=zstd_level)
    header = {
        "magic": MAGIC.decode("ascii", errors="ignore").strip(),
        "scheme": scheme_name,
        "original_layout": "C,T,H,W",
        "stored_layout": "T,C,H,W",
        "stored_shape": list(source_shape),
        "original_shape": list(latents_cthw.shape),
        "original_dtype": str(latents_cthw.dtype),
        "quantization": spec.quantization,
        "storage_dtype": spec.storage_dtype,
        "temporal_prediction": spec.uses_temporal_prediction,
        "zstd_level": zstd_level,
        "quant_meta": quant_meta,
    }
    if extra_meta:
        header["extra_meta"] = extra_meta

    header_bytes = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    blob = MAGIC + JSON_LEN_STRUCT.pack(len(header_bytes)) + header_bytes + compressed_payload
    output_path.write_bytes(blob)

    return {
        "scheme": scheme_name,
        "container_path": str(output_path),
        "container_bytes": output_path.stat().st_size,
        "raw_payload_bytes": len(raw_payload),
        "compressed_payload_bytes": len(compressed_payload),
        "header_bytes": len(header_bytes),
        "temporal_prediction": spec.uses_temporal_prediction,
        "quantization": spec.quantization,
    }


def decode_latents(input_path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    input_path = Path(input_path)
    blob = input_path.read_bytes()
    if not blob.startswith(MAGIC):
        raise ValueError(f"Invalid container magic for {input_path}")
    header_len = JSON_LEN_STRUCT.unpack(blob[len(MAGIC): len(MAGIC) + JSON_LEN_STRUCT.size])[0]
    header_start = len(MAGIC) + JSON_LEN_STRUCT.size
    header_end = header_start + header_len
    header = json.loads(blob[header_start:header_end].decode("utf-8"))
    payload = _zstd_decompress(blob[header_end:])

    stored_shape = tuple(int(x) for x in header["stored_shape"])
    quantization = header["quantization"]
    if quantization == "fp16":
        decoded_tchw = _decode_fp16(payload, stored_shape)
    elif quantization == "per_channel_int8":
        decoded_tchw = _decode_q8(payload, stored_shape, header["quant_meta"])
    else:
        raise ValueError(f"Unsupported quantization: {quantization}")

    if header.get("temporal_prediction"):
        decoded_tchw = _undo_temporal_prediction(decoded_tchw)

    decoded_cthw = _from_time_major(decoded_tchw)
    return decoded_cthw.to(torch.float32), header


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    if compressed_bytes <= 0:
        return math.inf
    return float(original_bytes) / float(compressed_bytes)
