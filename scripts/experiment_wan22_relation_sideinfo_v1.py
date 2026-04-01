#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import io
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/root/Wan2.2")

from lighttaew2_2_streaming_decode import (  # noqa: E402
    Wan22LightTAEStreamingDecoder,
    load_saved_latents,
    torch_dtype,
)
from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402


@dataclass(frozen=True)
class RelationScheme:
    name: str
    kind: str
    lowres_factor: int
    quantization: str
    tile_rows: int = 0
    tile_cols: int = 0


DEFAULT_LATENT_PATHS = [
    Path("/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/000_subject_consistency_r00_p003_a_person_eating_a_burger.pt"),
    Path("/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/015_motion_smoothness_r03_p064_a_bear_climbing_a_tree.pt"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Server/client proof-of-concept: store latents plus compact TAE↔VAE relation side information."
    )
    parser.add_argument("--latent-path", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("/workspace/models/Wan2.2-TI2V-5B"),
    )
    parser.add_argument(
        "--lighttae-path",
        type=Path,
        default=Path("/root/models/vae/lighttaew2_2.safetensors"),
    )
    parser.add_argument(
        "--lightx2v-root",
        type=Path,
        default=Path("/root/LightX2V"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/LatentsCompress/examples/profiling/wan22_relation_sideinfo_v1"),
    )
    return parser.parse_args()


def default_schemes() -> list[RelationScheme]:
    return [
        RelationScheme(name="tae_only", kind="none", lowres_factor=0, quantization="none"),
        RelationScheme(name="tile_affine_f8_4x4_fp16", kind="tile_affine", lowres_factor=8, quantization="fp16", tile_rows=4, tile_cols=4),
        RelationScheme(name="lowres_residual_f8_q8", kind="lowres_residual", lowres_factor=8, quantization="q8"),
        RelationScheme(name="lowres_residual_f4_q8", kind="lowres_residual", lowres_factor=4, quantization="q8"),
    ]


def maybe_cleanup(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def save_mp4(video_cthw: torch.Tensor, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video_cthw.detach().cpu()[None], save_file=str(path), fps=fps, nrow=1)


def ffmpeg_metric(ref_mp4: Path, dist_mp4: Path, metric: str) -> dict[str, float | None]:
    if metric == "psnr":
        cmd = ["ffmpeg", "-i", str(ref_mp4), "-i", str(dist_mp4), "-lavfi", "psnr", "-f", "null", "-"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        match = re.search(r"average:([0-9.]+)", proc.stderr)
        return {"mp4_psnr_db": float(match.group(1)) if match else None}
    if metric == "ssim":
        cmd = ["ffmpeg", "-i", str(ref_mp4), "-i", str(dist_mp4), "-lavfi", "ssim", "-f", "null", "-"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        match = re.search(r"All:([0-9.]+)", proc.stderr)
        return {"mp4_ssim": float(match.group(1)) if match else None}
    raise ValueError(metric)


def to_uint8_frames(video_cthw: torch.Tensor) -> torch.Tensor:
    frames = ((video_cthw.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return frames.permute(1, 2, 3, 0).contiguous()


def mse_and_psnr(ref: torch.Tensor, dist: torch.Tensor, max_val: float = 255.0) -> tuple[float, float]:
    ref_f = ref.to(torch.float32)
    dist_f = dist.to(torch.float32)
    mse = torch.mean((ref_f - dist_f) ** 2).item()
    if mse == 0.0:
        return 0.0, math.inf
    psnr = 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)
    return mse, psnr


def area_downsample_video(video_cthw: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return video_cthw
    tchw = video_cthw.permute(1, 0, 2, 3).contiguous()
    out_h = max(1, video_cthw.shape[-2] // factor)
    out_w = max(1, video_cthw.shape[-1] // factor)
    low = F.interpolate(tchw, size=(out_h, out_w), mode="area")
    return low.permute(1, 0, 2, 3).contiguous()


def bilinear_resize_video(video_cthw: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    tchw = video_cthw.permute(1, 0, 2, 3).contiguous()
    up = F.interpolate(tchw, size=out_hw, mode="bilinear", align_corners=False)
    return up.permute(1, 0, 2, 3).contiguous()


def tile_bounds(length: int, parts: int) -> list[tuple[int, int]]:
    edges = [round(i * length / parts) for i in range(parts + 1)]
    return [(edges[i], edges[i + 1]) for i in range(parts)]


def build_tile_affine_metadata(
    tae_low_cthw: torch.Tensor,
    vae_low_cthw: torch.Tensor,
    rows: int,
    cols: int,
    eps: float = 1e-4,
) -> dict[str, torch.Tensor]:
    c, t, h, w = tae_low_cthw.shape
    gain = torch.ones((t, c, rows, cols), dtype=torch.float32)
    bias = torch.zeros((t, c, rows, cols), dtype=torch.float32)
    row_bounds = tile_bounds(h, rows)
    col_bounds = tile_bounds(w, cols)
    for frame_idx in range(t):
        src = tae_low_cthw[:, frame_idx]
        dst = vae_low_cthw[:, frame_idx]
        for r, (hs, he) in enumerate(row_bounds):
            for q, (ws, we) in enumerate(col_bounds):
                src_tile = src[:, hs:he, ws:we].reshape(c, -1)
                dst_tile = dst[:, hs:he, ws:we].reshape(c, -1)
                src_mean = src_tile.mean(dim=1)
                dst_mean = dst_tile.mean(dim=1)
                src_std = src_tile.std(dim=1, unbiased=False)
                dst_std = dst_tile.std(dim=1, unbiased=False)
                g = (dst_std + eps) / (src_std + eps)
                b = dst_mean - g * src_mean
                gain[frame_idx, :, r, q] = g
                bias[frame_idx, :, r, q] = b
    return {"gain": gain, "bias": bias}


def apply_tile_affine_metadata(
    tae_video_cthw: torch.Tensor,
    metadata: dict[str, torch.Tensor],
    lowres_factor: int,
    rows: int,
    cols: int,
) -> torch.Tensor:
    low = area_downsample_video(tae_video_cthw, lowres_factor)
    high = tae_video_cthw - bilinear_resize_video(low, tuple(tae_video_cthw.shape[-2:]))
    corrected_low = low.clone()
    row_bounds = tile_bounds(corrected_low.shape[-2], rows)
    col_bounds = tile_bounds(corrected_low.shape[-1], cols)
    gain = metadata["gain"]
    bias = metadata["bias"]
    for frame_idx in range(corrected_low.shape[1]):
        for r, (hs, he) in enumerate(row_bounds):
            for q, (ws, we) in enumerate(col_bounds):
                g = gain[frame_idx, :, r, q].view(-1, 1, 1)
                b = bias[frame_idx, :, r, q].view(-1, 1, 1)
                corrected_low[:, frame_idx, hs:he, ws:we] = corrected_low[:, frame_idx, hs:he, ws:we] * g + b
    return (bilinear_resize_video(corrected_low, tuple(tae_video_cthw.shape[-2:])) + high).clamp(-1.0, 1.0)


def _zstd_compress(raw: bytes, level: int = 19) -> bytes:
    proc = subprocess.run(
        ["zstd", "-q", f"-{level}", "-c"],
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def serialize_tensor_fp16(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, data=tensor.cpu().numpy().astype(np.float16, copy=False))
    return _zstd_compress(buf.getvalue())


def quantize_q8(tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    arr = tensor.cpu().numpy().astype(np.float32, copy=False)
    # C,T,H,W -> T,C,H,W for per-frame/per-channel scales
    arr_tchw = np.transpose(arr, (1, 0, 2, 3))
    scales = np.max(np.abs(arr_tchw), axis=(2, 3), keepdims=True) / 127.0
    scales = np.maximum(scales, 1e-8).astype(np.float32)
    quant = np.clip(np.rint(arr_tchw / scales), -127, 127).astype(np.int8)
    return quant, scales.astype(np.float16)


def dequantize_q8(quant: np.ndarray, scales: np.ndarray) -> torch.Tensor:
    arr_tchw = quant.astype(np.float32) * scales.astype(np.float32)
    arr_cthw = np.transpose(arr_tchw, (1, 0, 2, 3))
    return torch.from_numpy(arr_cthw.copy())


def serialize_tensor_q8(tensor: torch.Tensor) -> bytes:
    quant, scales = quantize_q8(tensor)
    buf = io.BytesIO()
    np.savez(buf, quant=quant, scales=scales)
    return _zstd_compress(buf.getvalue())


def decode_vae_full(vae: Wan2_2_VAE, latents_cthw: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
    synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        video = vae.decode([latents_cthw.to(device=device, dtype=torch.float32)])[0]
    synchronize(device)
    dt = time.perf_counter() - t0
    out = video.detach().cpu()
    del video
    maybe_cleanup(device)
    return out, dt


def decode_tae_full(decoder: Wan22LightTAEStreamingDecoder, latents_cthw: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
    decoder.reset()
    groups: list[torch.Tensor] = []
    synchronize(device)
    t0 = time.perf_counter()
    for step_idx in range(int(latents_cthw.shape[1])):
        chunk = latents_cthw[:, step_idx:step_idx + 1].contiguous()
        frames = decoder.push_latent_chunk(chunk, return_cpu=True)
        group = frames.permute(1, 0, 2, 3).contiguous().mul(2.0).sub(1.0)
        groups.append(group)
    synchronize(device)
    dt = time.perf_counter() - t0
    return torch.cat(groups, dim=1), dt


def build_sideinfo(
    scheme: RelationScheme,
    tae_video: torch.Tensor,
    vae_video: torch.Tensor,
) -> tuple[dict[str, Any], bytes]:
    if scheme.kind == "none":
        return {"kind": "none"}, b""

    tae_low = area_downsample_video(tae_video, scheme.lowres_factor)
    vae_low = area_downsample_video(vae_video, scheme.lowres_factor)

    if scheme.kind == "tile_affine":
        metadata = build_tile_affine_metadata(tae_low, vae_low, scheme.tile_rows, scheme.tile_cols)
        payload = {
            "kind": scheme.kind,
            "lowres_factor": scheme.lowres_factor,
            "tile_rows": scheme.tile_rows,
            "tile_cols": scheme.tile_cols,
            "gain": metadata["gain"],
            "bias": metadata["bias"],
        }
        encoded = serialize_tensor_fp16(torch.cat([metadata["gain"], metadata["bias"]], dim=1))
        return payload, encoded

    if scheme.kind == "lowres_residual":
        residual = vae_low - tae_low
        payload = {
            "kind": scheme.kind,
            "lowres_factor": scheme.lowres_factor,
            "quantization": scheme.quantization,
            "residual": residual,
        }
        if scheme.quantization == "fp16":
            encoded = serialize_tensor_fp16(residual)
        elif scheme.quantization == "q8":
            encoded = serialize_tensor_q8(residual)
        else:
            raise ValueError(scheme.quantization)
        return payload, encoded

    raise ValueError(scheme.kind)


def reconstruct_client_video(
    scheme: RelationScheme,
    tae_video: torch.Tensor,
    payload: dict[str, Any],
) -> torch.Tensor:
    if scheme.kind == "none":
        return tae_video

    if scheme.kind == "tile_affine":
        metadata = {"gain": payload["gain"], "bias": payload["bias"]}
        return apply_tile_affine_metadata(
            tae_video_cthw=tae_video,
            metadata=metadata,
            lowres_factor=payload["lowres_factor"],
            rows=payload["tile_rows"],
            cols=payload["tile_cols"],
        )

    if scheme.kind == "lowres_residual":
        low = area_downsample_video(tae_video, payload["lowres_factor"])
        high = tae_video - bilinear_resize_video(low, tuple(tae_video.shape[-2:]))
        corrected_low = low + payload["residual"]
        return (bilinear_resize_video(corrected_low, tuple(tae_video.shape[-2:])) + high).clamp(-1.0, 1.0)

    raise ValueError(scheme.kind)


def payload_storage_bytes(payload: dict[str, Any], encoded_bytes: bytes, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded_bytes)
    return output_path.stat().st_size


def main() -> None:
    args = parse_args()
    latent_paths = [Path(p) for p in args.latent_path] if args.latent_path else list(DEFAULT_LATENT_PATHS)
    if args.limit is not None:
        latent_paths = latent_paths[:args.limit]
    schemes = default_schemes()

    args.output_root.mkdir(parents=True, exist_ok=True)

    vae = Wan2_2_VAE(
        vae_pth=str(args.ckpt_dir / "Wan2.2_VAE.pth"),
        dtype=torch_dtype(args.dtype),
        device=args.device,
    )
    tae = Wan22LightTAEStreamingDecoder(
        lightx2v_root=args.lightx2v_root,
        checkpoint_path=args.lighttae_path,
        device=args.device,
        dtype=torch_dtype(args.dtype),
        need_scaled=True,
    )

    all_samples: list[dict[str, Any]] = []
    for latent_path in latent_paths:
        latents, meta = load_saved_latents(latent_path)
        fps = int(meta.get("fps", 24))
        sample_dir = args.output_root / latent_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        vae_video, vae_sec = decode_vae_full(vae, latents, args.device)
        tae_video, tae_sec = decode_tae_full(tae, latents, args.device)

        baseline_mp4 = sample_dir / "baseline_vae.mp4"
        tae_mp4 = sample_dir / "tae_only.mp4"
        save_mp4(vae_video, baseline_mp4, fps)
        save_mp4(tae_video, tae_mp4, fps)

        rows = []
        for scheme in schemes:
            if scheme.kind == "none":
                recon = tae_video
                encoded_bytes = b""
            else:
                payload, encoded_bytes = build_sideinfo(scheme, tae_video, vae_video)
                sideinfo_path = sample_dir / f"{scheme.name}.rel"
                sideinfo_bytes = payload_storage_bytes(payload, encoded_bytes, sideinfo_path)
                recon = reconstruct_client_video(scheme, tae_video, payload)
            mp4_path = sample_dir / f"{scheme.name}.mp4"
            save_mp4(recon, mp4_path, fps)
            raw_mse, raw_psnr = mse_and_psnr(to_uint8_frames(vae_video), to_uint8_frames(recon))

            row = {
                "variant": scheme.name,
                "kind": scheme.kind,
                "lowres_factor": scheme.lowres_factor,
                "quantization": scheme.quantization,
                "raw_frame_mse": raw_mse,
                "raw_frame_psnr_db": raw_psnr,
                "mp4_path": str(mp4_path),
                "sideinfo_bytes": 0 if scheme.kind == "none" else sideinfo_bytes,
                "latent_file_bytes": latent_path.stat().st_size,
                "sideinfo_vs_latent_ratio": 0.0 if scheme.kind == "none" else sideinfo_bytes / max(latent_path.stat().st_size, 1),
                "server_vae_decode_sec": vae_sec if scheme.kind != "none" else 0.0,
                "client_tae_decode_sec": tae_sec,
            }
            row.update(ffmpeg_metric(baseline_mp4, mp4_path, "psnr"))
            row.update(ffmpeg_metric(baseline_mp4, mp4_path, "ssim"))
            row["mp4_bytes"] = mp4_path.stat().st_size
            rows.append(row)

        sample_report = {
            "latent_path": str(latent_path),
            "meta": meta,
            "baseline_mp4": str(baseline_mp4),
            "tae_only_mp4": str(tae_mp4),
            "variants": rows,
        }
        (sample_dir / "report.json").write_text(json.dumps(sample_report, indent=2), encoding="utf-8")
        all_samples.append(sample_report)
        maybe_cleanup(args.device)

    overall = {
        "latent_paths": [str(p) for p in latent_paths],
        "schemes": [scheme.__dict__ for scheme in schemes],
        "samples": all_samples,
    }
    out_path = args.output_root / "relation_sideinfo_report.json"
    out_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
