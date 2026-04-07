#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import boto3
import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode the same Wan2.2 latent twice, then compare the two decoded raw videos "
            "and the two MP4 exports."
        )
    )
    parser.add_argument("--bucket", default="video-latents")
    parser.add_argument(
        "--prefix",
        default="server_video_assets_2026-04-02/generated/video_bench/wan22_ti2v5b_vbench_16x4_seed42",
    )
    parser.add_argument("--wan-root", default="/root/Wan2.2")
    parser.add_argument("--ckpt-dir", default="/root/models/Wan2.2-TI2V-5B")
    parser.add_argument("--output-dir", default="/root/LatentsCompress/examples/wan22_same_latent_double_decode")
    parser.add_argument("--download-dir", default="/root/tmp/wan22_same_latent_double_decode")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--limit", type=int, default=2)
    return parser.parse_args()


def ensure_sys_path(wan_root: Path) -> None:
    root = str(wan_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_wan_vae_class(wan_root: Path):
    module_path = wan_root / "wan" / "modules" / "vae2_2.py"
    spec = importlib.util.spec_from_file_location("wan_vae2_2_direct", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Wan2_2_VAE


def torch_dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def list_sample_json_keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    final_prefix = f"{prefix.rstrip('/')}/final_24fps/"
    for page in paginator.paginate(Bucket=bucket, Prefix=final_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json"):
                keys.append(key)
    return sorted(keys)


def load_json_from_s3(s3: Any, bucket: str, key: str) -> dict[str, Any]:
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def download_s3_file(s3: Any, bucket: str, key: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(path))


def decode_with_vae(vae: Any, latents_cthw: torch.Tensor, device: str) -> torch.Tensor:
    with torch.inference_mode():
        return vae.decode([latents_cthw.to(device=device, dtype=torch.float32)])[0].detach().cpu()


def tensor_frames_from_video(video_cthw: torch.Tensor) -> np.ndarray:
    frames = ((video_cthw.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return frames.permute(1, 2, 3, 0).contiguous().cpu().numpy()


def compute_psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def save_mp4(frames: np.ndarray, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        path,
        frames,
        fps=fps,
        quality=8,
        output_params=["-loglevel", "error"],
    )


def file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    wan_root = Path(args.wan_root)
    ensure_sys_path(wan_root)
    Wan2_2_VAE = load_wan_vae_class(wan_root)

    s3 = boto3.client("s3")
    sample_json_keys = list_sample_json_keys(s3, args.bucket, args.prefix)[: args.limit]

    vae = Wan2_2_VAE(
        vae_pth=str(Path(args.ckpt_dir) / "Wan2.2_VAE.pth"),
        device=args.device,
        dtype=torch_dtype_from_name(args.dtype),
    )

    summary_rows: list[dict[str, Any]] = []
    for index, json_key in enumerate(sample_json_keys, start=1):
        meta = load_json_from_s3(s3, args.bucket, json_key)
        stem = Path(json_key).stem
        latent_key = f"{args.prefix.rstrip('/')}/latents/{stem}.pt"
        latent_path = download_dir / f"{stem}.pt"
        download_s3_file(s3, args.bucket, latent_key, latent_path)

        payload = torch.load(latent_path, map_location="cpu")
        latents = payload["latents"].to(torch.float32).cpu()
        fps = int(meta.get("export_fps") or meta.get("native_fps") or payload.get("fps") or 24)

        started = time.time()
        video_a = decode_with_vae(vae, latents, args.device)
        decode_a_seconds = time.time() - started
        started = time.time()
        video_b = decode_with_vae(vae, latents, args.device)
        decode_b_seconds = time.time() - started

        frames_a = tensor_frames_from_video(video_a)
        frames_b = tensor_frames_from_video(video_b)
        raw_psnr = compute_psnr_u8(frames_a, frames_b)

        sample_dir = output_dir / stem
        sample_dir.mkdir(parents=True, exist_ok=True)
        mp4_a = sample_dir / f"{stem}_decode_a.mp4"
        mp4_b = sample_dir / f"{stem}_decode_b.mp4"
        save_mp4(frames_a, mp4_a, fps)
        save_mp4(frames_b, mp4_b, fps)

        mp4_frames_a = iio.imread(mp4_a)
        mp4_frames_b = iio.imread(mp4_b)
        mp4_psnr = compute_psnr_u8(mp4_frames_a, mp4_frames_b)

        report = {
            "stem": stem,
            "prompt": payload.get("prompt"),
            "fps": fps,
            "frame_count": int(frames_a.shape[0]),
            "resolution": [int(frames_a.shape[2]), int(frames_a.shape[1])],
            "timings_seconds": {
                "decode_a": decode_a_seconds,
                "decode_b": decode_b_seconds,
            },
            "raw_frames_compare": {
                "psnr_db": raw_psnr,
                "identical_bytes": bool(np.array_equal(frames_a, frames_b)),
            },
            "mp4_compare": {
                "psnr_db": mp4_psnr,
                "identical_file_bytes": bool(mp4_a.read_bytes() == mp4_b.read_bytes()),
                "identical_file_sha256": bool(file_sha256(mp4_a) == file_sha256(mp4_b)),
            },
            "paths": {
                "latent_path": str(latent_path),
                "mp4_a": str(mp4_a),
                "mp4_b": str(mp4_b),
            },
        }
        report_path = reports_dir / f"{stem}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        summary_rows.append(
            {
                "stem": stem,
                "raw_psnr_db": raw_psnr,
                "raw_identical": bool(np.array_equal(frames_a, frames_b)),
                "mp4_psnr_db": mp4_psnr,
                "mp4_identical_bytes": bool(mp4_a.read_bytes() == mp4_b.read_bytes()),
                "mp4_identical_sha256": bool(file_sha256(mp4_a) == file_sha256(mp4_b)),
            }
        )
        latent_path.unlink(missing_ok=True)

    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
