#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import boto3
import imageio.v3 as iio
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze atomic lossy steps for the 64 WAN2.2 benchmark samples stored in S3. "
            "The pipeline is: decoded float video -> clamp -> uint8 quantization -> "
            "YUV420 round-trip -> H.264 MP4."
        )
    )
    parser.add_argument("--bucket", default="video-latents")
    parser.add_argument(
        "--prefix",
        default="server_video_assets_2026-04-02/generated/video_bench/wan22_ti2v5b_vbench_16x4_seed42",
    )
    parser.add_argument("--wan-root", default="/root/Wan2.2")
    parser.add_argument("--ckpt-dir", default="/root/models/Wan2.2-TI2V-5B")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument(
        "--output-dir",
        default="/root/LatentsCompress/examples/wan22_s3_stepwise_loss_analysis",
    )
    parser.add_argument(
        "--download-dir",
        default="/root/tmp/wan22_s3_stepwise_loss",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
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


def tensor_frames_from_video(video_cthw: torch.Tensor) -> torch.Tensor:
    frames = ((video_cthw.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return frames.permute(1, 2, 3, 0).contiguous()


def psnr_from_mse(mse: float, data_range: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10((data_range * data_range) / mse)


def compute_float_psnr(prev_video: torch.Tensor, next_video: torch.Tensor, data_range: float) -> float:
    mse = torch.mean((prev_video - next_video) ** 2).item()
    return psnr_from_mse(mse, data_range=data_range)


def compute_u8_psnr(prev_frames: np.ndarray, next_frames: np.ndarray) -> float:
    diff = prev_frames.astype(np.float32) - next_frames.astype(np.float32)
    mse = float(np.mean(diff * diff))
    return psnr_from_mse(mse, data_range=255.0)


def dequantize_u8_frames(frames_u8: torch.Tensor) -> torch.Tensor:
    frames = frames_u8.to(torch.float32) / 127.5 - 1.0
    return frames.permute(3, 0, 1, 2).contiguous()


def ffmpeg_yuv420_roundtrip_rgb(frames_rgb_u8: np.ndarray, width: int, height: int, fps: int) -> np.ndarray:
    expected_bytes = frames_rgb_u8.size
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-vf",
        "format=yuv420p,format=rgb24",
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        input=frames_rgb_u8.tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    raw = proc.stdout
    if len(raw) != expected_bytes:
        raise RuntimeError(
            f"Unexpected ffmpeg output length: got {len(raw)} bytes, expected {expected_bytes} bytes"
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape(frames_rgb_u8.shape)


def write_summary_files(output_dir: Path, reports: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not reports:
        return

    def mean_of(key_path: list[str]) -> float:
        values = reports
        vals = []
        for report in values:
            cur: Any = report
            for key in key_path:
                cur = cur[key]
            vals.append(float(cur))
        return float(np.mean(vals))

    summary = {
        "config": {
            "bucket": args.bucket,
            "prefix": args.prefix,
            "wan_root": args.wan_root,
            "ckpt_dir": args.ckpt_dir,
            "device": args.device,
            "dtype": args.dtype,
            "sample_count": len(reports),
        },
        "mean_sizes_bytes": {
            "fp16_frames": mean_of(["sizes_bytes", "fp16_frames"]),
            "clamped_fp16_frames": mean_of(["sizes_bytes", "clamped_fp16_frames"]),
            "rgb_u8_frames": mean_of(["sizes_bytes", "rgb_u8_frames"]),
            "yuv420_raw_frames": mean_of(["sizes_bytes", "yuv420_raw_frames"]),
            "final_mp4": mean_of(["sizes_bytes", "final_mp4"]),
        },
        "mean_storage_reduction_pct": {
            "from_fp16": {
                "clamp": mean_of(["storage_reduction_pct", "from_fp16", "clamp"]),
                "quant_u8": mean_of(["storage_reduction_pct", "from_fp16", "quant_u8"]),
                "yuv420": mean_of(["storage_reduction_pct", "from_fp16", "yuv420"]),
                "mp4": mean_of(["storage_reduction_pct", "from_fp16", "mp4"]),
            },
            "from_previous": {
                "clamp": mean_of(["storage_reduction_pct", "from_previous", "clamp"]),
                "quant_u8": mean_of(["storage_reduction_pct", "from_previous", "quant_u8"]),
                "yuv420": mean_of(["storage_reduction_pct", "from_previous", "yuv420"]),
                "mp4": mean_of(["storage_reduction_pct", "from_previous", "mp4"]),
            },
        },
        "mean_step_psnr_db": {
            "clamp": mean_of(["step_psnr_db", "clamp"]),
            "quant_u8": mean_of(["step_psnr_db", "quant_u8"]),
            "yuv420": mean_of(["step_psnr_db", "yuv420"]),
            "h264_from_yuv420": mean_of(["step_psnr_db", "h264_from_yuv420"]),
        },
        "mean_cumulative_psnr_db": {
            "after_clamp": mean_of(["cumulative_psnr_db", "after_clamp"]),
            "after_quant_u8": mean_of(["cumulative_psnr_db", "after_quant_u8"]),
            "after_yuv420": mean_of(["cumulative_psnr_db", "after_yuv420"]),
            "after_mp4": mean_of(["cumulative_psnr_db", "after_mp4"]),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = output_dir / "per_sample_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stem",
                "prompt",
                "frame_count",
                "width",
                "height",
                "fps",
                "fp16_frames_bytes",
                "rgb_u8_frames_bytes",
                "yuv420_raw_frames_bytes",
                "final_mp4_bytes",
                "reduce_from_fp16_clamp_pct",
                "reduce_from_fp16_quant_u8_pct",
                "reduce_from_fp16_yuv420_pct",
                "reduce_from_fp16_mp4_pct",
                "reduce_from_prev_clamp_pct",
                "reduce_from_prev_quant_u8_pct",
                "reduce_from_prev_yuv420_pct",
                "reduce_from_prev_mp4_pct",
                "step_psnr_clamp_db",
                "step_psnr_quant_u8_db",
                "step_psnr_yuv420_db",
                "step_psnr_h264_from_yuv420_db",
                "cumulative_psnr_after_clamp_db",
                "cumulative_psnr_after_quant_u8_db",
                "cumulative_psnr_after_yuv420_db",
                "cumulative_psnr_after_mp4_db",
            ],
        )
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "stem": report["stem"],
                    "prompt": report["prompt"],
                    "frame_count": report["frame_count"],
                    "width": report["width"],
                    "height": report["height"],
                    "fps": report["fps"],
                    "fp16_frames_bytes": report["sizes_bytes"]["fp16_frames"],
                    "rgb_u8_frames_bytes": report["sizes_bytes"]["rgb_u8_frames"],
                    "yuv420_raw_frames_bytes": report["sizes_bytes"]["yuv420_raw_frames"],
                    "final_mp4_bytes": report["sizes_bytes"]["final_mp4"],
                    "reduce_from_fp16_clamp_pct": report["storage_reduction_pct"]["from_fp16"]["clamp"],
                    "reduce_from_fp16_quant_u8_pct": report["storage_reduction_pct"]["from_fp16"]["quant_u8"],
                    "reduce_from_fp16_yuv420_pct": report["storage_reduction_pct"]["from_fp16"]["yuv420"],
                    "reduce_from_fp16_mp4_pct": report["storage_reduction_pct"]["from_fp16"]["mp4"],
                    "reduce_from_prev_clamp_pct": report["storage_reduction_pct"]["from_previous"]["clamp"],
                    "reduce_from_prev_quant_u8_pct": report["storage_reduction_pct"]["from_previous"]["quant_u8"],
                    "reduce_from_prev_yuv420_pct": report["storage_reduction_pct"]["from_previous"]["yuv420"],
                    "reduce_from_prev_mp4_pct": report["storage_reduction_pct"]["from_previous"]["mp4"],
                    "step_psnr_clamp_db": report["step_psnr_db"]["clamp"],
                    "step_psnr_quant_u8_db": report["step_psnr_db"]["quant_u8"],
                    "step_psnr_yuv420_db": report["step_psnr_db"]["yuv420"],
                    "step_psnr_h264_from_yuv420_db": report["step_psnr_db"]["h264_from_yuv420"],
                    "cumulative_psnr_after_clamp_db": report["cumulative_psnr_db"]["after_clamp"],
                    "cumulative_psnr_after_quant_u8_db": report["cumulative_psnr_db"]["after_quant_u8"],
                    "cumulative_psnr_after_yuv420_db": report["cumulative_psnr_db"]["after_yuv420"],
                    "cumulative_psnr_after_mp4_db": report["cumulative_psnr_db"]["after_mp4"],
                }
            )


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
    sample_json_keys = list_sample_json_keys(s3, args.bucket, args.prefix)
    if args.limit is not None:
        sample_json_keys = sample_json_keys[: args.limit]

    vae = Wan2_2_VAE(
        vae_pth=str(Path(args.ckpt_dir) / "Wan2.2_VAE.pth"),
        device=args.device,
        dtype=torch_dtype_from_name(args.dtype),
    )

    reports: list[dict[str, Any]] = []
    for index, json_key in enumerate(sample_json_keys):
        meta = load_json_from_s3(s3, args.bucket, json_key)
        stem = Path(json_key).stem
        report_path = reports_dir / f"{stem}.json"
        if args.skip_existing and report_path.exists():
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            print(f"[skip] {stem}")
            continue

        latent_key = f"{args.prefix.rstrip('/')}/latents/{stem}.pt"
        mp4_key = f"{args.prefix.rstrip('/')}/final_24fps/{stem}.mp4"
        latent_path = download_dir / f"{stem}.pt"
        mp4_path = download_dir / f"{stem}.mp4"
        download_s3_file(s3, args.bucket, latent_key, latent_path)
        download_s3_file(s3, args.bucket, mp4_key, mp4_path)

        payload = torch.load(latent_path, map_location="cpu")
        latents = payload["latents"].to(torch.float32).cpu()
        decoded_video = decode_with_vae(vae, latents, args.device).to(torch.float32).cpu()
        clamped_video = decoded_video.clamp(-1.0, 1.0)

        frames_u8_t = tensor_frames_from_video(decoded_video)
        frames_u8 = frames_u8_t.numpy()

        dequant_video = dequantize_u8_frames(frames_u8_t)
        yuv420_rgb = ffmpeg_yuv420_roundtrip_rgb(
            frames_rgb_u8=frames_u8,
            width=int(frames_u8.shape[2]),
            height=int(frames_u8.shape[1]),
            fps=int(meta.get("export_fps") or payload.get("fps") or 24),
        )
        mp4_frames = iio.imread(mp4_path)
        if mp4_frames.shape != frames_u8.shape:
            raise RuntimeError(
                f"Frame shape mismatch for {stem}: decoded {frames_u8.shape} vs mp4 {mp4_frames.shape}"
            )

        yuv420_video = dequantize_u8_frames(torch.from_numpy(yuv420_rgb))
        mp4_video = dequantize_u8_frames(torch.from_numpy(mp4_frames))

        fp16_bytes = int(decoded_video.numel() * 2)
        clamped_fp16_bytes = fp16_bytes
        rgb_u8_bytes = int(frames_u8.size)
        yuv420_bytes = int(frames_u8.shape[0] * frames_u8.shape[1] * frames_u8.shape[2] * 3 / 2)
        mp4_bytes = int(mp4_path.stat().st_size)

        report = {
            "stem": stem,
            "prompt": payload.get("prompt"),
            "frame_count": int(frames_u8.shape[0]),
            "width": int(frames_u8.shape[2]),
            "height": int(frames_u8.shape[1]),
            "fps": int(meta.get("export_fps") or payload.get("fps") or 24),
            "sizes_bytes": {
                "fp16_frames": fp16_bytes,
                "clamped_fp16_frames": clamped_fp16_bytes,
                "rgb_u8_frames": rgb_u8_bytes,
                "yuv420_raw_frames": yuv420_bytes,
                "final_mp4": mp4_bytes,
            },
            "storage_reduction_pct": {
                "from_fp16": {
                    "clamp": 100.0 * (1.0 - clamped_fp16_bytes / fp16_bytes),
                    "quant_u8": 100.0 * (1.0 - rgb_u8_bytes / fp16_bytes),
                    "yuv420": 100.0 * (1.0 - yuv420_bytes / fp16_bytes),
                    "mp4": 100.0 * (1.0 - mp4_bytes / fp16_bytes),
                },
                "from_previous": {
                    "clamp": 100.0 * (1.0 - clamped_fp16_bytes / fp16_bytes),
                    "quant_u8": 100.0 * (1.0 - rgb_u8_bytes / clamped_fp16_bytes),
                    "yuv420": 100.0 * (1.0 - yuv420_bytes / rgb_u8_bytes),
                    "mp4": 100.0 * (1.0 - mp4_bytes / yuv420_bytes),
                },
            },
            "step_psnr_db": {
                "clamp": compute_float_psnr(decoded_video, clamped_video, data_range=2.0),
                "quant_u8": compute_float_psnr(clamped_video, dequant_video, data_range=2.0),
                "yuv420": compute_u8_psnr(frames_u8, yuv420_rgb),
                "h264_from_yuv420": compute_u8_psnr(yuv420_rgb, mp4_frames),
            },
            "cumulative_psnr_db": {
                "after_clamp": compute_float_psnr(decoded_video, clamped_video, data_range=2.0),
                "after_quant_u8": compute_float_psnr(decoded_video, dequant_video, data_range=2.0),
                "after_yuv420": compute_float_psnr(decoded_video, yuv420_video, data_range=2.0),
                "after_mp4": compute_float_psnr(decoded_video, mp4_video, data_range=2.0),
            },
            "s3": {
                "json_key": json_key,
                "latent_key": latent_key,
                "mp4_key": mp4_key,
            },
        }

        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports.append(report)
        write_summary_files(output_dir, reports, args)
        print(
            f"[{index + 1}/{len(sample_json_keys)}] {stem} "
            f"clamp={report['step_psnr_db']['clamp']:.4f}dB "
            f"quant={report['step_psnr_db']['quant_u8']:.4f}dB "
            f"yuv420={report['step_psnr_db']['yuv420']:.4f}dB "
            f"h264={report['step_psnr_db']['h264_from_yuv420']:.4f}dB"
        )

        latent_path.unlink(missing_ok=True)
        mp4_path.unlink(missing_ok=True)
        del payload
        del latents
        del decoded_video
        del clamped_video
        del frames_u8_t
        del frames_u8
        del dequant_video
        del yuv420_rgb
        del yuv420_video
        del mp4_frames
        del mp4_video
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_summary_files(output_dir, reports, args)


if __name__ == "__main__":
    main()
