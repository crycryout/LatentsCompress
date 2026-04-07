#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import io
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import boto3
import imageio.v2 as imageio
import lpips
import numpy as np
import torch
from pytorch_msssim import ssim as ms_ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Wan2.2 VAE round-trip loss on the 64-sample WAN benchmark "
            "stored in S3, using original generation latents as the baseline source."
        )
    )
    parser.add_argument("--bucket", default="video-latents")
    parser.add_argument(
        "--prefix",
        default="server_video_assets_2026-04-02/generated/video_bench/wan22_ti2v5b_vbench_16x4_seed42",
        help="S3 prefix containing final_24fps/, latents/, and native_16fps/.",
    )
    parser.add_argument("--wan-root", default="/root/Wan2.2")
    parser.add_argument("--ckpt-dir", default="/root/models/Wan2.2-TI2V-5B")
    parser.add_argument(
        "--output-dir",
        default="/root/LatentsCompress/examples/wan22_s3_vae_roundtrip_eval",
    )
    parser.add_argument(
        "--download-dir",
        default="/root/tmp/wan22_s3_roundtrip_eval",
        help="Temporary directory for downloaded latent payloads.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--save-recon-mp4-limit",
        type=int,
        default=0,
        help="Save reconstructed MP4s for the first N samples only.",
    )
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


def to_s3_key(prefix: str, stem: str, subdir: str, suffix: str) -> str:
    return f"{prefix.rstrip('/')}/{subdir}/{stem}{suffix}"


def load_json_from_s3(s3: Any, bucket: str, key: str) -> dict[str, Any]:
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def download_s3_file(s3: Any, bucket: str, key: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(path))


def tensor_frames_from_video(video_cthw: torch.Tensor) -> torch.Tensor:
    frames = ((video_cthw.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return frames.permute(1, 2, 3, 0).contiguous()


def raw_rgb_bytes(video_cthw: torch.Tensor) -> int:
    frames_u8 = tensor_frames_from_video(video_cthw)
    return int(frames_u8.numel() * frames_u8.element_size())


def decode_with_vae(vae: Any, latents_cthw: torch.Tensor, device: str) -> torch.Tensor:
    with torch.inference_mode():
        return vae.decode([latents_cthw.to(device=device, dtype=torch.float32)])[0].detach().cpu()


def encode_with_vae(vae: Any, video_cthw: torch.Tensor, device: str) -> torch.Tensor:
    with torch.inference_mode():
        return vae.encode([video_cthw.to(device=device, dtype=torch.float32)])[0].detach().cpu()


def latent_payload_bytes(payload: dict[str, Any]) -> int:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.tell()


def save_mp4(video_cthw: torch.Tensor, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = tensor_frames_from_video(video_cthw).cpu().numpy()
    imageio.mimwrite(
        path,
        frames,
        fps=fps,
        quality=8,
        output_params=["-loglevel", "error"],
    )


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std(ddof=0)),
    }


def compute_metrics(
    original_frames: torch.Tensor,
    recon_frames: torch.Tensor,
    batch_size: int,
    device: torch.device,
    lpips_model: Any,
) -> dict[str, Any]:
    frame_count = int(original_frames.shape[0])
    mse_values: list[float] = []
    mae_values: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []

    with torch.no_grad():
        for start in range(0, frame_count, batch_size):
            end = min(start + batch_size, frame_count)
            orig = original_frames[start:end].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
            recon = recon_frames[start:end].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)

            orig_01 = orig / 255.0
            recon_01 = recon / 255.0
            diff = orig_01 - recon_01

            mse_batch = diff.square().mean(dim=(1, 2, 3))
            mae_batch = diff.abs().mean(dim=(1, 2, 3))
            psnr_batch = 10.0 * torch.log10(1.0 / torch.clamp(mse_batch, min=1e-12))
            ssim_batch = ms_ssim(orig_01, recon_01, data_range=1.0, size_average=False)

            orig_lpips = orig_01 * 2.0 - 1.0
            recon_lpips = recon_01 * 2.0 - 1.0
            lpips_batch = lpips_model(orig_lpips, recon_lpips).flatten()

            mse_values.extend(mse_batch.detach().cpu().tolist())
            mae_values.extend(mae_batch.detach().cpu().tolist())
            psnr_values.extend(psnr_batch.detach().cpu().tolist())
            ssim_values.extend(ssim_batch.detach().cpu().tolist())
            lpips_values.extend(lpips_batch.detach().cpu().tolist())

    return {
        "frame_count": frame_count,
        "mse": summarize(mse_values),
        "mae": summarize(mae_values),
        "psnr_db": summarize(psnr_values),
        "ssim": summarize(ssim_values),
        "lpips_alex": summarize(lpips_values),
    }


def write_summary_files(
    output_dir: Path,
    reports: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if not reports:
        return

    def metric_list(name: str) -> list[float]:
        return [float(report["metrics"][name]["mean"]) for report in reports]

    sorted_by_psnr = sorted(reports, key=lambda x: x["metrics"]["psnr_db"]["mean"])
    sorted_by_lpips = sorted(reports, key=lambda x: x["metrics"]["lpips_alex"]["mean"], reverse=True)

    summary = {
        "config": {
            "bucket": args.bucket,
            "prefix": args.prefix,
            "wan_root": args.wan_root,
            "ckpt_dir": args.ckpt_dir,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "sample_count": len(reports),
        },
        "aggregate_metrics": {
            "psnr_db_mean_of_means": float(np.mean(metric_list("psnr_db"))),
            "ssim_mean_of_means": float(np.mean(metric_list("ssim"))),
            "lpips_mean_of_means": float(np.mean(metric_list("lpips_alex"))),
            "mse_mean_of_means": float(np.mean(metric_list("mse"))),
            "mae_mean_of_means": float(np.mean(metric_list("mae"))),
        },
        "aggregate_sizes": {
            "mean_original_mp4_mb": float(np.mean([r["sizes"]["original_mp4_bytes"] / 1_000_000 for r in reports])),
            "mean_original_latent_pt_mb": float(np.mean([r["sizes"]["original_latent_pt_bytes"] / 1_000_000 for r in reports])),
            "mean_reencoded_latent_pt_mb": float(np.mean([r["sizes"]["reencoded_latent_pt_bytes"] / 1_000_000 for r in reports])),
            "mean_raw_rgb_gb": float(np.mean([r["sizes"]["baseline_raw_rgb_bytes"] / 1_000_000_000 for r in reports])),
        },
        "worst_psnr_samples": [
            {
                "stem": report["stem"],
                "psnr_db": report["metrics"]["psnr_db"]["mean"],
                "ssim": report["metrics"]["ssim"]["mean"],
                "lpips_alex": report["metrics"]["lpips_alex"]["mean"],
            }
            for report in sorted_by_psnr[:5]
        ],
        "worst_lpips_samples": [
            {
                "stem": report["stem"],
                "psnr_db": report["metrics"]["psnr_db"]["mean"],
                "ssim": report["metrics"]["ssim"]["mean"],
                "lpips_alex": report["metrics"]["lpips_alex"]["mean"],
            }
            for report in sorted_by_lpips[:5]
        ],
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
                "original_mp4_bytes",
                "original_latent_pt_bytes",
                "reencoded_latent_pt_bytes",
                "baseline_raw_rgb_bytes",
                "psnr_db_mean",
                "ssim_mean",
                "lpips_alex_mean",
                "latent_mse",
                "latent_mae",
                "latent_max_abs",
                "decode_baseline_seconds",
                "encode_seconds",
                "decode_reconstructed_seconds",
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
                    "original_mp4_bytes": report["sizes"]["original_mp4_bytes"],
                    "original_latent_pt_bytes": report["sizes"]["original_latent_pt_bytes"],
                    "reencoded_latent_pt_bytes": report["sizes"]["reencoded_latent_pt_bytes"],
                    "baseline_raw_rgb_bytes": report["sizes"]["baseline_raw_rgb_bytes"],
                    "psnr_db_mean": report["metrics"]["psnr_db"]["mean"],
                    "ssim_mean": report["metrics"]["ssim"]["mean"],
                    "lpips_alex_mean": report["metrics"]["lpips_alex"]["mean"],
                    "latent_mse": report["latent_delta"]["mse"],
                    "latent_mae": report["latent_delta"]["mae"],
                    "latent_max_abs": report["latent_delta"]["max_abs"],
                    "decode_baseline_seconds": report["timings"]["decode_baseline_seconds"],
                    "encode_seconds": report["timings"]["encode_seconds"],
                    "decode_reconstructed_seconds": report["timings"]["decode_reconstructed_seconds"],
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

    device = torch.device(args.device)
    vae = Wan2_2_VAE(
        vae_pth=str(Path(args.ckpt_dir) / "Wan2.2_VAE.pth"),
        device=args.device,
        dtype=torch_dtype_from_name(args.dtype),
    )
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    reports: list[dict[str, Any]] = []
    for index, json_key in enumerate(sample_json_keys):
        meta = load_json_from_s3(s3, args.bucket, json_key)
        stem = Path(json_key).stem
        report_path = reports_dir / f"{stem}.json"
        if args.skip_existing and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            reports.append(report)
            print(f"[skip] {stem}")
            continue

        latent_key = to_s3_key(args.prefix, stem, "latents", ".pt")
        mp4_key = to_s3_key(args.prefix, stem, "final_24fps", ".mp4")

        latent_path = download_dir / f"{stem}.pt"
        download_s3_file(s3, args.bucket, latent_key, latent_path)
        original_latent_pt_bytes = latent_path.stat().st_size
        original_mp4_bytes = s3.head_object(Bucket=args.bucket, Key=mp4_key)["ContentLength"]

        payload = torch.load(latent_path, map_location="cpu")
        original_latents = payload["latents"].to(torch.float32).cpu()

        timings: dict[str, float] = {}

        started = time.time()
        baseline_video = decode_with_vae(vae, original_latents, args.device)
        timings["decode_baseline_seconds"] = time.time() - started

        started = time.time()
        reencoded_latents = encode_with_vae(vae, baseline_video, args.device).to(torch.float32)
        timings["encode_seconds"] = time.time() - started

        started = time.time()
        reconstructed_video = decode_with_vae(vae, reencoded_latents, args.device)
        timings["decode_reconstructed_seconds"] = time.time() - started

        baseline_frames = tensor_frames_from_video(baseline_video)
        reconstructed_frames = tensor_frames_from_video(reconstructed_video)
        metrics = compute_metrics(
            original_frames=baseline_frames,
            recon_frames=reconstructed_frames,
            batch_size=args.batch_size,
            device=device,
            lpips_model=lpips_model,
        )

        latent_payload = {
            "task": payload.get("task"),
            "prompt": payload.get("prompt"),
            "size": payload.get("size"),
            "frame_num": payload.get("frame_num"),
            "fps": payload.get("fps"),
            "seed": payload.get("seed"),
            "latents": reencoded_latents,
        }
        reencoded_latent_pt_bytes = latent_payload_bytes(latent_payload)

        latent_delta = original_latents - reencoded_latents
        height = int(baseline_frames.shape[1])
        width = int(baseline_frames.shape[2])
        fps = int(meta.get("export_fps") or meta.get("native_fps") or payload.get("fps") or 24)

        report: dict[str, Any] = {
            "stem": stem,
            "prompt": payload["prompt"],
            "dimension": meta["dimension"],
            "frame_count": int(baseline_frames.shape[0]),
            "width": width,
            "height": height,
            "fps": fps,
            "s3": {
                "json_key": json_key,
                "latent_key": latent_key,
                "mp4_key": mp4_key,
            },
            "sizes": {
                "original_mp4_bytes": int(original_mp4_bytes),
                "original_latent_pt_bytes": int(original_latent_pt_bytes),
                "original_latent_raw_bytes": int(original_latents.numel() * original_latents.element_size()),
                "reencoded_latent_pt_bytes": int(reencoded_latent_pt_bytes),
                "reencoded_latent_raw_bytes": int(reencoded_latents.numel() * reencoded_latents.element_size()),
                "baseline_raw_rgb_bytes": int(raw_rgb_bytes(baseline_video)),
            },
            "timings": timings,
            "latent_delta": {
                "mse": float(latent_delta.square().mean().item()),
                "mae": float(latent_delta.abs().mean().item()),
                "max_abs": float(latent_delta.abs().max().item()),
            },
            "metrics": metrics,
        }

        if index < args.save_recon_mp4_limit:
            recon_dir = output_dir / "reconstructed_mp4"
            recon_path = recon_dir / f"{stem}.mp4"
            started = time.time()
            save_mp4(reconstructed_video, recon_path, fps=fps)
            report["reconstructed_mp4"] = str(recon_path)
            report["timings"]["write_reconstructed_mp4_seconds"] = time.time() - started
            report["sizes"]["reconstructed_mp4_bytes"] = int(recon_path.stat().st_size)

        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports.append(report)
        write_summary_files(output_dir, reports, args)
        print(
            f"[{index + 1}/{len(sample_json_keys)}] {stem} "
            f"psnr={report['metrics']['psnr_db']['mean']:.4f} "
            f"ssim={report['metrics']['ssim']['mean']:.6f} "
            f"lpips={report['metrics']['lpips_alex']['mean']:.6f}"
        )

        latent_path.unlink(missing_ok=True)
        del payload
        del original_latents
        del baseline_video
        del reencoded_latents
        del reconstructed_video
        del baseline_frames
        del reconstructed_frames
        del latent_delta
        torch.cuda.empty_cache()
        gc.collect()

    write_summary_files(output_dir, reports, args)


if __name__ == "__main__":
    main()
