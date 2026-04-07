#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import lpips
import numpy as np
import torch
from pytorch_msssim import ssim as ms_ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode a deduplicated SkyReels full latent and compare it against the "
            "original chunked-decode raw frames."
        )
    )
    parser.add_argument("--skyreels-root", default="/root/SkyReels-V2")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--latent-chunks-pt", required=True)
    parser.add_argument("--raw-frames", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weight-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-mp4", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def ensure_sys_path(skyreels_root: Path) -> None:
    root = str(skyreels_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stitch_chunks(chunks: list[torch.Tensor], latent_overlap: int) -> torch.Tensor:
    parts = []
    for idx, chunk in enumerate(chunks):
        parts.append(chunk if idx == 0 else chunk[:, :, latent_overlap:, :, :])
    return torch.cat(parts, dim=2).contiguous()


def tensor_to_uint8_frames(video_tensor: torch.Tensor) -> np.ndarray:
    video_tensor = video_tensor.clamp(-1, 1)
    video_tensor = (video_tensor / 2 + 0.5).clamp(0, 1)
    video_tensor = video_tensor[0].permute(1, 2, 3, 0) * 255.0
    return video_tensor.round().to(torch.uint8).cpu().numpy()


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
    original_frames: np.ndarray,
    restored_frames: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float] | int]:
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    frame_count = int(min(len(original_frames), len(restored_frames)))
    mse_values: list[float] = []
    mae_values: list[float] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []

    with torch.no_grad():
        for start in range(0, frame_count, batch_size):
            end = min(start + batch_size, frame_count)
            orig = torch.tensor(original_frames[start:end], device=device, dtype=torch.float32).permute(0, 3, 1, 2)
            rec = torch.tensor(restored_frames[start:end], device=device, dtype=torch.float32).permute(0, 3, 1, 2)

            orig_01 = orig / 255.0
            rec_01 = rec / 255.0
            diff = orig_01 - rec_01

            mse_batch = diff.square().mean(dim=(1, 2, 3))
            mae_batch = diff.abs().mean(dim=(1, 2, 3))
            psnr_batch = 10.0 * torch.log10(1.0 / torch.clamp(mse_batch, min=1e-12))
            ssim_batch = ms_ssim(orig_01, rec_01, data_range=1.0, size_average=False)

            orig_lpips = orig_01 * 2.0 - 1.0
            rec_lpips = rec_01 * 2.0 - 1.0
            lpips_batch = lpips_model(orig_lpips, rec_lpips).flatten()

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


def main() -> None:
    args = parse_args()
    skyreels_root = Path(args.skyreels_root)
    ensure_sys_path(skyreels_root)
    from skyreels_v2_infer.modules import WanVAE, download_model

    latent_chunks_path = Path(args.latent_chunks_pt)
    raw_frames_path = Path(args.raw_frames)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(latent_chunks_path, map_location="cpu")
    latent_chunks = payload["latent_chunks"]
    overlap_history = int(payload["overlap_history"])
    fps = int(payload["fps"])
    latent_overlap = (overlap_history - 1) // 4 + 1
    stitched = stitch_chunks(latent_chunks, latent_overlap)

    dedup_latent_path = output_dir / "full_video_latents_dedup.pt"
    torch.save(stitched, dedup_latent_path)

    model_path = args.model_path if args.model_path is not None else download_model(payload["model_id"])
    dtype = torch.float32 if args.weight_dtype == "float32" else torch.bfloat16
    device = torch.device(args.device)
    timings: dict[str, float] = {}

    started = time.time()
    vae = WanVAE(model_path).to(args.device).to(dtype)
    vae.vae.requires_grad_(False)
    vae.vae.eval()
    timings["load_vae_seconds"] = time.time() - started

    started = time.time()
    with torch.no_grad():
        restored = vae.decode(stitched.to(args.device, dtype=dtype)).float().cpu()
    timings["decode_seconds"] = time.time() - started

    restored_frames = tensor_to_uint8_frames(restored)
    restored_frames_path = output_dir / "restored_from_dedup_latents_raw_frames_uint8.npy"
    np.save(restored_frames_path, restored_frames, allow_pickle=False)

    restored_mp4_path = output_dir / "restored_from_dedup_latents.mp4"
    if args.save_mp4:
        started = time.time()
        imageio.mimwrite(
            restored_mp4_path,
            restored_frames,
            fps=fps,
            quality=8,
            output_params=["-loglevel", "error"],
        )
        timings["write_mp4_seconds"] = time.time() - started

    original_frames = np.load(raw_frames_path, mmap_mode="r")
    started = time.time()
    metrics = compute_metrics(original_frames, restored_frames, args.batch_size, device)
    timings["metrics_seconds"] = time.time() - started

    report = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_chunk_file": str(latent_chunks_path),
        "source_raw_frames_file": str(raw_frames_path),
        "dedup_full_latent_file": str(dedup_latent_path),
        "restored_frames_file": str(restored_frames_path),
        "restored_mp4_file": str(restored_mp4_path) if args.save_mp4 else None,
        "device": args.device,
        "weight_dtype": args.weight_dtype,
        "source_chunk_count": len(latent_chunks),
        "source_chunk_shape": list(latent_chunks[0].shape),
        "latent_overlap_timesteps_removed_per_following_chunk": latent_overlap,
        "dedup_full_latent_shape": list(stitched.shape),
        "metrics_vs_original_chunked_decode_raw_frames": metrics,
        "timings_seconds": timings,
    }
    write_json(output_dir / "comparison_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
