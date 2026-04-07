#!/usr/bin/env python3
import argparse
import json
import math
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skyreels-root", default="/root/SkyReels-V2")
    parser.add_argument("--latent-chunks-pt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weight-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--save-restored-frames", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def ensure_sys_path(skyreels_root: Path) -> None:
    root = str(skyreels_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stitch_chunks(chunks: list[torch.Tensor], latent_overlap: int) -> torch.Tensor:
    stitched_parts = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            stitched_parts.append(chunk)
        else:
            stitched_parts.append(chunk[:, :, latent_overlap:, :, :])
    return torch.cat(stitched_parts, dim=2)


def tensor_to_uint8_frames(video_tensor: torch.Tensor) -> np.ndarray:
    video_tensor = video_tensor.clamp(-1, 1)
    video_tensor = (video_tensor / 2 + 0.5).clamp(0, 1)
    video_tensor = video_tensor[0].permute(1, 2, 3, 0) * 255
    return video_tensor.cpu().numpy().astype(np.uint8)


def main():
    args = parse_args()
    skyreels_root = Path(args.skyreels_root)
    ensure_sys_path(skyreels_root)
    from skyreels_v2_infer.modules import WanVAE, download_model

    latent_chunks_path = Path(args.latent_chunks_pt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(latent_chunks_path, map_location="cpu")
    latent_chunks = payload["latent_chunks"]
    overlap_history = int(payload["overlap_history"])
    fps = int(payload["fps"])
    num_frames = int(payload["num_frames"])

    latent_overlap = (overlap_history - 1) // 4 + 1
    stitched = stitch_chunks(latent_chunks, latent_overlap)
    latent_full_path = output_dir / "full_video_latents_dedup.pt"
    torch.save(
        {
            "latents_stage": "pre_vae_decode_full_video_dedup",
            "description": "Full-video latents stitched from async pre-VAE chunks with overlap removed in latent time.",
            "source_chunk_file": str(latent_chunks_path),
            "video_id": payload["video_id"],
            "prompt": payload["prompt"],
            "seed": payload["seed"],
            "model_id": payload["model_id"],
            "resolution": payload["resolution"],
            "width": payload["width"],
            "height": payload["height"],
            "fps": fps,
            "num_frames": num_frames,
            "overlap_history": overlap_history,
            "latent_overlap_timesteps": latent_overlap,
            "source_chunk_count": len(latent_chunks),
            "source_chunk_shape": list(latent_chunks[0].shape),
            "dedup_full_latent": stitched.to("cpu"),
            "dedup_full_latent_shape": list(stitched.shape),
            "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        latent_full_path,
    )

    model_path = args.model_path if args.model_path is not None else download_model(payload["model_id"])
    dtype = torch.float32 if args.weight_dtype == "float32" else torch.bfloat16
    vae = WanVAE(model_path).to(args.device).to(dtype)
    vae.vae.requires_grad_(False)
    vae.vae.eval()

    with torch.no_grad():
        restored = vae.decode(stitched.to(args.device, dtype=dtype)).float().cpu()
    restored_frames = tensor_to_uint8_frames(restored)

    restored_frames_path = output_dir / "restored_from_dedup_latents_raw_frames_uint8.npy"
    restored_mp4_path = output_dir / "restored_from_dedup_latents.mp4"
    if args.save_restored_frames:
        np.save(restored_frames_path, restored_frames, allow_pickle=False)
    imageio.mimwrite(restored_mp4_path, restored_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])

    metrics = {}
    original_frames_path = latent_chunks_path.parent / "raw_frames_uint8.npy"
    if original_frames_path.exists():
        original_frames = np.load(original_frames_path, mmap_mode="r")
        compare_frames = min(len(original_frames), len(restored_frames))
        orig = original_frames[:compare_frames].astype(np.float32) / 255.0
        rec = restored_frames[:compare_frames].astype(np.float32) / 255.0
        mse = float(np.mean((orig - rec) ** 2))
        mae = float(np.mean(np.abs(orig - rec)))
        psnr = float("inf") if mse == 0 else float(10.0 * math.log10(1.0 / mse))
        metrics = {
            "compared_frames": compare_frames,
            "mse": mse,
            "mae": mae,
            "psnr": psnr,
        }

    report = {
        "source_chunk_file": str(latent_chunks_path),
        "full_dedup_latent_file": str(latent_full_path),
        "restored_frames_file": str(restored_frames_path) if args.save_restored_frames else None,
        "restored_mp4_file": str(restored_mp4_path),
        "latent_overlap_timesteps_removed_per_following_chunk": latent_overlap,
        "source_chunk_count": len(latent_chunks),
        "source_chunk_shape": list(latent_chunks[0].shape),
        "dedup_full_latent_shape": list(stitched.shape),
        "source_total_latent_timesteps": int(sum(chunk.shape[2] for chunk in latent_chunks)),
        "dedup_total_latent_timesteps": int(stitched.shape[2]),
        "expected_total_latent_timesteps": int((num_frames - 1) // 4 + 1),
        "metrics_vs_original_raw_frames": metrics,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(output_dir / "dedup_restore_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
