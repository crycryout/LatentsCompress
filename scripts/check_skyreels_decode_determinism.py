#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether SkyReels decoding is deterministic for two decode methods: "
            "chunked overlap-aware decode and direct full-latent decode."
        )
    )
    parser.add_argument("--skyreels-root", default="/root/SkyReels-V2")
    parser.add_argument(
        "--model-path",
        default=(
            "/root/.cache/huggingface/hub/models--Skywork--SkyReels-V2-DF-14B-720P/"
            "snapshots/21bae7ad4dd1335500988ce7ad56d6c532fc8a44/Wan2.1_VAE.pth"
        ),
    )
    parser.add_argument("--latent-chunks-pt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    return parser.parse_args()


def ensure_sys_path(skyreels_root: Path) -> None:
    root = str(skyreels_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def tensor_to_uint8_frames(video_tensor: torch.Tensor) -> np.ndarray:
    video_tensor = video_tensor.clamp(-1, 1)
    video_tensor = (video_tensor / 2 + 0.5).clamp(0, 1)
    video_tensor = video_tensor.permute(1, 2, 3, 0) * 255.0
    return video_tensor.round().to(torch.uint8).cpu().numpy()


def stitch_chunks(chunks: list[torch.Tensor], latent_overlap: int) -> torch.Tensor:
    stitched = []
    for idx, chunk in enumerate(chunks):
        stitched.append(chunk if idx == 0 else chunk[:, :, latent_overlap:, :, :])
    return torch.cat(stitched, dim=2).contiguous()


def compute_psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def save_mp4(frames: np.ndarray, path: Path, fps: int) -> None:
    imageio.mimwrite(
        path,
        frames,
        fps=fps,
        quality=8,
        output_params=["-loglevel", "error"],
    )


def compare_runs(frames_a: np.ndarray, frames_b: np.ndarray, mp4_a: Path, mp4_b: Path) -> dict:
    mp4_frames_a = iio.imread(mp4_a)
    mp4_frames_b = iio.imread(mp4_b)
    return {
        "raw_frames_psnr_db": compute_psnr_u8(frames_a, frames_b),
        "raw_frames_identical_bytes": bool(np.array_equal(frames_a, frames_b)),
        "mp4_psnr_db": compute_psnr_u8(mp4_frames_a, mp4_frames_b),
        "mp4_identical_file_bytes": bool(mp4_a.read_bytes() == mp4_b.read_bytes()),
        "mp4_identical_sha256": bool(sha256_path(mp4_a) == sha256_path(mp4_b)),
    }


def main() -> None:
    args = parse_args()
    skyreels_root = Path(args.skyreels_root)
    ensure_sys_path(skyreels_root)
    from skyreels_v2_infer.modules import WanVAE

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(Path(args.latent_chunks_pt), map_location="cpu")
    latent_chunks = payload["latent_chunks"]
    overlap_history = int(payload["overlap_history"])
    fps = int(payload["fps"])
    latent_overlap = (overlap_history - 1) // 4 + 1
    dtype = torch.float32 if args.weight_dtype == "float32" else torch.bfloat16

    started = time.time()
    vae = WanVAE(args.model_path).to(args.device).to(dtype)
    vae.vae.requires_grad_(False)
    vae.vae.eval()
    timings = {"load_vae_seconds": time.time() - started}

    def decode_chunked_once(tag: str) -> tuple[np.ndarray, Path]:
        parts = []
        started_local = time.time()
        with torch.no_grad():
            for idx, chunk in enumerate(latent_chunks):
                video = vae.decode(chunk.to(args.device, dtype=dtype)).float().cpu()[0]
                frames = tensor_to_uint8_frames(video)
                parts.append(frames if idx == 0 else frames[overlap_history:])
        timings[f"chunked_decode_{tag}_seconds"] = time.time() - started_local
        merged = np.concatenate(parts, axis=0)
        np.save(output_dir / f"chunked_decode_{tag}_raw_frames_uint8.npy", merged, allow_pickle=False)
        mp4_path = output_dir / f"chunked_decode_{tag}.mp4"
        save_mp4(merged, mp4_path, fps)
        return merged, mp4_path

    def decode_direct_once(tag: str, full_latents: torch.Tensor) -> tuple[np.ndarray, Path]:
        started_local = time.time()
        with torch.no_grad():
            video = vae.decode(full_latents.to(args.device, dtype=dtype)).float().cpu()[0]
        timings[f"direct_decode_{tag}_seconds"] = time.time() - started_local
        frames = tensor_to_uint8_frames(video)
        np.save(output_dir / f"direct_decode_{tag}_raw_frames_uint8.npy", frames, allow_pickle=False)
        mp4_path = output_dir / f"direct_decode_{tag}.mp4"
        save_mp4(frames, mp4_path, fps)
        return frames, mp4_path

    full_latents = stitch_chunks(latent_chunks, latent_overlap)
    torch.save(full_latents, output_dir / "full_video_latents_dedup.pt")

    chunked_a, chunked_mp4_a = decode_chunked_once("a")
    chunked_b, chunked_mp4_b = decode_chunked_once("b")
    direct_a, direct_mp4_a = decode_direct_once("a", full_latents)
    direct_b, direct_mp4_b = decode_direct_once("b", full_latents)

    report = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_chunk_file": str(args.latent_chunks_pt),
        "source_chunk_count": len(latent_chunks),
        "source_chunk_shape": list(latent_chunks[0].shape),
        "overlap_history_frames": overlap_history,
        "latent_overlap_timesteps": latent_overlap,
        "full_dedup_latent_shape": list(full_latents.shape),
        "device": args.device,
        "weight_dtype": args.weight_dtype,
        "chunked_method": compare_runs(chunked_a, chunked_b, chunked_mp4_a, chunked_mp4_b),
        "direct_method": compare_runs(direct_a, direct_b, direct_mp4_a, direct_mp4_b),
        "cross_method": {
            "raw_frames_psnr_db": compute_psnr_u8(chunked_a, direct_a),
            "raw_frames_identical_bytes": bool(np.array_equal(chunked_a, direct_a)),
            "mp4_psnr_db": compute_psnr_u8(iio.imread(chunked_mp4_a), iio.imread(direct_mp4_a)),
            "mp4_identical_file_bytes": bool(chunked_mp4_a.read_bytes() == direct_mp4_a.read_bytes()),
            "mp4_identical_sha256": bool(sha256_path(chunked_mp4_a) == sha256_path(direct_mp4_a)),
        },
        "timings_seconds": timings,
    }
    write_json(output_dir / "determinism_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
