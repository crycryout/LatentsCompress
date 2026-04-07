#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import lpips
import numpy as np
import torch
from numpy.lib.format import open_memmap
from pytorch_msssim import ssim as ms_ssim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skyreels-root", default="/root/SkyReels-V2")
    parser.add_argument("--model-id", default="Skywork/SkyReels-V2-DF-14B-720P")
    parser.add_argument(
        "--input-frames",
        default=(
            "/root/SkyReels-V2/result/skyreels_v2_dynamic10_720p24_async/"
            "wingsuit_rescue_glacier_pullup/raw_frames_uint8.npy"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/root/SkyReels-V2/result/skyreels_v2_dynamic10_720p24_async/"
            "wingsuit_rescue_glacier_pullup/vae_roundtrip_eval"
        ),
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def ensure_sys_path(skyreels_root: Path) -> None:
    root = str(skyreels_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_vae_path(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path
    candidate = model_path / "Wan2.1_VAE.pth"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not locate Wan2.1_VAE.pth under {model_path}")


def bytes_to_human(num_bytes: int) -> dict:
    return {
        "bytes": int(num_bytes),
        "KiB": num_bytes / 1024,
        "MiB": num_bytes / 1024**2,
        "GiB": num_bytes / 1024**3,
        "MB": num_bytes / 1000**2,
        "GB": num_bytes / 1000**3,
    }


def tensor_video_from_numpy(frames_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    video = torch.tensor(np.asarray(frames_uint8), device=device, dtype=torch.bfloat16)
    video = video.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    return video.div_(127.5).sub_(1.0)


def save_recon_frames(
    recon_video: torch.Tensor,
    output_path: Path,
) -> np.memmap:
    # recon_video: [1, C, T, H, W] in [-1, 1]
    frames = ((recon_video[0].permute(1, 2, 3, 0) / 2 + 0.5).clamp(0, 1) * 255.0).round()
    frames_uint8 = frames.to(torch.uint8).cpu().numpy()
    arr = open_memmap(output_path, mode="w+", dtype=np.uint8, shape=frames_uint8.shape)
    arr[:] = frames_uint8
    arr.flush()
    return arr


def compute_metrics(
    original_frames: np.ndarray,
    recon_frames: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict:
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    frame_count = original_frames.shape[0]

    mse_values = []
    mae_values = []
    psnr_values = []
    ssim_values = []
    lpips_values = []

    with torch.no_grad():
        for start in range(0, frame_count, batch_size):
            end = min(start + batch_size, frame_count)
            orig = torch.tensor(np.asarray(original_frames[start:end]), device=device, dtype=torch.float32).permute(
                0, 3, 1, 2
            )
            recon = torch.tensor(np.asarray(recon_frames[start:end]), device=device, dtype=torch.float32).permute(
                0, 3, 1, 2
            )

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

    def summarize(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "std": float(arr.std(ddof=0)),
        }

    return {
        "frame_count": frame_count,
        "mse": summarize(mse_values),
        "mae": summarize(mae_values),
        "psnr_db": summarize(psnr_values),
        "ssim": summarize(ssim_values),
        "lpips_alex": summarize(lpips_values),
    }


def main():
    args = parse_args()
    skyreels_root = Path(args.skyreels_root)
    input_frames_path = Path(args.input_frames)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_sys_path(skyreels_root)

    from skyreels_v2_infer.modules import download_model, get_vae

    device = torch.device("cuda")
    model_path = Path(download_model(args.model_id))
    vae_path = resolve_vae_path(model_path)

    original_frames = np.load(input_frames_path, mmap_mode="r")
    if original_frames.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frames, got {original_frames.dtype}")

    timings = {}
    started = time.time()
    vae = get_vae(str(vae_path), device=device, weight_dtype=torch.bfloat16)
    timings["load_vae_seconds"] = time.time() - started

    video = tensor_video_from_numpy(np.asarray(original_frames), device=device)

    started = time.time()
    with torch.no_grad():
        encoded_latents = vae.encode(video)
    timings["encode_seconds"] = time.time() - started

    latents_cpu = encoded_latents.detach().cpu()
    latents_path = output_dir / "vae_encoded_latents.pt"
    torch.save(
        {
            "latents_stage": "vae_encoded_full_video",
            "description": "Video encoded from raw uint8 frames using SkyReels WanVAE, before VAE decode.",
            "model_id": args.model_id,
            "input_frames_path": str(input_frames_path),
            "fps": args.fps,
            "latent_shape": list(latents_cpu.shape),
            "latent_dtype": str(latents_cpu.dtype),
            "latents": latents_cpu,
        },
        latents_path,
    )

    started = time.time()
    with torch.no_grad():
        decoded_video = vae.decode(encoded_latents.to(device=device, dtype=torch.bfloat16))
    timings["decode_seconds"] = time.time() - started

    recon_frames_path = output_dir / "reconstructed_raw_frames_uint8.npy"
    recon_memmap = save_recon_frames(decoded_video, recon_frames_path)

    recon_mp4_path = output_dir / "reconstructed_roundtrip.mp4"
    started = time.time()
    imageio.mimwrite(
        recon_mp4_path,
        recon_memmap,
        fps=args.fps,
        quality=8,
        output_params=["-loglevel", "error"],
    )
    timings["write_mp4_seconds"] = time.time() - started

    # Free VAE tensors before metric pass.
    del video
    del encoded_latents
    del decoded_video
    del vae
    torch.cuda.empty_cache()

    started = time.time()
    metrics = compute_metrics(original_frames, recon_memmap, args.batch_size, device)
    timings["metrics_seconds"] = time.time() - started

    report = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": args.model_id,
        "resolved_model_path": str(model_path),
        "resolved_vae_path": str(vae_path),
        "input_frames_path": str(input_frames_path),
        "output_dir": str(output_dir),
        "fps": args.fps,
        "batch_size": args.batch_size,
        "original_frames": {
            "shape": list(original_frames.shape),
            "dtype": str(original_frames.dtype),
            **bytes_to_human(input_frames_path.stat().st_size),
        },
        "encoded_latents": {
            "path": str(latents_path),
            "shape": list(latents_cpu.shape),
            "dtype": str(latents_cpu.dtype),
            **bytes_to_human(latents_path.stat().st_size),
        },
        "reconstructed_frames": {
            "path": str(recon_frames_path),
            "shape": list(recon_memmap.shape),
            "dtype": str(recon_memmap.dtype),
            **bytes_to_human(recon_frames_path.stat().st_size),
        },
        "reconstructed_mp4": {
            "path": str(recon_mp4_path),
            **bytes_to_human(recon_mp4_path.stat().st_size),
        },
        "size_ratios": {
            "raw_to_latents": input_frames_path.stat().st_size / latents_path.stat().st_size,
            "raw_to_reconstructed_mp4": input_frames_path.stat().st_size / recon_mp4_path.stat().st_size,
            "latents_to_reconstructed_mp4": latents_path.stat().st_size / recon_mp4_path.stat().st_size,
        },
        "timings_seconds": timings,
        "metrics": metrics,
    }

    report_path = output_dir / "metrics_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    main()
