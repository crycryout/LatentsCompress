#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.cuda.amp as amp

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/root/Wan2.2")

from lighttaew2_2_streaming_decode import (  # noqa: E402
    Wan22LightTAEStreamingDecoder,
    load_saved_latents,
    torch_dtype,
)
from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused decode harness for Nsight Systems / Nsight Compute on Wan2.2 VAE and lighttaew2_2.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "wan_vae_wrapper",
            "wan_vae_direct_batch",
            "lighttae_full",
            "lighttae_stream",
        ],
    )
    parser.add_argument(
        "--latent-path",
        type=Path,
        default=Path("/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/000_subject_consistency_r00_p003_a_person_eating_a_burger.pt"),
    )
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--max-latent-steps", type=int, default=None)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def maybe_cleanup(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def nvtx_push(name: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def nvtx_pop() -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def load_latents(path: Path, max_latent_steps: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
    latents, meta = load_saved_latents(path)
    if max_latent_steps is not None:
        latents = latents[:, :max_latent_steps].contiguous()
    return latents, meta


def run_wan_vae_wrapper(vae: Wan2_2_VAE, latents: torch.Tensor, batch_size: int, device: str) -> dict[str, Any]:
    batch = [latents.to(device=device, dtype=torch.float32) for _ in range(batch_size)]
    with torch.inference_mode():
        nvtx_push("decode")
        synchronize(device)
        t0 = time.perf_counter()
        outs = vae.decode(batch)
        synchronize(device)
        dt = time.perf_counter() - t0
        nvtx_pop()
    frames = sum(int(out.shape[1]) for out in outs)
    del outs, batch
    return {
        "batch_size": batch_size,
        "total_frames": frames,
        "wall_sec": dt,
        "frames_per_sec": (frames / dt) if dt > 0 else None,
    }


def run_wan_vae_direct_batch(vae: Wan2_2_VAE, latents: torch.Tensor, batch_size: int, device: str, dtype_name: str) -> dict[str, Any]:
    batch = latents.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        nvtx_push("decode")
        synchronize(device)
        t0 = time.perf_counter()
        with amp.autocast(dtype=torch_dtype(dtype_name)):
            out = vae.model.decode(batch, vae.scale).float().clamp_(-1, 1)
        synchronize(device)
        dt = time.perf_counter() - t0
        nvtx_pop()
    frames = int(out.shape[0] * out.shape[2])
    del out, batch
    return {
        "batch_size": batch_size,
        "total_frames": frames,
        "wall_sec": dt,
        "frames_per_sec": (frames / dt) if dt > 0 else None,
    }


def run_lighttae_full(decoder: Wan22LightTAEStreamingDecoder, latents: torch.Tensor, device: str) -> dict[str, Any]:
    lat_b = latents.unsqueeze(0).to(device=device, dtype=torch.float32)
    lat_b = decoder._scale_latents(lat_b)
    lat_ntchw = lat_b.permute(0, 2, 1, 3, 4).contiguous().to(device=device, dtype=decoder.dtype)
    with torch.inference_mode():
        nvtx_push("decode")
        synchronize(device)
        t0 = time.perf_counter()
        out = decoder.model.decode_video(lat_ntchw, parallel=False, show_progress_bar=False)
        synchronize(device)
        dt = time.perf_counter() - t0
        nvtx_pop()
    frames = int(out.shape[1])
    del out, lat_b, lat_ntchw
    return {
        "batch_size": 1,
        "total_frames": frames,
        "wall_sec": dt,
        "frames_per_sec": (frames / dt) if dt > 0 else None,
    }


def run_lighttae_stream(decoder: Wan22LightTAEStreamingDecoder, latents: torch.Tensor, chunk_size: int, device: str) -> dict[str, Any]:
    decoder.reset()
    total_frames = 0
    groups = []
    with torch.inference_mode():
        nvtx_push("decode")
        synchronize(device)
        t0 = time.perf_counter()
        for start in range(0, int(latents.shape[1]), chunk_size):
            chunk = latents[:, start:start + chunk_size].contiguous()
            out = decoder.push_latent_chunk(chunk)
            emitted = int(out.shape[0]) if out.ndim == 4 else 0
            groups.append(
                {
                    "latent_start": start,
                    "decoded_frames": emitted,
                }
            )
            total_frames += emitted
            del out
        synchronize(device)
        dt = time.perf_counter() - t0
        nvtx_pop()
    return {
        "batch_size": 1,
        "total_frames": total_frames,
        "wall_sec": dt,
        "frames_per_sec": (total_frames / dt) if dt > 0 else None,
        "groups": groups,
    }


def build_runner(args: argparse.Namespace, latents: torch.Tensor):
    if args.mode.startswith("wan_vae"):
        vae = Wan2_2_VAE(
            vae_pth=str(args.ckpt_dir / "Wan2.2_VAE.pth"),
            dtype=torch_dtype(args.dtype),
            device=args.device,
        )
        if args.mode == "wan_vae_wrapper":
            return lambda: run_wan_vae_wrapper(vae, latents, args.batch_size, args.device)
        return lambda: run_wan_vae_direct_batch(vae, latents, args.batch_size, args.device, args.dtype)

    decoder = Wan22LightTAEStreamingDecoder(
        lightx2v_root=args.lightx2v_root,
        checkpoint_path=args.lighttae_path,
        device=args.device,
        dtype=torch_dtype(args.dtype),
        need_scaled=True,
    )
    if args.mode == "lighttae_full":
        return lambda: run_lighttae_full(decoder, latents, args.device)
    return lambda: run_lighttae_stream(decoder, latents, args.chunk_size, args.device)


def main() -> None:
    args = parse_args()
    latents, meta = load_latents(args.latent_path, args.max_latent_steps)
    runner = build_runner(args, latents)

    for _ in range(args.warmup_iters):
        _ = runner()
        maybe_cleanup(args.device)

    runs = []
    for _ in range(args.profile_iters):
        runs.append(runner())
        maybe_cleanup(args.device)

    report = {
        "mode": args.mode,
        "latent_path": str(args.latent_path),
        "latent_shape": list(latents.shape),
        "meta": meta,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "warmup_iters": args.warmup_iters,
        "profile_iters": args.profile_iters,
        "runs": runs,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
