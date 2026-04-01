#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import statistics as stats
import sys
import time
from pathlib import Path

import torch
import torch.cuda.amp as amp

sys.path.insert(0, "/root/Wan2.2")

from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Wan2.2 VAE batch scaling for wrapper-serial and direct batched decode paths.")
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def maybe_cleanup(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def median(xs: list[float]) -> float:
    return float(stats.median(xs))


def load_latents(path: Path) -> tuple[torch.Tensor, dict]:
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict):
        meta = {k: v for k, v in blob.items() if k != "latents"}
        latents = blob["latents"]
    else:
        meta = {}
        latents = blob
    return latents.to(torch.float32).cpu(), meta


def time_wrapper_decode(vae: Wan2_2_VAE, latents: torch.Tensor, batch_size: int, device: str) -> float:
    batch = [latents.to(device=device, dtype=torch.float32) for _ in range(batch_size)]
    with torch.inference_mode():
        synchronize(device)
        t0 = time.perf_counter()
        outs = vae.decode(batch)
        synchronize(device)
        dt = time.perf_counter() - t0
    del outs, batch
    maybe_cleanup(device)
    return dt


def time_direct_batch_decode(vae: Wan2_2_VAE, latents: torch.Tensor, batch_size: int, device: str, dtype_name: str) -> float:
    batch = latents.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        synchronize(device)
        t0 = time.perf_counter()
        with amp.autocast(dtype=torch_dtype(dtype_name)):
            out = vae.model.decode(batch, vae.scale).float().clamp_(-1, 1)
        synchronize(device)
        dt = time.perf_counter() - t0
    del out, batch
    maybe_cleanup(device)
    return dt


def main() -> None:
    args = parse_args()
    latents, meta = load_latents(args.latent_path)
    vae = Wan2_2_VAE(
        vae_pth=str(args.ckpt_dir / "Wan2.2_VAE.pth"),
        dtype=torch_dtype(args.dtype),
        device=args.device,
    )

    results = []
    for batch_size in args.batch_sizes:
        for _ in range(args.warmup_runs):
            _ = time_wrapper_decode(vae, latents, batch_size, args.device)
            _ = time_direct_batch_decode(vae, latents, batch_size, args.device, args.dtype)

        wrapper_runs = [time_wrapper_decode(vae, latents, batch_size, args.device) for _ in range(args.measure_runs)]
        direct_runs = [time_direct_batch_decode(vae, latents, batch_size, args.device, args.dtype) for _ in range(args.measure_runs)]

        results.append(
            {
                "batch_size": batch_size,
                "wrapper_serial_median_sec": median(wrapper_runs),
                "wrapper_serial_mean_sec": sum(wrapper_runs) / len(wrapper_runs),
                "direct_batch_median_sec": median(direct_runs),
                "direct_batch_mean_sec": sum(direct_runs) / len(direct_runs),
            }
        )

    b1_wrapper = next(row["wrapper_serial_median_sec"] for row in results if row["batch_size"] == 1)
    b1_direct = next(row["direct_batch_median_sec"] for row in results if row["batch_size"] == 1)
    for row in results:
        row["wrapper_vs_b1_ratio"] = row["wrapper_serial_median_sec"] / b1_wrapper if b1_wrapper > 0 else None
        row["direct_vs_b1_ratio"] = row["direct_batch_median_sec"] / b1_direct if b1_direct > 0 else None
        row["wrapper_per_sample_median_sec"] = row["wrapper_serial_median_sec"] / row["batch_size"]
        row["direct_per_sample_median_sec"] = row["direct_batch_median_sec"] / row["batch_size"]

    report = {
        "latent_path": str(args.latent_path),
        "meta": meta,
        "measure_runs": args.measure_runs,
        "warmup_runs": args.warmup_runs,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
