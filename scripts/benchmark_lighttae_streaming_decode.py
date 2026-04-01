#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import statistics as stats
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lighttaew2_2_streaming_decode import (  # noqa: E402
    Wan22LightTAEStreamingDecoder,
    load_saved_latents,
    torch_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark step-wise lighttaew2_2 streaming decode on saved Wan2.2 latents.")
    parser.add_argument(
        "--latent-path",
        type=Path,
        default=Path("/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/000_subject_consistency_r00_p003_a_person_eating_a_burger.pt"),
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
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=5)
    parser.add_argument("--max-latent-steps", type=int, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def maybe_cleanup(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def median(xs: list[float]) -> float | None:
    return stats.median(xs) if xs else None


def pstdev(xs: list[float]) -> float | None:
    return stats.pstdev(xs) if len(xs) > 1 else 0.0 if xs else None


def cv(xs: list[float]) -> float | None:
    if not xs:
        return None
    m = mean(xs)
    if not m:
        return None
    s = pstdev(xs)
    return s / m if s is not None else None


def bench_once(
    decoder: Wan22LightTAEStreamingDecoder,
    latents: torch.Tensor,
    chunk_size: int,
    device: str,
) -> dict[str, Any]:
    decoder.reset()
    groups: list[dict[str, Any]] = []
    total_frames = 0
    total_wall_t0 = time.perf_counter()
    for start in range(0, int(latents.shape[1]), chunk_size):
        chunk = latents[:, start:start + chunk_size].contiguous()
        synchronize(device)
        t0 = time.perf_counter()
        frames = decoder.push_latent_chunk(chunk)
        synchronize(device)
        dt = time.perf_counter() - t0
        emitted = int(frames.shape[0]) if frames.ndim == 4 else 0
        groups.append(
            {
                "latent_start": start,
                "latent_len": int(chunk.shape[1]),
                "decoded_frames": emitted,
                "decode_sec": dt,
                "decode_fps": (emitted / dt) if dt > 0 else None,
            }
        )
        total_frames += emitted
        del frames
        maybe_cleanup(device)
    total_wall = time.perf_counter() - total_wall_t0

    first = groups[0]
    steady = groups[1:]
    steady_secs = [g["decode_sec"] for g in steady]
    steady_frames = [g["decoded_frames"] for g in steady]
    steady_4frame_secs = [g["decode_sec"] for g in steady if g["decoded_frames"] == 4]
    steady_total_frames = sum(steady_frames)
    steady_total_sec = sum(steady_secs)
    steady_fps = (steady_total_frames / steady_total_sec) if steady_total_sec > 0 else None

    return {
        "total_latent_steps": int(latents.shape[1]),
        "total_frames": total_frames,
        "total_wall_sec": total_wall,
        "throughput_fps": (total_frames / total_wall) if total_wall > 0 else None,
        "first_frame_latency_sec": first["decode_sec"],
        "first_group_frames": first["decoded_frames"],
        "steady_group_count": len(steady),
        "steady_group_frames_unique": sorted(set(steady_frames)),
        "steady_4frame_decode_sec_mean": mean(steady_4frame_secs),
        "steady_4frame_decode_sec_median": median(steady_4frame_secs),
        "steady_4frame_decode_sec_cv": cv(steady_4frame_secs),
        "steady_fps": steady_fps,
        "groups": groups,
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(key: str) -> list[float]:
        out: list[float] = []
        for run in runs:
            value = run.get(key)
            if value is not None:
                out.append(float(value))
        return out

    return {
        "run_count": len(runs),
        "first_frame_latency_sec_mean": mean(collect("first_frame_latency_sec")),
        "first_frame_latency_sec_median": median(collect("first_frame_latency_sec")),
        "steady_4frame_decode_sec_mean": mean(collect("steady_4frame_decode_sec_mean")),
        "steady_4frame_decode_sec_median": median(collect("steady_4frame_decode_sec_mean")),
        "steady_fps_mean": mean(collect("steady_fps")),
        "steady_fps_median": median(collect("steady_fps")),
        "throughput_fps_mean": mean(collect("throughput_fps")),
        "throughput_fps_median": median(collect("throughput_fps")),
    }


def main() -> None:
    args = parse_args()
    latents, meta = load_saved_latents(args.latent_path)
    if args.max_latent_steps is not None:
        latents = latents[:, :args.max_latent_steps].contiguous()

    decoder = Wan22LightTAEStreamingDecoder(
        lightx2v_root=args.lightx2v_root,
        checkpoint_path=args.lighttae_path,
        device=args.device,
        dtype=torch_dtype(args.dtype),
        need_scaled=True,
    )

    for _ in range(args.warmup_runs):
        _ = bench_once(decoder, latents, args.chunk_size, args.device)
        maybe_cleanup(args.device)

    runs = []
    for _ in range(args.measure_runs):
        runs.append(bench_once(decoder, latents, args.chunk_size, args.device))
        maybe_cleanup(args.device)

    ordered = sorted(runs, key=lambda run: float(run["total_wall_sec"]))
    median_run = ordered[len(ordered) // 2]
    report = {
        "latent_path": str(args.latent_path),
        "meta": meta,
        "chunk_size": args.chunk_size,
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.measure_runs,
        "median_run": median_run,
        "summary": summarize_runs(runs),
        "runs": runs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
