#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import statistics as stats
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/root/Wan2.2")

from lighttaew2_2_streaming_decode import (  # noqa: E402
    Wan22LightTAEStreamingDecoder,
    load_saved_latents,
    torch_dtype,
)
from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402


DEFAULT_LATENT_PATHS = [
    Path("/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/000_subject_consistency_r00_p003_a_person_eating_a_burger.pt"),
]


@dataclass
class VariantConfig:
    name: str
    key_interval_latent: int
    vae_window_latent: int
    repair_mode: str
    residual_decay: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment with hybrid Wan2.2 decode: VAE key groups + TAE inter-groups + causal repair.")
    parser.add_argument("--latent-path", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
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
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Repeatable. Format: name:key_interval:vae_window:repair_mode[:residual_decay]. "
             "Example: k8w3_residual:8:3:residual:1.0",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/LatentsCompress/examples/profiling/wan22_hybrid_decode"),
    )
    return parser.parse_args()


def parse_variant(spec: str) -> VariantConfig:
    parts = spec.split(":")
    if len(parts) not in (4, 5):
        raise ValueError(f"Invalid variant spec: {spec}")
    name, key_interval, vae_window, repair_mode, *rest = parts
    return VariantConfig(
        name=name,
        key_interval_latent=int(key_interval),
        vae_window_latent=int(vae_window),
        repair_mode=repair_mode,
        residual_decay=float(rest[0]) if rest else 1.0,
    )


def default_variants() -> list[VariantConfig]:
    return [
        VariantConfig(name="tae_only", key_interval_latent=10_000, vae_window_latent=1, repair_mode="none"),
        VariantConfig(name="k8_w2_keyreplace", key_interval_latent=8, vae_window_latent=2, repair_mode="keyreplace"),
        VariantConfig(name="k8_w2_residual", key_interval_latent=8, vae_window_latent=2, repair_mode="residual", residual_decay=1.0),
        VariantConfig(name="k8_w3_keyreplace", key_interval_latent=8, vae_window_latent=3, repair_mode="keyreplace"),
        VariantConfig(name="k8_w3_residual", key_interval_latent=8, vae_window_latent=3, repair_mode="residual", residual_decay=1.0),
    ]


def maybe_cleanup(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def save_mp4(video_cthw: torch.Tensor, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video_cthw[None], save_file=str(path), fps=fps, nrow=1)


def ffmpeg_metric(ref_mp4: Path, dist_mp4: Path, metric: str) -> dict[str, float | None]:
    if metric == "psnr":
        cmd = [
            "ffmpeg",
            "-i",
            str(ref_mp4),
            "-i",
            str(dist_mp4),
            "-lavfi",
            "psnr",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        match = re.search(r"average:([0-9.]+)", proc.stderr)
        return {"mp4_psnr_db": float(match.group(1)) if match else None}
    if metric == "ssim":
        cmd = [
            "ffmpeg",
            "-i",
            str(ref_mp4),
            "-i",
            str(dist_mp4),
            "-lavfi",
            "ssim",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        match = re.search(r"All:([0-9.]+)", proc.stderr)
        return {"mp4_ssim": float(match.group(1)) if match else None}
    raise ValueError(metric)


def to_uint8_frames(video_cthw: torch.Tensor) -> torch.Tensor:
    frames = ((video_cthw.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return frames.permute(1, 2, 3, 0).contiguous()


def mse_and_psnr(ref: torch.Tensor, dist: torch.Tensor, max_val: float = 255.0) -> tuple[float, float]:
    ref_f = ref.to(torch.float32)
    dist_f = dist.to(torch.float32)
    mse = torch.mean((ref_f - dist_f) ** 2).item()
    if mse == 0.0:
        return 0.0, math.inf
    psnr = 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)
    return mse, psnr


def full_group_start(step_idx: int) -> int:
    if step_idx == 0:
        return 0
    return 1 + 4 * (step_idx - 1)


def full_group_stop(step_idx: int) -> int:
    return full_group_start(step_idx) + (1 if step_idx == 0 else 4)


def extract_group(video_cthw: torch.Tensor, step_idx: int) -> torch.Tensor:
    return video_cthw[:, full_group_start(step_idx):full_group_stop(step_idx)].contiguous()


def decode_vae_full(vae: Wan2_2_VAE, latents_cthw: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
    synchronize(device)
    import time
    t0 = time.perf_counter()
    with torch.inference_mode():
        video = vae.decode([latents_cthw.to(device=device, dtype=torch.float32)])[0]
    synchronize(device)
    dt = time.perf_counter() - t0
    out = video.detach().cpu()
    del video
    maybe_cleanup(device)
    return out, dt


def decode_vae_window_group(vae: Wan2_2_VAE, latents_cthw: torch.Tensor, step_idx: int, window_latent: int, device: str) -> tuple[torch.Tensor, float]:
    import time
    start = max(0, step_idx - window_latent + 1)
    chunk = latents_cthw[:, start:step_idx + 1].contiguous()
    synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        decoded = vae.decode([chunk.to(device=device, dtype=torch.float32)])[0]
    synchronize(device)
    dt = time.perf_counter() - t0
    out = decoded.detach().cpu()
    del decoded
    maybe_cleanup(device)
    if step_idx == 0:
        group = out[:, :1].contiguous()
    else:
        group = out[:, -4:].contiguous()
    return group, dt


def decode_tae_stream_groups(
    decoder: Wan22LightTAEStreamingDecoder,
    latents_cthw: torch.Tensor,
    device: str,
) -> tuple[list[torch.Tensor], list[float], torch.Tensor]:
    import time
    decoder.reset()
    groups: list[torch.Tensor] = []
    secs: list[float] = []
    for step_idx in range(int(latents_cthw.shape[1])):
        chunk = latents_cthw[:, step_idx:step_idx + 1].contiguous()
        synchronize(device)
        t0 = time.perf_counter()
        frames_tchw = decoder.push_latent_chunk(chunk)
        synchronize(device)
        secs.append(time.perf_counter() - t0)
        group = frames_tchw.permute(1, 0, 2, 3).contiguous().mul(2.0).sub(1.0)
        groups.append(group)
    return groups, secs, torch.cat(groups, dim=1)


def apply_repair(
    tae_group: torch.Tensor,
    last_key_residual: torch.Tensor | None,
    distance_from_key: int,
    key_interval_latent: int,
    mode: str,
    residual_decay: float,
) -> torch.Tensor:
    if mode == "none" or mode == "keyreplace" or last_key_residual is None:
        return tae_group
    if mode != "residual":
        raise ValueError(f"Unsupported repair mode: {mode}")
    progress = distance_from_key / max(key_interval_latent, 1)
    weight = max(0.0, 1.0 - progress) * residual_decay
    return (tae_group + weight * last_key_residual).clamp(-1.0, 1.0)


def key_steps(total_steps: int, interval: int) -> list[int]:
    if interval <= 0:
        return [0]
    out = [0]
    step = interval
    while step < total_steps:
        out.append(step)
        step += interval
    return out


def estimate_realtime(
    tae_group_secs: list[float],
    key_decode_secs: list[float],
    total_steps: int,
    target_fps: float,
) -> dict[str, float | bool | None]:
    steady_tae = stats.mean(tae_group_secs[1:]) if len(tae_group_secs) > 1 else tae_group_secs[0]
    steady_tae_per_frame = steady_tae / 4.0 if steady_tae is not None else None
    avg_key_per_step = (sum(key_decode_secs) / total_steps) if total_steps > 0 else None
    avg_key_per_frame = (avg_key_per_step / 4.0) if avg_key_per_step is not None else None
    avg_total_per_frame = None
    if steady_tae_per_frame is not None and avg_key_per_frame is not None:
        avg_total_per_frame = steady_tae_per_frame + avg_key_per_frame
    frame_budget = 1.0 / target_fps if target_fps > 0 else None
    return {
        "steady_tae_group_sec_mean": steady_tae,
        "steady_tae_sec_per_frame": steady_tae_per_frame,
        "avg_key_decode_sec_per_latent_step": avg_key_per_step,
        "avg_key_decode_sec_per_output_frame": avg_key_per_frame,
        "estimated_total_sec_per_output_frame": avg_total_per_frame,
        "target_frame_budget_sec": frame_budget,
        "estimated_realtime_possible": (avg_total_per_frame is not None and frame_budget is not None and avg_total_per_frame <= frame_budget),
    }


def estimate_serial_streaming_budget(
    tae_group_secs: list[float],
    key_decode_by_step: dict[int, float],
    target_fps: float,
) -> dict[str, float | bool | int | list[int] | None]:
    if target_fps <= 0:
        return {
            "serial_required_initial_buffer_sec": None,
            "serial_required_initial_buffer_frames": None,
            "serial_peak_step_decode_sec": None,
            "serial_peak_step_budget_sec": None,
            "serial_deadline_miss_step_count": None,
            "serial_deadline_miss_steps": None,
            "serial_realtime_without_prefetch": None,
        }

    cum_service = 0.0
    cum_budget = 0.0
    max_prefix_excess = 0.0
    deadline_miss_steps: list[int] = []
    peak_step_decode = 0.0
    peak_step_budget = 0.0

    for step_idx, tae_sec in enumerate(tae_group_secs):
        out_frames = 1 if step_idx == 0 else 4
        step_budget = out_frames / target_fps
        step_service = tae_sec + key_decode_by_step.get(step_idx, 0.0)
        cum_service += step_service
        cum_budget += step_budget
        max_prefix_excess = max(max_prefix_excess, cum_service - cum_budget)
        if step_service > step_budget:
            deadline_miss_steps.append(step_idx)
        peak_step_decode = max(peak_step_decode, step_service)
        peak_step_budget = max(peak_step_budget, step_budget)

    return {
        "serial_required_initial_buffer_sec": max_prefix_excess,
        "serial_required_initial_buffer_frames": math.ceil(max_prefix_excess * target_fps),
        "serial_peak_step_decode_sec": peak_step_decode,
        "serial_peak_step_budget_sec": peak_step_budget,
        "serial_deadline_miss_step_count": len(deadline_miss_steps),
        "serial_deadline_miss_steps": deadline_miss_steps,
        "serial_realtime_without_prefetch": max_prefix_excess <= 0.0,
    }


def summarize_psnr(values: list[float]) -> tuple[float | None, float | None, int]:
    if not values:
        return None, None, 0
    finite = [v for v in values if math.isfinite(v)]
    exact = len(values) - len(finite)
    if not finite:
        return math.inf, math.inf, exact
    return stats.mean(finite), min(finite), exact


def run_variant(
    variant: VariantConfig,
    baseline_video: torch.Tensor,
    tae_groups: list[torch.Tensor],
    tae_group_secs: list[float],
    latents_cthw: torch.Tensor,
    fps: int,
    device: str,
    output_dir: Path,
    vae_window_cache: dict[int, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    total_steps = int(latents_cthw.shape[1])
    ksteps = set(key_steps(total_steps, variant.key_interval_latent))
    last_key_residual: torch.Tensor | None = None
    last_key_step = 0
    key_decode_secs: list[float] = []
    key_decode_by_step: dict[int, float] = {}
    anchor_psnrs: list[float] = []
    anchor_tae_psnrs: list[float] = []
    parts: list[torch.Tensor] = []

    for step_idx in range(total_steps):
        tae_group = tae_groups[step_idx]
        if step_idx in ksteps and variant.repair_mode != "none":
            cached = vae_window_cache[variant.vae_window_latent][step_idx]
            vae_group = cached["group"]
            dt = cached["decode_sec"]
            key_decode_secs.append(dt)
            key_decode_by_step[step_idx] = dt
            anchor_psnrs.append(cached["anchor_raw_frame_psnr_db"])
            anchor_tae_psnrs.append(cached["anchor_vs_tae_raw_frame_psnr_db"])
            last_key_residual = (vae_group - tae_group).detach().clone()
            last_key_step = step_idx
            parts.append(vae_group)
        else:
            repaired = apply_repair(
                tae_group=tae_group,
                last_key_residual=last_key_residual,
                distance_from_key=step_idx - last_key_step,
                key_interval_latent=variant.key_interval_latent,
                mode=variant.repair_mode,
                residual_decay=variant.residual_decay,
            )
            parts.append(repaired)

    video = torch.cat(parts, dim=1)
    raw_ref = to_uint8_frames(baseline_video)
    raw_dist = to_uint8_frames(video)
    raw_mse, raw_psnr = mse_and_psnr(raw_ref, raw_dist, 255.0)

    mp4_path = output_dir / f"{variant.name}.mp4"
    save_mp4(video, mp4_path, fps=fps)

    anchor_mean, anchor_min, anchor_exact = summarize_psnr(anchor_psnrs)
    anchor_tae_mean, anchor_tae_min, anchor_tae_exact = summarize_psnr(anchor_tae_psnrs)

    report = {
        "variant": variant.name,
        "key_interval_latent": variant.key_interval_latent,
        "vae_window_latent": variant.vae_window_latent,
        "repair_mode": variant.repair_mode,
        "residual_decay": variant.residual_decay,
        "raw_frame_mse": raw_mse,
        "raw_frame_psnr_db": raw_psnr,
        "mp4_path": str(mp4_path),
        "key_steps": sorted(ksteps),
        "key_decode_sec_total": sum(key_decode_secs),
        "key_decode_sec_mean": (sum(key_decode_secs) / len(key_decode_secs)) if key_decode_secs else 0.0,
        "anchor_raw_frame_psnr_db_mean": anchor_mean,
        "anchor_raw_frame_psnr_db_min": anchor_min,
        "anchor_raw_frame_psnr_exact_match_count": anchor_exact,
        "anchor_vs_tae_raw_frame_psnr_db_mean": anchor_tae_mean,
        "anchor_vs_tae_raw_frame_psnr_db_min": anchor_tae_min,
        "anchor_vs_tae_raw_frame_psnr_exact_match_count": anchor_tae_exact,
    }
    report.update(estimate_realtime(tae_group_secs, key_decode_secs, total_steps, float(fps)))
    report.update(estimate_serial_streaming_budget(tae_group_secs, key_decode_by_step, float(fps)))
    return report


def summarize_sample_variants(variants: list[dict[str, Any]]) -> dict[str, Any]:
    best_raw = max(variants, key=lambda row: row["raw_frame_psnr_db"])
    best_mp4 = max(variants, key=lambda row: row.get("mp4_psnr_db") or -1.0)
    realtime = [row["variant"] for row in variants if row.get("estimated_realtime_possible")]
    return {
        "best_raw_psnr_variant": best_raw["variant"],
        "best_raw_psnr_db": best_raw["raw_frame_psnr_db"],
        "best_mp4_psnr_variant": best_mp4["variant"],
        "best_mp4_psnr_db": best_mp4.get("mp4_psnr_db"),
        "estimated_realtime_possible_variants": realtime,
    }


def main() -> None:
    args = parse_args()
    latent_paths = [Path(p) for p in args.latent_path] if args.latent_path else list(DEFAULT_LATENT_PATHS)
    if args.limit is not None:
        latent_paths = latent_paths[:args.limit]
    variants = [parse_variant(spec) for spec in args.variant] if args.variant else default_variants()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    vae = Wan2_2_VAE(
        vae_pth=str(args.ckpt_dir / "Wan2.2_VAE.pth"),
        dtype=torch_dtype(args.dtype),
        device=args.device,
    )
    tae = Wan22LightTAEStreamingDecoder(
        lightx2v_root=args.lightx2v_root,
        checkpoint_path=args.lighttae_path,
        device=args.device,
        dtype=torch_dtype(args.dtype),
        need_scaled=True,
    )

    all_samples = []
    for latent_path in latent_paths:
        latents, meta = load_saved_latents(latent_path)
        fps = int(meta.get("fps", 24))
        stem = latent_path.stem
        sample_dir = output_root / stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        baseline_video, baseline_sec = decode_vae_full(vae, latents, args.device)
        baseline_mp4 = sample_dir / "baseline_vae.mp4"
        save_mp4(baseline_video, baseline_mp4, fps=fps)

        tae_groups, tae_group_secs, tae_video = decode_tae_stream_groups(tae, latents, args.device)
        tae_only_mp4 = sample_dir / "tae_only.mp4"
        save_mp4(tae_video, tae_only_mp4, fps=fps)
        tae_only_raw_ref = to_uint8_frames(baseline_video)
        tae_only_raw_dist = to_uint8_frames(tae_video)
        tae_only_raw_mse, tae_only_raw_psnr = mse_and_psnr(tae_only_raw_ref, tae_only_raw_dist, 255.0)
        tae_only = {
            "variant": "tae_only",
            "key_interval_latent": None,
            "vae_window_latent": None,
            "repair_mode": "none",
            "residual_decay": None,
            "raw_frame_mse": tae_only_raw_mse,
            "raw_frame_psnr_db": tae_only_raw_psnr,
            "mp4_path": str(tae_only_mp4),
            "key_steps": [],
            "key_decode_sec_total": 0.0,
            "key_decode_sec_mean": 0.0,
            "anchor_raw_frame_psnr_db_mean": None,
            "anchor_raw_frame_psnr_db_min": None,
            "anchor_raw_frame_psnr_exact_match_count": 0,
            "anchor_vs_tae_raw_frame_psnr_db_mean": None,
            "anchor_vs_tae_raw_frame_psnr_db_min": None,
            "anchor_vs_tae_raw_frame_psnr_exact_match_count": 0,
        }
        tae_only.update(estimate_realtime(tae_group_secs, [], int(latents.shape[1]), float(fps)))
        tae_only.update(estimate_serial_streaming_budget(tae_group_secs, {}, float(fps)))

        per_variant = [tae_only]
        needed_windows = sorted({variant.vae_window_latent for variant in variants if variant.repair_mode != "none"})
        window_cache: dict[int, dict[int, dict[str, Any]]] = {}
        for window_latent in needed_windows:
            cache_for_window: dict[int, dict[str, Any]] = {}
            all_ksteps = sorted({
                step_idx
                for v in variants
                if v.vae_window_latent == window_latent and v.repair_mode != "none"
                for step_idx in key_steps(int(latents.shape[1]), v.key_interval_latent)
            })
            for step_idx in all_ksteps:
                group, dt = decode_vae_window_group(
                    vae=vae,
                    latents_cthw=latents,
                    step_idx=step_idx,
                    window_latent=window_latent,
                    device=args.device,
                )
                ref_group = extract_group(baseline_video, step_idx)
                tae_group = tae_groups[step_idx]
                anchor_mse, anchor_psnr = mse_and_psnr(
                    to_uint8_frames(ref_group),
                    to_uint8_frames(group),
                    255.0,
                )
                anchor_tae_mse, anchor_tae_psnr = mse_and_psnr(
                    to_uint8_frames(ref_group),
                    to_uint8_frames(tae_group),
                    255.0,
                )
                cache_for_window[step_idx] = {
                    "group": group,
                    "decode_sec": dt,
                    "anchor_raw_frame_mse": anchor_mse,
                    "anchor_raw_frame_psnr_db": anchor_psnr,
                    "anchor_vs_tae_raw_frame_mse": anchor_tae_mse,
                    "anchor_vs_tae_raw_frame_psnr_db": anchor_tae_psnr,
                }
            window_cache[window_latent] = cache_for_window
        for variant in variants:
            if variant.name == "tae_only":
                continue
            report = run_variant(
                variant=variant,
                baseline_video=baseline_video,
                tae_groups=tae_groups,
                tae_group_secs=tae_group_secs,
                latents_cthw=latents,
                fps=fps,
                device=args.device,
                output_dir=sample_dir,
                vae_window_cache=window_cache,
            )
            per_variant.append(report)

        for row in per_variant:
            mp4_metrics = {}
            mp4_metrics.update(ffmpeg_metric(baseline_mp4, Path(row["mp4_path"]), "psnr"))
            mp4_metrics.update(ffmpeg_metric(baseline_mp4, Path(row["mp4_path"]), "ssim"))
            row.update(mp4_metrics)
            row["mp4_bytes"] = Path(row["mp4_path"]).stat().st_size

        sample_report = {
            "latent_path": str(latent_path),
            "meta": meta,
            "baseline_vae_decode_sec": baseline_sec,
            "baseline_mp4": str(baseline_mp4),
            "variants": per_variant,
            "summary": summarize_sample_variants(per_variant),
        }
        (sample_dir / "report.json").write_text(json.dumps(sample_report, indent=2), encoding="utf-8")
        all_samples.append(sample_report)
        maybe_cleanup(args.device)

    final_report = {
        "latent_paths": [str(p) for p in latent_paths],
        "variants": [variant.__dict__ for variant in variants],
        "samples": all_samples,
    }
    out_json = output_root / "hybrid_decode_report.json"
    out_json.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()
