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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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
    Path("/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/015_motion_smoothness_r03_p064_a_bear_climbing_a_tree.pt"),
]


@dataclass
class VariantConfig:
    name: str
    scheduler: str
    key_window_latent: int
    fixed_interval: int
    min_interval: int
    max_interval: int
    decay_steps: float
    lowpass_kernel: int
    tile_rows: int
    tile_cols: int
    correction_divisor: int
    scene_threshold: float


@dataclass
class CorrectionState:
    global_gain: torch.Tensor
    global_bias: torch.Tensor
    tile_gain: torch.Tensor
    tile_bias: torch.Tensor
    lowpass_kernel: int
    tile_rows: int
    tile_cols: int
    corr_h: int
    corr_w: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid Wan2.2 decode v2: sparse VAE keys + streaming TAE + causal low-frequency/tile-wise correction."
    )
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
        "--output-root",
        type=Path,
        default=Path("/root/LatentsCompress/examples/profiling/wan22_hybrid_decode_v2"),
    )
    return parser.parse_args()


def default_variants() -> list[VariantConfig]:
    return [
        VariantConfig(
            name="tae_only",
            scheduler="none",
            key_window_latent=1,
            fixed_interval=9999,
            min_interval=9999,
            max_interval=9999,
            decay_steps=1.0,
            lowpass_kernel=1,
            tile_rows=1,
            tile_cols=1,
            correction_divisor=1,
            scene_threshold=999.0,
        ),
        VariantConfig(
            name="fixed_k8_w3_lfcolor_tile",
            scheduler="fixed",
            key_window_latent=3,
            fixed_interval=8,
            min_interval=4,
            max_interval=8,
            decay_steps=6.0,
            lowpass_kernel=15,
            tile_rows=4,
            tile_cols=4,
            correction_divisor=8,
            scene_threshold=0.10,
        ),
        VariantConfig(
            name="adaptive_w3_lfcolor_tile",
            scheduler="adaptive",
            key_window_latent=3,
            fixed_interval=8,
            min_interval=4,
            max_interval=10,
            decay_steps=6.0,
            lowpass_kernel=15,
            tile_rows=4,
            tile_cols=4,
            correction_divisor=8,
            scene_threshold=0.085,
        ),
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
    save_video(video_cthw.detach().cpu()[None], save_file=str(path), fps=fps, nrow=1)


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


def avg_blur(video_cthw: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return video_cthw
    pad = kernel // 2
    tchw = video_cthw.permute(1, 0, 2, 3).contiguous()
    blurred = F.avg_pool2d(tchw, kernel_size=kernel, stride=1, padding=pad)
    return blurred.permute(1, 0, 2, 3).contiguous()


def resize_spatial(video_cthw: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    tchw = video_cthw.permute(1, 0, 2, 3).contiguous()
    resized = F.interpolate(tchw, size=out_hw, mode="bilinear", align_corners=False)
    return resized.permute(1, 0, 2, 3).contiguous()


def downsample_lowfreq(video_cthw: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    tchw = video_cthw.permute(1, 0, 2, 3).contiguous()
    resized = F.interpolate(tchw, size=out_hw, mode="area")
    return resized.permute(1, 0, 2, 3).contiguous()


def channel_affine(src: torch.Tensor, dst: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    c = src.shape[0]
    src_flat = src.view(c, -1)
    dst_flat = dst.view(c, -1)
    src_mean = src_flat.mean(dim=1).view(c, 1, 1, 1)
    dst_mean = dst_flat.mean(dim=1).view(c, 1, 1, 1)
    src_std = src_flat.std(dim=1, unbiased=False).view(c, 1, 1, 1)
    dst_std = dst_flat.std(dim=1, unbiased=False).view(c, 1, 1, 1)
    gain = (dst_std + eps) / (src_std + eps)
    bias = dst_mean - gain * src_mean
    return gain, bias


def tile_bounds(length: int, parts: int) -> list[tuple[int, int]]:
    edges = [round(i * length / parts) for i in range(parts + 1)]
    return [(edges[i], edges[i + 1]) for i in range(parts)]


def tile_affine(src: torch.Tensor, dst: torch.Tensor, rows: int, cols: int, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    c, _, h, w = src.shape
    gain = torch.ones((c, rows, cols), dtype=src.dtype, device=src.device)
    bias = torch.zeros((c, rows, cols), dtype=src.dtype, device=src.device)
    row_bounds = tile_bounds(h, rows)
    col_bounds = tile_bounds(w, cols)
    for r, (hs, he) in enumerate(row_bounds):
        for q, (ws, we) in enumerate(col_bounds):
            src_tile = src[:, :, hs:he, ws:we].contiguous().view(c, -1)
            dst_tile = dst[:, :, hs:he, ws:we].contiguous().view(c, -1)
            src_mean = src_tile.mean(dim=1)
            dst_mean = dst_tile.mean(dim=1)
            src_std = src_tile.std(dim=1, unbiased=False)
            dst_std = dst_tile.std(dim=1, unbiased=False)
            gain[:, r, q] = (dst_std + eps) / (src_std + eps)
            bias[:, r, q] = dst_mean - gain[:, r, q] * src_mean
    return gain, bias


def apply_tile_affine(x: torch.Tensor, gain: torch.Tensor, bias: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    out = x.clone()
    _, _, h, w = x.shape
    row_bounds = tile_bounds(h, rows)
    col_bounds = tile_bounds(w, cols)
    for r, (hs, he) in enumerate(row_bounds):
        for q, (ws, we) in enumerate(col_bounds):
            g = gain[:, r, q].view(-1, 1, 1, 1)
            b = bias[:, r, q].view(-1, 1, 1, 1)
            out[:, :, hs:he, ws:we] = out[:, :, hs:he, ws:we] * g + b
    return out


def build_correction_state(
    tae_group: torch.Tensor,
    vae_group: torch.Tensor,
    lowpass_kernel: int,
    tile_rows: int,
    tile_cols: int,
    correction_divisor: int,
) -> CorrectionState:
    h, w = tae_group.shape[-2:]
    corr_h = max(16, h // max(correction_divisor, 1))
    corr_w = max(16, w // max(correction_divisor, 1))
    tae_ds = downsample_lowfreq(tae_group, (corr_h, corr_w))
    vae_ds = downsample_lowfreq(vae_group, (corr_h, corr_w))
    global_gain, global_bias = channel_affine(tae_ds, vae_ds)
    tae_ds_global = tae_ds * global_gain + global_bias
    tile_gain, tile_bias = tile_affine(tae_ds_global, vae_ds, tile_rows, tile_cols)
    return CorrectionState(
        global_gain=global_gain,
        global_bias=global_bias,
        tile_gain=tile_gain,
        tile_bias=tile_bias,
        lowpass_kernel=lowpass_kernel,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        corr_h=corr_h,
        corr_w=corr_w,
    )


def apply_correction_state(
    tae_group: torch.Tensor,
    state: CorrectionState,
    age_steps: int,
    decay_steps: float,
) -> torch.Tensor:
    low_ds = downsample_lowfreq(tae_group, (state.corr_h, state.corr_w))
    corrected_low = low_ds * state.global_gain + state.global_bias
    corrected_low = apply_tile_affine(corrected_low, state.tile_gain, state.tile_bias, state.tile_rows, state.tile_cols)
    corrected_low = resize_spatial(corrected_low, tuple(tae_group.shape[-2:]))
    high = tae_group - resize_spatial(low_ds, tuple(tae_group.shape[-2:]))
    corrected = (corrected_low + high).clamp(-1.0, 1.0)
    weight = math.exp(-float(age_steps) / max(decay_steps, 1e-4))
    return (tae_group * (1.0 - weight) + corrected * weight).clamp(-1.0, 1.0)


def group_motion_score(cur_group: torch.Tensor, prev_group: torch.Tensor, lowpass_kernel: int) -> float:
    del lowpass_kernel
    cur_low = downsample_lowfreq(cur_group, (32, 32)).mean(dim=1)
    prev_low = downsample_lowfreq(prev_group, (32, 32)).mean(dim=1)
    return torch.mean(torch.abs(cur_low - prev_low)).item()


def decode_vae_full(vae: Wan2_2_VAE, latents_cthw: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
    synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        video = vae.decode([latents_cthw.to(device=device, dtype=torch.float32)])[0]
    synchronize(device)
    dt = time.perf_counter() - t0
    out = video.detach().cpu()
    del video
    maybe_cleanup(device)
    return out, dt


def decode_vae_window_group(
    vae: Wan2_2_VAE,
    latents_cthw: torch.Tensor,
    step_idx: int,
    window_latent: int,
    device: str,
) -> tuple[torch.Tensor, float]:
    start = max(0, step_idx - window_latent + 1)
    chunk = latents_cthw[:, start:step_idx + 1].contiguous()
    synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        decoded = vae.decode([chunk.to(device=device, dtype=torch.float32)])[0]
    synchronize(device)
    dt = time.perf_counter() - t0
    out = decoded.detach()
    del decoded
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
    decoder.reset()
    groups: list[torch.Tensor] = []
    secs: list[float] = []
    for step_idx in range(int(latents_cthw.shape[1])):
        chunk = latents_cthw[:, step_idx:step_idx + 1].contiguous()
        synchronize(device)
        t0 = time.perf_counter()
        frames_tchw = decoder.push_latent_chunk(chunk, return_cpu=False)
        synchronize(device)
        secs.append(time.perf_counter() - t0)
        group = frames_tchw.permute(1, 0, 2, 3).contiguous().mul(2.0).sub(1.0).to(torch.float32)
        groups.append(group)
    return groups, secs, torch.cat(groups, dim=1)


def required_fixed_key_steps(total_steps: int, interval: int) -> list[int]:
    out = [0]
    if total_steps > 1:
        out.append(1)
    step = 1 + interval
    while step < total_steps:
        out.append(step)
        step += interval
    if out[-1] != total_steps - 1:
        out.append(total_steps - 1)
    return sorted(set(out))


def estimate_next_interval(anchor_tae_psnr: float, max_interval: int) -> int:
    if anchor_tae_psnr < 22.0:
        return 4
    if anchor_tae_psnr < 25.0:
        return 6
    if anchor_tae_psnr < 28.0:
        return 8
    return max_interval


def streaming_budget_report(
    step_service_secs: list[float],
    step_output_frames: list[int],
    target_fps: float,
) -> dict[str, float | int | bool]:
    startup = step_service_secs[0]
    backlog = 0.0
    total_stall = 0.0
    stall_events = 0
    peak_backlog = 0.0
    max_step_service = 0.0
    max_step_budget = 0.0
    deadline_miss_steps = 0
    for service, frames in zip(step_service_secs, step_output_frames):
        budget = frames / target_fps
        max_step_service = max(max_step_service, service)
        max_step_budget = max(max_step_budget, budget)
        if service > budget:
            total_stall += service - budget
            stall_events += 1
            deadline_miss_steps += 1
        backlog = max(0.0, backlog + service - budget)
        peak_backlog = max(peak_backlog, backlog)
    return {
        "stream_startup_first_frame_sec": startup,
        "stream_required_initial_buffer_sec": peak_backlog,
        "stream_required_initial_buffer_frames": math.ceil(peak_backlog * target_fps),
        "stream_total_stall_sec_zero_prefetch": total_stall,
        "stream_stall_event_count_zero_prefetch": stall_events,
        "stream_deadline_miss_step_count": deadline_miss_steps,
        "stream_peak_step_service_sec": max_step_service,
        "stream_peak_step_budget_sec": max_step_budget,
        "stream_realtime_without_prefetch": peak_backlog <= 0.0,
    }


def key_background_report(
    key_steps: list[int],
    key_decode_secs: list[float],
    target_fps: float,
) -> dict[str, float | bool | None]:
    nonzero_steps = [step for step in key_steps if step > 0]
    if not nonzero_steps or not key_decode_secs or target_fps <= 0:
        return {
            "async_key_decode_mean_sec": stats.mean(key_decode_secs) if key_decode_secs else 0.0,
            "async_key_gap_mean_sec": None,
            "async_key_gap_min_sec": None,
            "async_key_decode_utilization_mean": None,
            "async_key_decode_utilization_max": None,
            "async_key_decode_feasible_if_fully_parallel_front": None,
        }

    gap_secs = []
    utilizations = []
    prev = 0
    for step, sec in zip(nonzero_steps, key_decode_secs[1:]):
        gap = (step - prev) * 4.0 / target_fps if prev > 0 else (1 + max(step - 1, 0) * 4.0) / target_fps
        gap_secs.append(gap)
        utilizations.append(sec / gap if gap > 0 else math.inf)
        prev = step
    return {
        "async_key_decode_mean_sec": stats.mean(key_decode_secs),
        "async_key_gap_mean_sec": stats.mean(gap_secs) if gap_secs else None,
        "async_key_gap_min_sec": min(gap_secs) if gap_secs else None,
        "async_key_decode_utilization_mean": stats.mean(utilizations) if utilizations else None,
        "async_key_decode_utilization_max": max(utilizations) if utilizations else None,
        "async_key_decode_feasible_if_fully_parallel_front": all(u <= 1.0 for u in utilizations) if utilizations else None,
    }


def select_key_steps(
    variant: VariantConfig,
    tae_groups: list[torch.Tensor],
    fetch_anchor: Any,
) -> tuple[list[int], dict[int, dict[str, Any]], list[float]]:
    total_steps = len(tae_groups)
    key_info: dict[int, dict[str, Any]] = {}
    motion_scores = [0.0] * total_steps
    key_steps = [0]
    if total_steps > 1:
        key_steps.append(1)

    for step_idx in key_steps:
        key_info[step_idx] = fetch_anchor(step_idx)

    if variant.scheduler == "none":
        return sorted(set(key_steps)), key_info, motion_scores

    if variant.scheduler == "fixed":
        for step_idx in required_fixed_key_steps(total_steps, variant.fixed_interval):
            if step_idx not in key_info:
                key_info[step_idx] = fetch_anchor(step_idx)
        return sorted(key_info.keys()), key_info, motion_scores

    if variant.scheduler != "adaptive":
        raise ValueError(f"Unsupported scheduler: {variant.scheduler}")

    last_key = 1 if total_steps > 1 else 0
    next_interval = estimate_next_interval(key_info[last_key]["anchor_vs_tae_raw_frame_psnr_db"], variant.max_interval)
    prev_group = tae_groups[last_key]
    for step_idx in range(last_key + 1, total_steps):
        motion = group_motion_score(tae_groups[step_idx], prev_group, variant.lowpass_kernel)
        motion_scores[step_idx] = motion
        age = step_idx - last_key
        should_key = False
        if step_idx == total_steps - 1:
            should_key = True
        elif age >= next_interval:
            should_key = True
        elif age >= variant.min_interval and motion > variant.scene_threshold:
            should_key = True
        if should_key:
            key_steps.append(step_idx)
            info = fetch_anchor(step_idx)
            info["trigger"] = "scene" if (age >= variant.min_interval and motion > variant.scene_threshold and step_idx != total_steps - 1 and age < next_interval) else "interval"
            key_info[step_idx] = info
            last_key = step_idx
            next_interval = estimate_next_interval(info["anchor_vs_tae_raw_frame_psnr_db"], variant.max_interval)
        prev_group = tae_groups[step_idx]
    return sorted(set(key_steps)), key_info, motion_scores


def run_variant(
    variant: VariantConfig,
    baseline_video: torch.Tensor,
    baseline_mp4: Path,
    tae_groups: list[torch.Tensor],
    tae_group_secs: list[float],
    fps: int,
    sample_dir: Path,
    fetch_anchor: Any,
) -> dict[str, Any]:
    if variant.scheduler == "none":
        video = torch.cat(tae_groups, dim=1)
        raw_ref = to_uint8_frames(baseline_video).cpu()
        raw_dist = to_uint8_frames(video).cpu()
        raw_mse, raw_psnr = mse_and_psnr(raw_ref, raw_dist, 255.0)
        mp4_path = sample_dir / f"{variant.name}.mp4"
        save_mp4(video, mp4_path, fps=fps)
        report = {
            "variant": variant.name,
            "scheduler": variant.scheduler,
            "key_steps": [],
            "raw_frame_mse": raw_mse,
            "raw_frame_psnr_db": raw_psnr,
            "mp4_path": str(mp4_path),
            "key_decode_sec_total": 0.0,
            "key_decode_sec_mean": 0.0,
            "correction_sec_total": 0.0,
            "correction_sec_mean": 0.0,
        }
        report.update(ffmpeg_metric(baseline_mp4, mp4_path, "psnr"))
        report.update(ffmpeg_metric(baseline_mp4, mp4_path, "ssim"))
        report.update(
            streaming_budget_report(
                step_service_secs=list(tae_group_secs),
                step_output_frames=[1] + [4] * (len(tae_group_secs) - 1),
                target_fps=float(fps),
            )
        )
        return report

    key_steps, key_info, motion_scores = select_key_steps(variant, tae_groups, fetch_anchor)
    key_set = set(key_steps)

    parts: list[torch.Tensor] = []
    step_service_secs: list[float] = []
    correction_secs: list[float] = []
    key_decode_secs: list[float] = []
    async_step_service_secs: list[float] = []
    last_key_step = 0
    state = build_correction_state(
        tae_group=tae_groups[0],
        vae_group=key_info[0]["group"],
        lowpass_kernel=variant.lowpass_kernel,
        tile_rows=variant.tile_rows,
        tile_cols=variant.tile_cols,
        correction_divisor=variant.correction_divisor,
    )

    for step_idx, tae_group in enumerate(tae_groups):
        t0 = time.perf_counter()
        if step_idx in key_set:
            info = key_info[step_idx]
            vae_group = info["group"]
            parts.append(vae_group)
            state = build_correction_state(
                tae_group=tae_group,
                vae_group=vae_group,
                lowpass_kernel=variant.lowpass_kernel,
                tile_rows=variant.tile_rows,
                tile_cols=variant.tile_cols,
                correction_divisor=variant.correction_divisor,
            )
            last_key_step = step_idx
            key_decode_secs.append(info["decode_sec"])
            correction_sec = time.perf_counter() - t0
            correction_secs.append(correction_sec)
            step_service_secs.append(tae_group_secs[step_idx] + info["decode_sec"] + correction_sec)
            async_step_service_secs.append(tae_group_secs[step_idx] + correction_sec)
        else:
            corrected = apply_correction_state(
                tae_group=tae_group,
                state=state,
                age_steps=step_idx - last_key_step,
                decay_steps=variant.decay_steps,
            )
            parts.append(corrected)
            correction_sec = time.perf_counter() - t0
            correction_secs.append(correction_sec)
            step_service_secs.append(tae_group_secs[step_idx] + correction_sec)
            async_step_service_secs.append(tae_group_secs[step_idx] + correction_sec)

    video = torch.cat(parts, dim=1)
    raw_ref = to_uint8_frames(baseline_video).cpu()
    raw_dist = to_uint8_frames(video).cpu()
    raw_mse, raw_psnr = mse_and_psnr(raw_ref, raw_dist, 255.0)
    mp4_path = sample_dir / f"{variant.name}.mp4"
    save_mp4(video, mp4_path, fps=fps)

    trigger_counts: dict[str, int] = {}
    for step_idx in key_steps:
        trigger = key_info[step_idx].get("trigger", "warmup")
        trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1

    report = {
        "variant": variant.name,
        "scheduler": variant.scheduler,
        "key_steps": key_steps,
        "key_count": len(key_steps),
        "raw_frame_mse": raw_mse,
        "raw_frame_psnr_db": raw_psnr,
        "mp4_path": str(mp4_path),
        "key_decode_sec_total": sum(key_decode_secs),
        "key_decode_sec_mean": stats.mean(key_decode_secs) if key_decode_secs else 0.0,
        "correction_sec_total": sum(correction_secs),
        "correction_sec_mean": stats.mean(correction_secs) if correction_secs else 0.0,
        "anchor_vs_tae_raw_frame_psnr_db_mean": stats.mean(info["anchor_vs_tae_raw_frame_psnr_db"] for info in key_info.values()),
        "anchor_vs_tae_raw_frame_psnr_db_min": min(info["anchor_vs_tae_raw_frame_psnr_db"] for info in key_info.values()),
        "adaptive_motion_score_max": max(motion_scores) if motion_scores else 0.0,
        "adaptive_trigger_counts": trigger_counts,
    }
    report.update(ffmpeg_metric(baseline_mp4, mp4_path, "psnr"))
    report.update(ffmpeg_metric(baseline_mp4, mp4_path, "ssim"))
    report.update(
        streaming_budget_report(
            step_service_secs=step_service_secs,
            step_output_frames=[1] + [4] * (len(step_service_secs) - 1),
            target_fps=float(fps),
        )
    )
    async_front = streaming_budget_report(
        step_service_secs=async_step_service_secs,
        step_output_frames=[1] + [4] * (len(async_step_service_secs) - 1),
        target_fps=float(fps),
    )
    report.update({
        f"async_front_{k}": v for k, v in async_front.items()
    })
    report.update(key_background_report(key_steps, key_decode_secs, float(fps)))
    return report


def summarize_sample_variants(variants: list[dict[str, Any]]) -> dict[str, Any]:
    best_raw = max(variants, key=lambda row: row["raw_frame_psnr_db"])
    best_mp4 = max(variants, key=lambda row: row.get("mp4_psnr_db") or -1.0)
    best_stream = min(variants, key=lambda row: row.get("stream_required_initial_buffer_sec", math.inf))
    return {
        "best_raw_psnr_variant": best_raw["variant"],
        "best_raw_psnr_db": best_raw["raw_frame_psnr_db"],
        "best_mp4_psnr_variant": best_mp4["variant"],
        "best_mp4_psnr_db": best_mp4.get("mp4_psnr_db"),
        "lowest_buffer_variant": best_stream["variant"],
        "lowest_buffer_sec": best_stream.get("stream_required_initial_buffer_sec"),
    }


def main() -> None:
    args = parse_args()
    latent_paths = [Path(p) for p in args.latent_path] if args.latent_path else list(DEFAULT_LATENT_PATHS)
    if args.limit is not None:
        latent_paths = latent_paths[:args.limit]
    variants = default_variants()

    args.output_root.mkdir(parents=True, exist_ok=True)

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

    all_samples: list[dict[str, Any]] = []
    for latent_path in latent_paths:
        latents, meta = load_saved_latents(latent_path)
        fps = int(meta.get("fps", 24))
        sample_dir = args.output_root / latent_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        baseline_video, baseline_sec = decode_vae_full(vae, latents, args.device)
        baseline_mp4 = sample_dir / "baseline_vae.mp4"
        save_mp4(baseline_video, baseline_mp4, fps=fps)

        tae_groups, tae_group_secs, _ = decode_tae_stream_groups(tae, latents, args.device)

        anchor_cache: dict[int, dict[str, Any]] = {}

        def fetch_anchor(step_idx: int) -> dict[str, Any]:
            if step_idx not in anchor_cache:
                group, dt = decode_vae_window_group(
                    vae=vae,
                    latents_cthw=latents,
                    step_idx=step_idx,
                    window_latent=3,
                    device=args.device,
                )
                tae_group = tae_groups[step_idx]
                anchor_mse, anchor_psnr = mse_and_psnr(
                    to_uint8_frames(group),
                    to_uint8_frames(tae_group),
                    255.0,
                )
                anchor_cache[step_idx] = {
                    "group": group,
                    "decode_sec": dt,
                    "anchor_vs_tae_raw_frame_mse": anchor_mse,
                    "anchor_vs_tae_raw_frame_psnr_db": anchor_psnr,
                }
            return anchor_cache[step_idx]

        per_variant = []
        for variant in variants:
            report = run_variant(
                variant=variant,
                baseline_video=baseline_video,
                baseline_mp4=baseline_mp4,
                tae_groups=tae_groups,
                tae_group_secs=tae_group_secs,
                fps=fps,
                sample_dir=sample_dir,
                fetch_anchor=fetch_anchor,
            )
            per_variant.append(report)

        sample_report = {
            "latent_path": str(latent_path),
            "meta": meta,
            "baseline_vae_decode_sec": baseline_sec,
            "baseline_mp4": str(baseline_mp4),
            "variants": per_variant,
            "summary": summarize_sample_variants(per_variant),
        }
        with open(sample_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(sample_report, f, indent=2)
        all_samples.append(sample_report)

    overall: dict[str, Any] = {
        "latent_paths": [str(p) for p in latent_paths],
        "variants": [variant.__dict__ for variant in variants],
        "samples": all_samples,
    }
    with open(args.output_root / "hybrid_decode_v2_report.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print(args.output_root / "hybrid_decode_v2_report.json")


if __name__ == "__main__":
    main()
