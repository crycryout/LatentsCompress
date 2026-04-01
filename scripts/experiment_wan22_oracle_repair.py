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


@dataclass
class VariantConfig:
    name: str
    key_interval_latent: int
    mode: str
    decay: float = 1.0
    lowpass_kernel: int = 33
    tile_size: int = 64
    affine_clip: float = 1.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle-anchor analysis for Wan2.2 hybrid repair.")
    parser.add_argument("--latent-path", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=Path("/workspace/models/Wan2.2-TI2V-5B"))
    parser.add_argument("--lighttae-path", type=Path, default=Path("/root/models/vae/lighttaew2_2.safetensors"))
    parser.add_argument("--lightx2v-root", type=Path, default=Path("/root/LightX2V"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/LatentsCompress/examples/profiling/wan22_oracle_repair"),
    )
    return parser.parse_args()


def default_latents() -> list[Path]:
    return [
        Path("/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/000_subject_consistency_r00_p003_a_person_eating_a_burger.pt"),
        Path("/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/008_temporal_flickering_r00_p003_a_tranquil_tableau_of_alley.pt"),
    ]


def default_variants() -> list[VariantConfig]:
    return [
        VariantConfig(name="oracle_k8_keyreplace", key_interval_latent=8, mode="keyreplace"),
        VariantConfig(name="oracle_k8_lpf", key_interval_latent=8, mode="lpf", decay=0.9, lowpass_kernel=33),
        VariantConfig(name="oracle_k8_tile_affine", key_interval_latent=8, mode="tile_affine", decay=0.92, lowpass_kernel=33, tile_size=64, affine_clip=1.25),
        VariantConfig(name="oracle_k8_combo", key_interval_latent=8, mode="combo", decay=0.94, lowpass_kernel=33, tile_size=64, affine_clip=1.25),
        VariantConfig(name="oracle_k12_combo", key_interval_latent=12, mode="combo", decay=0.96, lowpass_kernel=33, tile_size=64, affine_clip=1.25),
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
        cmd = ["ffmpeg", "-i", str(ref_mp4), "-i", str(dist_mp4), "-lavfi", "psnr", "-f", "null", "-"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        match = re.search(r"average:([0-9.]+)", proc.stderr)
        return {"mp4_psnr_db": float(match.group(1)) if match else None}
    if metric == "ssim":
        cmd = ["ffmpeg", "-i", str(ref_mp4), "-i", str(dist_mp4), "-lavfi", "ssim", "-f", "null", "-"]
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


def key_steps(total_steps: int, interval: int) -> list[int]:
    out = [0]
    step = interval
    while step < total_steps:
        out.append(step)
        step += interval
    return out


def decode_vae_full(vae: Wan2_2_VAE, latents_cthw: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
    import time
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


def lowpass_group(group_cthw: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return group_cthw
    frames = group_cthw.permute(1, 0, 2, 3)
    pad = kernel // 2
    low = F.avg_pool2d(frames, kernel_size=kernel, stride=1, padding=pad)
    return low.permute(1, 0, 2, 3).contiguous()


def compute_tile_affine_params(
    src_cthw: torch.Tensor,
    dst_cthw: torch.Tensor,
    tile_size: int,
    affine_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    frames_src = src_cthw.permute(1, 0, 2, 3)
    frames_dst = dst_cthw.permute(1, 0, 2, 3)
    h, w = frames_src.shape[-2:]
    grid_h = max(1, math.ceil(h / tile_size))
    grid_w = max(1, math.ceil(w / tile_size))
    mu_src = F.adaptive_avg_pool2d(frames_src, (grid_h, grid_w))
    mu_dst = F.adaptive_avg_pool2d(frames_dst, (grid_h, grid_w))
    var_src = F.adaptive_avg_pool2d(frames_src * frames_src, (grid_h, grid_w)) - mu_src * mu_src
    var_dst = F.adaptive_avg_pool2d(frames_dst * frames_dst, (grid_h, grid_w)) - mu_dst * mu_dst
    std_src = var_src.clamp_min(1e-6).sqrt()
    std_dst = var_dst.clamp_min(1e-6).sqrt()
    a = (std_dst / std_src).clamp(1.0 / affine_clip, affine_clip)
    b = mu_dst - a * mu_src
    return a, b


def apply_tile_affine(
    group_cthw: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    lowpass_kernel: int,
) -> torch.Tensor:
    work = group_cthw.to(torch.float32)
    frames = work.permute(1, 0, 2, 3)
    low = lowpass_group(work, lowpass_kernel).permute(1, 0, 2, 3)
    h, w = low.shape[-2:]
    a_up = F.interpolate(a, size=(h, w), mode="bilinear", align_corners=False)
    b_up = F.interpolate(b, size=(h, w), mode="bilinear", align_corners=False)
    corrected_low = a_up * low + b_up
    out = corrected_low + (frames - low)
    return out.permute(1, 0, 2, 3).contiguous().clamp(-1.0, 1.0)


def estimate_realtime(tae_group_secs: list[float], total_steps: int, target_fps: float) -> dict[str, float | bool | None]:
    steady_tae = stats.mean(tae_group_secs[1:]) if len(tae_group_secs) > 1 else tae_group_secs[0]
    steady_tae_per_frame = steady_tae / 4.0 if steady_tae is not None else None
    frame_budget = 1.0 / target_fps if target_fps > 0 else None
    return {
        "steady_tae_group_sec_mean": steady_tae,
        "steady_tae_sec_per_frame": steady_tae_per_frame,
        "estimated_total_sec_per_output_frame": steady_tae_per_frame,
        "target_frame_budget_sec": frame_budget,
        "estimated_realtime_possible": (
            steady_tae_per_frame is not None and frame_budget is not None and steady_tae_per_frame <= frame_budget
        ),
    }


def run_variant(
    variant: VariantConfig,
    baseline_video: torch.Tensor,
    tae_groups: list[torch.Tensor],
    tae_group_secs: list[float],
    fps: int,
    output_dir: Path,
) -> dict[str, Any]:
    total_steps = len(tae_groups)
    ksteps = set(key_steps(total_steps, variant.key_interval_latent))
    last_low_residual: torch.Tensor | None = None
    last_affine: tuple[torch.Tensor, torch.Tensor] | None = None
    last_key_step = 0
    parts: list[torch.Tensor] = []
    anchor_psnrs: list[float] = []
    tae_key_psnrs: list[float] = []

    for step_idx in range(total_steps):
        tae_group = tae_groups[step_idx]
        ref_group = extract_group(baseline_video, step_idx)
        if step_idx in ksteps:
            key_ref = ref_group
            key_tae = tae_group
            anchor_mse, anchor_psnr = mse_and_psnr(to_uint8_frames(ref_group), to_uint8_frames(key_ref))
            del anchor_mse
            tae_mse, tae_psnr = mse_and_psnr(to_uint8_frames(ref_group), to_uint8_frames(key_tae))
            del tae_mse
            anchor_psnrs.append(anchor_psnr)
            tae_key_psnrs.append(tae_psnr)
            last_low_residual = lowpass_group(key_ref, variant.lowpass_kernel) - lowpass_group(key_tae, variant.lowpass_kernel)
            last_affine = compute_tile_affine_params(
                lowpass_group(key_tae, variant.lowpass_kernel),
                lowpass_group(key_ref, variant.lowpass_kernel),
                variant.tile_size,
                variant.affine_clip,
            )
            last_key_step = step_idx
            parts.append(key_ref)
            continue

        progress = (step_idx - last_key_step) / max(variant.key_interval_latent, 1)
        weight = max(0.0, 1.0 - progress) * variant.decay
        if variant.mode == "keyreplace":
            corrected = tae_group
        elif variant.mode == "lpf":
            assert last_low_residual is not None
            corrected = (tae_group + weight * last_low_residual).clamp(-1.0, 1.0)
        elif variant.mode == "tile_affine":
            assert last_affine is not None
            affine_group = apply_tile_affine(tae_group, last_affine[0], last_affine[1], variant.lowpass_kernel)
            corrected = torch.lerp(tae_group.to(torch.float32), affine_group, weight).clamp(-1.0, 1.0)
        elif variant.mode == "combo":
            assert last_low_residual is not None and last_affine is not None
            affine_group = apply_tile_affine(tae_group, last_affine[0], last_affine[1], variant.lowpass_kernel)
            combo = affine_group + weight * last_low_residual
            corrected = torch.lerp(tae_group.to(torch.float32), combo.clamp(-1.0, 1.0), weight).clamp(-1.0, 1.0)
        else:
            raise ValueError(variant.mode)
        parts.append(corrected)

    video = torch.cat(parts, dim=1)
    raw_mse, raw_psnr = mse_and_psnr(to_uint8_frames(baseline_video), to_uint8_frames(video))
    mp4_path = output_dir / f"{variant.name}.mp4"
    save_mp4(video, mp4_path, fps)
    report = {
        "variant": variant.name,
        "key_interval_latent": variant.key_interval_latent,
        "mode": variant.mode,
        "decay": variant.decay,
        "lowpass_kernel": variant.lowpass_kernel,
        "tile_size": variant.tile_size,
        "affine_clip": variant.affine_clip,
        "raw_frame_mse": raw_mse,
        "raw_frame_psnr_db": raw_psnr,
        "key_steps": sorted(ksteps),
        "anchor_raw_frame_psnr_db_mean": stats.mean(anchor_psnrs) if anchor_psnrs else None,
        "tae_key_raw_frame_psnr_db_mean": stats.mean(tae_key_psnrs) if tae_key_psnrs else None,
        "mp4_path": str(mp4_path),
    }
    report.update(estimate_realtime(tae_group_secs, total_steps, float(fps)))
    return report


def summarize_sample(variants: list[dict[str, Any]]) -> dict[str, Any]:
    best_raw = max(variants, key=lambda row: row["raw_frame_psnr_db"])
    best_mp4 = max(variants, key=lambda row: row.get("mp4_psnr_db") or -1.0)
    return {
        "best_raw_psnr_variant": best_raw["variant"],
        "best_raw_psnr_db": best_raw["raw_frame_psnr_db"],
        "best_mp4_psnr_variant": best_mp4["variant"],
        "best_mp4_psnr_db": best_mp4.get("mp4_psnr_db"),
    }


def main() -> None:
    args = parse_args()
    latent_paths = [Path(p) for p in args.latent_path] if args.latent_path else default_latents()
    if args.limit is not None:
        latent_paths = latent_paths[:args.limit]
    variants = default_variants()

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
        sample_dir = output_root / latent_path.stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        baseline_video, baseline_sec = decode_vae_full(vae, latents, args.device)
        baseline_mp4 = sample_dir / "baseline_vae.mp4"
        save_mp4(baseline_video, baseline_mp4, fps)

        tae_groups, tae_group_secs, tae_video = decode_tae_stream_groups(tae, latents, args.device)
        tae_mp4 = sample_dir / "tae_only.mp4"
        save_mp4(tae_video, tae_mp4, fps)
        tae_mse, tae_psnr = mse_and_psnr(to_uint8_frames(baseline_video), to_uint8_frames(tae_video))
        rows = [{
            "variant": "tae_only",
            "key_interval_latent": None,
            "mode": "none",
            "decay": None,
            "lowpass_kernel": None,
            "tile_size": None,
            "affine_clip": None,
            "raw_frame_mse": tae_mse,
            "raw_frame_psnr_db": tae_psnr,
            "anchor_raw_frame_psnr_db_mean": None,
            "tae_key_raw_frame_psnr_db_mean": None,
            "mp4_path": str(tae_mp4),
            **estimate_realtime(tae_group_secs, int(latents.shape[1]), float(fps)),
        }]

        for variant in variants:
            rows.append(run_variant(variant, baseline_video, tae_groups, tae_group_secs, fps, sample_dir))

        for row in rows:
            row.update(ffmpeg_metric(baseline_mp4, Path(row["mp4_path"]), "psnr"))
            row.update(ffmpeg_metric(baseline_mp4, Path(row["mp4_path"]), "ssim"))
            row["mp4_bytes"] = Path(row["mp4_path"]).stat().st_size

        sample_report = {
            "latent_path": str(latent_path),
            "meta": meta,
            "baseline_vae_decode_sec": baseline_sec,
            "baseline_mp4": str(baseline_mp4),
            "variants": rows,
            "summary": summarize_sample(rows),
        }
        (sample_dir / "report.json").write_text(json.dumps(sample_report, indent=2), encoding="utf-8")
        all_samples.append(sample_report)
        maybe_cleanup(args.device)

    out = {
        "latent_paths": [str(p) for p in latent_paths],
        "variants": [variant.__dict__ for variant in variants],
        "samples": all_samples,
    }
    out_path = output_root / "oracle_repair_report.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
