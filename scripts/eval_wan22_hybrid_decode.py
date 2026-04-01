#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import re
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/Wan2.2")

from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hybrid Wan2.2 decode with sparse VAE anchors and TAE fast-path decode.")
    parser.add_argument(
        "--latent-path",
        type=Path,
        default=Path("/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/000_subject_consistency_r00_p003_a_person_eating_a_burger.pt"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/LatentsCompress/examples/hybrid_decode/wan22_subject_consistency"),
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
    parser.add_argument("--anchor-interval-steps", type=int, default=8, help="Anchor spacing over 4-frame latent groups starting from step 1.")
    parser.add_argument("--anchor-context-steps", type=int, default=3, help="How many latent steps to decode in each local VAE anchor window.")
    parser.add_argument("--lowpass-kernel", type=int, default=17)
    parser.add_argument("--include-noncausal-interp", action="store_true", help="Also evaluate the non-causal interpolated residual mode as an oracle upper bound.")
    parser.add_argument("--save-videos", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def maybe_cleanup(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_saved_latents(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict):
        meta = {k: v for k, v in blob.items() if k != "latents"}
        latents = blob["latents"]
    else:
        meta = {}
        latents = blob
    return latents.to(torch.float32).cpu(), meta


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


def save_mp4(video_cthw: torch.Tensor, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video_cthw[None], save_file=str(path), fps=fps, nrow=1)


def load_lighttae_cls(lightx2v_root: Path):
    for name in [
        "lightx2v",
        "lightx2v.models",
        "lightx2v.models.video_encoders",
        "lightx2v.models.video_encoders.hf",
        "lightx2v.models.video_encoders.hf.wan",
    ]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

    def load_module(mod_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    load_module(
        "lightx2v.models.video_encoders.hf.tae",
        lightx2v_root / "lightx2v/models/video_encoders/hf/tae.py",
    )
    vae_tiny = load_module(
        "lightx2v.models.video_encoders.hf.wan.vae_tiny",
        lightx2v_root / "lightx2v/models/video_encoders/hf/wan/vae_tiny.py",
    )
    return vae_tiny.Wan2_2_VAE_tiny


@dataclass
class FrameGroup:
    latent_step: int
    frame_start: int
    frame_end: int

    @property
    def frame_count(self) -> int:
        return self.frame_end - self.frame_start


def build_frame_groups(num_steps: int) -> list[FrameGroup]:
    groups: list[FrameGroup] = []
    frame_cursor = 0
    for step in range(num_steps):
        frame_count = 1 if step == 0 else 4
        groups.append(FrameGroup(step, frame_cursor, frame_cursor + frame_count))
        frame_cursor += frame_count
    return groups


def keyframe_steps(num_steps: int, interval_steps: int) -> list[int]:
    steps = [0, 1]
    step = 1 + interval_steps
    while step < num_steps:
        steps.append(step)
        step += interval_steps
    if steps[-1] != num_steps - 1:
        steps.append(num_steps - 1)
    return sorted(set(steps))


def avg_blur(frames_tchw: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return frames_tchw
    pad = kernel // 2
    return F.avg_pool2d(frames_tchw, kernel_size=kernel, stride=1, padding=pad)


def decode_official(vae: Wan2_2_VAE, latents_cthw: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
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


def decode_lighttae(lighttae, latents_cthw: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
    synchronize(device)
    import time

    t0 = time.perf_counter()
    with torch.inference_mode():
        video = lighttae.decode(latents_cthw.to(device=device, dtype=torch.float32))
    synchronize(device)
    dt = time.perf_counter() - t0
    out = video.squeeze(0).detach().cpu()
    del video
    maybe_cleanup(device)
    return out, dt


def decode_anchor_window(
    vae: Wan2_2_VAE,
    latents: torch.Tensor,
    group: FrameGroup,
    context_steps: int,
    device: str,
) -> tuple[torch.Tensor, float, int]:
    start_step = max(0, group.latent_step - context_steps + 1)
    window = latents[:, start_step:group.latent_step + 1].contiguous()
    decoded, dt = decode_official(vae, window, device)
    keep_frames = group.frame_count
    anchor = decoded[:, -keep_frames:].contiguous()
    return anchor, dt, int(window.shape[1])


def write_video_metrics(
    name: str,
    video_cthw: torch.Tensor,
    baseline_video_cthw: torch.Tensor,
    baseline_mp4: Path,
    video_dir: Path,
    fps: int,
) -> dict[str, Any]:
    frames_u8 = to_uint8_frames(video_cthw)
    ref_u8 = to_uint8_frames(baseline_video_cthw)
    raw_mse, raw_psnr = mse_and_psnr(ref_u8, frames_u8)
    mp4_path = video_dir / f"{name}.mp4"
    save_mp4(video_cthw, mp4_path, fps=fps)
    mp4_psnr = ffmpeg_metric(baseline_mp4, mp4_path, "psnr")["mp4_psnr_db"]
    mp4_ssim = ffmpeg_metric(baseline_mp4, mp4_path, "ssim")["mp4_ssim"]
    return {
        "name": name,
        "mp4_path": str(mp4_path),
        "raw_frame_mse": raw_mse,
        "raw_frame_psnr_db": raw_psnr,
        "mp4_psnr_db": mp4_psnr,
        "mp4_ssim": mp4_ssim,
        "mp4_bytes": mp4_path.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    video_dir = args.output_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    latents, meta = load_saved_latents(args.latent_path)
    num_steps = int(latents.shape[1])
    groups = build_frame_groups(num_steps)
    anchors = keyframe_steps(num_steps, args.anchor_interval_steps)

    dtype = torch_dtype(args.dtype)
    official_vae = Wan2_2_VAE(
        vae_pth=str(args.ckpt_dir / "Wan2.2_VAE.pth"),
        device=args.device,
        dtype=dtype,
    )
    lighttae_cls = load_lighttae_cls(args.lightx2v_root)
    lighttae = lighttae_cls(
        vae_path=str(args.lighttae_path),
        device=args.device,
        dtype=dtype,
        need_scaled=True,
    ).to(args.device)
    if hasattr(lighttae, "taehv"):
        lighttae.taehv = lighttae.taehv.to(device=args.device, dtype=dtype).eval()

    baseline_video, baseline_sec = decode_official(official_vae, latents, args.device)
    tae_video, tae_sec = decode_lighttae(lighttae, latents, args.device)
    fps = int(meta.get("fps", 24))
    baseline_mp4 = video_dir / "vae_baseline.mp4"
    save_mp4(baseline_video, baseline_mp4, fps=fps)

    anchor_frames: dict[int, torch.Tensor] = {}
    anchor_meta: list[dict[str, Any]] = []
    for step in anchors:
        group = groups[step]
        anchor, dt, window_steps = decode_anchor_window(
            official_vae,
            latents,
            group,
            context_steps=args.anchor_context_steps,
            device=args.device,
        )
        anchor_frames[step] = anchor
        ref = baseline_video[:, group.frame_start:group.frame_end]
        tae_ref = tae_video[:, group.frame_start:group.frame_end]
        anchor_u8 = to_uint8_frames(anchor)
        ref_u8 = to_uint8_frames(ref)
        tae_ref_u8 = to_uint8_frames(tae_ref)
        anchor_mse, anchor_psnr = mse_and_psnr(ref_u8, anchor_u8)
        tae_anchor_mse, tae_anchor_psnr = mse_and_psnr(ref_u8, tae_ref_u8)
        anchor_meta.append(
            {
                "latent_step": step,
                "frame_start": group.frame_start,
                "frame_end": group.frame_end,
                "frame_count": group.frame_count,
                "window_steps": window_steps,
                "decode_sec": dt,
                "anchor_psnr_db_vs_full_vae": anchor_psnr,
                "tae_psnr_db_vs_full_vae": tae_anchor_psnr,
            }
        )

    tae_only = tae_video.clone()
    key_replace = tae_video.clone()
    prev_lpf = tae_video.clone()
    interp_lpf = tae_video.clone()

    slot_residuals: dict[int, torch.Tensor] = {}
    for step in anchors:
        group = groups[step]
        anchor = anchor_frames[step]
        tae_group = tae_video[:, group.frame_start:group.frame_end]
        key_replace[:, group.frame_start:group.frame_end] = anchor
        prev_lpf[:, group.frame_start:group.frame_end] = anchor
        interp_lpf[:, group.frame_start:group.frame_end] = anchor
        residual = avg_blur((anchor - tae_group).permute(1, 0, 2, 3), args.lowpass_kernel).permute(1, 0, 2, 3)
        slot_residuals[step] = residual

    anchor_steps_sorted = sorted([step for step in anchors if step >= 1])
    if 1 not in anchor_steps_sorted:
        anchor_steps_sorted = [1] + anchor_steps_sorted

    for idx, prev_step in enumerate(anchor_steps_sorted):
        next_step = anchor_steps_sorted[idx + 1] if idx + 1 < len(anchor_steps_sorted) else None
        prev_res = slot_residuals[prev_step]
        next_res = slot_residuals[next_step] if next_step is not None else prev_res

        start_step = prev_step + 1
        end_step = next_step if next_step is not None else num_steps
        for step in range(start_step, end_step):
            group = groups[step]
            tae_group = tae_video[:, group.frame_start:group.frame_end]

            prev_lpf[:, group.frame_start:group.frame_end] = torch.clamp(
                tae_group + prev_res,
                -1,
                1,
            )

            if next_step is not None:
                alpha = (step - prev_step) / float(next_step - prev_step)
                blended_res = (1.0 - alpha) * prev_res + alpha * next_res
            else:
                blended_res = prev_res
            interp_lpf[:, group.frame_start:group.frame_end] = torch.clamp(
                tae_group + blended_res,
                -1,
                1,
            )

    methods = {
        "tae_only": tae_only,
        "hybrid_key_replace": key_replace,
        "hybrid_prev_lpf": prev_lpf,
    }
    if args.include_noncausal_interp:
        methods["hybrid_interp_lpf"] = interp_lpf

    metrics = []
    for name, video in methods.items():
        metrics.append(
            write_video_metrics(
                name=name,
                video_cthw=video,
                baseline_video_cthw=baseline_video,
                baseline_mp4=baseline_mp4,
                video_dir=video_dir,
                fps=fps,
            )
        )

    total_anchor_sec = sum(item["decode_sec"] for item in anchor_meta)
    effective_total_sec = tae_sec + total_anchor_sec
    total_frames = int(tae_video.shape[1])
    group_budget_sec = 4.0 / fps
    mean_anchor_sec = total_anchor_sec / max(len(anchor_meta), 1)
    amortized_group_sec = (tae_sec / max(num_steps - 1, 1)) + (mean_anchor_sec / max(args.anchor_interval_steps, 1))

    report = {
        "latent_path": str(args.latent_path),
        "meta": meta,
        "latent_shape": list(latents.shape),
        "anchor_interval_steps": args.anchor_interval_steps,
        "anchor_context_steps": args.anchor_context_steps,
        "lowpass_kernel": args.lowpass_kernel,
        "anchor_steps": anchors,
        "baseline_vae_decode_sec": baseline_sec,
        "tae_decode_sec": tae_sec,
        "total_anchor_decode_sec": total_anchor_sec,
        "effective_total_decode_sec": effective_total_sec,
        "effective_total_fps": (total_frames / effective_total_sec) if effective_total_sec > 0 else None,
        "amortized_group_sec_estimate": amortized_group_sec,
        "amortized_group_budget_sec": group_budget_sec,
        "amortized_group_realtime_feasible": amortized_group_sec <= group_budget_sec,
        "anchors": anchor_meta,
        "metrics": metrics,
    }

    (args.output_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Wan2.2 Hybrid Decode Report",
        "",
        f"- latent: `{args.latent_path}`",
        f"- latent_shape: `{list(latents.shape)}`",
        f"- anchor_interval_steps: `{args.anchor_interval_steps}`",
        f"- anchor_context_steps: `{args.anchor_context_steps}`",
        f"- anchor_steps: `{anchors}`",
        f"- baseline_vae_decode_sec: `{baseline_sec:.4f}`",
        f"- tae_decode_sec: `{tae_sec:.4f}`",
        f"- total_anchor_decode_sec: `{total_anchor_sec:.4f}`",
        f"- effective_total_decode_sec: `{effective_total_sec:.4f}`",
        f"- effective_total_fps: `{report['effective_total_fps']:.4f}`",
        f"- amortized_group_sec_estimate: `{amortized_group_sec:.6f}`",
        f"- amortized_group_budget_sec: `{group_budget_sec:.6f}`",
        f"- amortized_group_realtime_feasible: `{report['amortized_group_realtime_feasible']}`",
        "",
        "## Anchor Fidelity",
        "",
        "| latent_step | frames | window_steps | decode_sec | anchor_psnr_db_vs_full_vae | tae_psnr_db_vs_full_vae |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in anchor_meta:
        lines.append(
            f"| `{item['latent_step']}` | `{item['frame_count']}` | `{item['window_steps']}` | `{item['decode_sec']:.4f}` | "
            f"`{item['anchor_psnr_db_vs_full_vae']:.4f}` | `{item['tae_psnr_db_vs_full_vae']:.4f}` |"
        )
    lines.extend(
        [
            "",
            "## Method Metrics",
            "",
            "| method | raw_psnr_db | mp4_psnr_db | mp4_ssim | mp4_bytes |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in metrics:
        lines.append(
            f"| `{item['name']}` | `{item['raw_frame_psnr_db']:.4f}` | `{item['mp4_psnr_db']:.4f}` | "
            f"`{item['mp4_ssim']:.6f}` | `{item['mp4_bytes']}` |"
        )
    (args.output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_root / "report.json")


if __name__ == "__main__":
    main()
