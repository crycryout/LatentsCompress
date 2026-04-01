from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, "/root/Wan2.2")

from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode Wan2.2 saved latents with LightTAE and compare against official VAE MP4 baselines.")
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=Path("/root/LatentsCompress/examples/vbench_codec/video_samples"),
    )
    parser.add_argument(
        "--latents-root",
        type=Path,
        default=Path("/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents"),
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
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


def sample_stems(sample_root: Path, limit: int | None) -> list[str]:
    stems = sorted(p.stem for p in (sample_root / "baseline_mp4").glob("*.mp4"))
    if limit is not None:
        stems = stems[:limit]
    return stems


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


def load_latents(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    meta = {k: v for k, v in payload.items() if k != "latents"}
    latents = payload["latents"].to(torch.float32).cpu()
    return latents, meta


def decode_official(vae: Wan2_2_VAE, latents_cthw: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
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


def decode_lighttae(lighttae, latents_cthw: torch.Tensor, device: str) -> tuple[torch.Tensor, float]:
    synchronize(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        video = lighttae.decode(latents_cthw.to(device=device, dtype=torch.float32))
    synchronize(device)
    dt = time.perf_counter() - t0
    out = video.squeeze(0).detach().cpu()
    del video
    maybe_cleanup(device)
    return out, dt


def write_reports(report_root: Path, per_sample: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = report_root / "lighttaew2_2_report.json"
    csv_path = report_root / "lighttaew2_2_report.csv"
    md_path = report_root / "lighttaew2_2_report.md"
    summary_path = report_root / "lighttaew2_2_summary.json"

    json_path.write_text(json.dumps({"summary": summary, "samples": per_sample}, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fieldnames = [
        "stem",
        "fps",
        "frame_num",
        "official_decode_sec",
        "lighttae_decode_sec",
        "speedup_vs_official",
        "raw_frame_mse",
        "raw_frame_psnr_db",
        "mp4_psnr_db",
        "mp4_ssim",
        "baseline_mp4_bytes",
        "lighttae_mp4_bytes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_sample:
            writer.writerow({k: row.get(k) for k in fieldnames})

    lines = [
        "# Wan2.2 LightTAE Sample Report",
        "",
        "This report compares `lighttaew2_2` MP4 outputs against the official `Wan2.2_VAE` MP4 baseline for the 10 curated sample prompts already committed in `examples/vbench_codec/video_samples/`.",
        "",
        "## Summary",
        "",
        f"- sample_count: `{summary['sample_count']}`",
        f"- mean_official_decode_sec: `{summary['mean_official_decode_sec']:.4f}`",
        f"- mean_lighttae_decode_sec: `{summary['mean_lighttae_decode_sec']:.4f}`",
        f"- mean_speedup_vs_official: `{summary['mean_speedup_vs_official']:.4f}x`",
        f"- mean_raw_frame_psnr_db: `{summary['mean_raw_frame_psnr_db']:.4f}`",
        f"- mean_mp4_psnr_db: `{summary['mean_mp4_psnr_db']:.4f}`",
        f"- mean_mp4_ssim: `{summary['mean_mp4_ssim']:.6f}`",
        "",
        "## Per-sample",
        "",
        "| stem | official_decode_sec | lighttae_decode_sec | speedup | raw_psnr_db | mp4_psnr_db | mp4_ssim |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in per_sample:
        lines.append(
            f"| `{row['stem']}` | `{row['official_decode_sec']:.4f}` | `{row['lighttae_decode_sec']:.4f}` | "
            f"`{row['speedup_vs_official']:.4f}x` | `{row['raw_frame_psnr_db']:.4f}` | "
            f"`{row['mp4_psnr_db']:.4f}` | `{row['mp4_ssim']:.6f}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    stems = sample_stems(args.sample_root, args.limit)
    if not stems:
        raise SystemExit(f"No sample MP4s found under {args.sample_root}")
    if not args.lighttae_path.exists():
        raise SystemExit(f"Missing LightTAE checkpoint: {args.lighttae_path}")

    lighttae_cls = load_lighttae_cls(args.lightx2v_root)
    dtype = torch_dtype(args.dtype)

    official_vae = Wan2_2_VAE(
        vae_pth=str(args.ckpt_dir / "Wan2.2_VAE.pth"),
        device=args.device,
        dtype=dtype,
    )
    lighttae = lighttae_cls(
        vae_path=str(args.lighttae_path),
        dtype=dtype,
        device=args.device,
        need_scaled=True,
    ).to(args.device)

    lighttae_root = args.sample_root / "lighttaew2_2"
    report_root = args.sample_root.parent
    per_sample: list[dict[str, Any]] = []

    for stem in stems:
        latent_path = args.latents_root / f"{stem}.pt"
        baseline_mp4 = args.sample_root / "baseline_mp4" / f"{stem}.mp4"
        lighttae_mp4 = lighttae_root / f"{stem}.mp4"
        if not latent_path.exists():
            raise FileNotFoundError(latent_path)
        if not baseline_mp4.exists():
            raise FileNotFoundError(baseline_mp4)

        latents, meta = load_latents(latent_path)

        official_video, official_sec = decode_official(official_vae, latents, args.device)
        lighttae_video, lighttae_sec = decode_lighttae(lighttae, latents, args.device)

        official_frames_u8 = to_uint8_frames(official_video)
        lighttae_frames_u8 = to_uint8_frames(lighttae_video)
        raw_mse, raw_psnr = mse_and_psnr(official_frames_u8, lighttae_frames_u8)

        if not (args.skip_existing and lighttae_mp4.exists()):
            save_mp4(lighttae_video, lighttae_mp4, fps=int(meta["fps"]))

        mp4_psnr = ffmpeg_metric(baseline_mp4, lighttae_mp4, "psnr")["mp4_psnr_db"]
        mp4_ssim = ffmpeg_metric(baseline_mp4, lighttae_mp4, "ssim")["mp4_ssim"]

        sample_report = {
            "stem": stem,
            "fps": int(meta["fps"]),
            "frame_num": int(meta["frame_num"]),
            "official_decode_sec": official_sec,
            "lighttae_decode_sec": lighttae_sec,
            "speedup_vs_official": official_sec / lighttae_sec if lighttae_sec > 0 else None,
            "raw_frame_mse": raw_mse,
            "raw_frame_psnr_db": raw_psnr,
            "mp4_psnr_db": mp4_psnr,
            "mp4_ssim": mp4_ssim,
            "baseline_mp4_path": str(baseline_mp4),
            "lighttae_mp4_path": str(lighttae_mp4),
            "baseline_mp4_bytes": baseline_mp4.stat().st_size,
            "lighttae_mp4_bytes": lighttae_mp4.stat().st_size,
        }
        per_sample.append(sample_report)

    summary = {
        "sample_count": len(per_sample),
        "mean_official_decode_sec": sum(x["official_decode_sec"] for x in per_sample) / len(per_sample),
        "mean_lighttae_decode_sec": sum(x["lighttae_decode_sec"] for x in per_sample) / len(per_sample),
        "mean_speedup_vs_official": sum(x["speedup_vs_official"] for x in per_sample if x["speedup_vs_official"] is not None) / len(per_sample),
        "mean_raw_frame_psnr_db": sum(x["raw_frame_psnr_db"] for x in per_sample) / len(per_sample),
        "mean_mp4_psnr_db": sum((x["mp4_psnr_db"] or 0.0) for x in per_sample) / len(per_sample),
        "mean_mp4_ssim": sum((x["mp4_ssim"] or 0.0) for x in per_sample) / len(per_sample),
    }
    write_reports(report_root, per_sample, summary)


if __name__ == "__main__":
    main()
