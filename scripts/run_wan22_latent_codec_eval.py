from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, "/root/Wan2.2")

from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402

from wan22_zstd_codec import SCHEMES, compression_ratio, decode_latents, encode_latents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Wan2.2 latent compression schemes.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/video_bench/latent_codec/runs/wan22_ti2v5b_vbench_16x4_seed42"),
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("/workspace/models/Wan2.2-TI2V-5B"),
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=[
            "intra_fp16_zstd",
            "inter_delta_fp16_zstd",
            "intra_q8_zstd",
            "inter_delta_q8_zstd",
        ],
        choices=sorted(SCHEMES.keys()),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--zstd-level", type=int, default=19)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def raw_rgb_bytes(video_cthw: torch.Tensor) -> int:
    frames_u8 = to_uint8_frames(video_cthw)
    return int(frames_u8.numel() * frames_u8.element_size())


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
    import subprocess

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
        text = proc.stderr
        m = re.search(r"average:([0-9.]+)", text)
        return {"mp4_psnr_db": float(m.group(1)) if m else None}
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
        text = proc.stderr
        m = re.search(r"All:([0-9.]+)", text)
        return {"mp4_ssim": float(m.group(1)) if m else None}
    raise ValueError(metric)


def decode_with_vae(vae: Wan2_2_VAE, latents_cthw: torch.Tensor, device: str) -> torch.Tensor:
    with torch.inference_mode():
        video = vae.decode([latents_cthw.to(device=device, dtype=torch.float32)])[0]
    out = video.detach().cpu()
    del video
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    return out


def save_mp4(video_cthw: torch.Tensor, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video_cthw[None], save_file=str(path), fps=fps, nrow=1)


def completed_items(input_root: Path) -> list[Path]:
    meta_dir = input_root / "final_24fps"
    return sorted(meta_dir.glob("*.json"))


def summarize_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    best_by_family: dict[str, dict[str, Any]] = {}
    for family in ("intra", "inter"):
        candidates = [r for r in results if r["family"] == family]
        if not candidates:
            continue
        scored = sorted(
            candidates,
            key=lambda x: (
                -x["mean_mp4_vs_latent_size_gap_mb"],
                -x["mean_raw_psnr_db"],
                -x["mean_mp4_psnr_db"],
            ),
        )
        best_by_family[family] = scored[0]
    return best_by_family


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    items = completed_items(args.input_root)
    if args.limit is not None:
        items = items[: args.limit]

    container_root = args.output_root / "containers"
    recon_root = args.output_root / "reconstructed"
    baseline_root = args.output_root / "baseline_mp4"
    report_root = args.output_root / "reports"
    for path in (container_root, recon_root, baseline_root, report_root):
        path.mkdir(parents=True, exist_ok=True)

    vae = Wan2_2_VAE(vae_pth=str(args.ckpt_dir / "Wan2.2_VAE.pth"), device=args.device)

    per_sample_reports: list[dict[str, Any]] = []

    for meta_path in items:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stem = meta_path.stem
        report_path = report_root / f"{stem}.json"
        if args.skip_existing and report_path.exists():
            per_sample_reports.append(json.loads(report_path.read_text(encoding="utf-8")))
            continue

        payload = load_payload(Path(meta["latents_path"]))
        orig_latents = payload["latents"].to(torch.float32).cpu()
        orig_latent_pt_bytes = Path(meta["latents_path"]).stat().st_size
        orig_latent_raw_bytes = int(orig_latents.numel() * orig_latents.element_size())
        orig_mp4_path = Path(meta["final_video_path"])
        orig_mp4_bytes = orig_mp4_path.stat().st_size

        orig_video = decode_with_vae(vae, orig_latents, args.device)
        orig_frames_u8 = to_uint8_frames(orig_video)
        baseline_mp4 = baseline_root / f"{stem}.mp4"
        save_mp4(orig_video, baseline_mp4, fps=int(meta["native_fps"]))

        item_report: dict[str, Any] = {
            "stem": stem,
            "prompt": meta["prompt"],
            "seed": meta["seed"],
            "fps": meta["native_fps"],
            "frame_num": meta["frame_num"],
            "size": meta["size"],
            "latent_shape": list(orig_latents.shape),
            "orig_latent_pt_bytes": orig_latent_pt_bytes,
            "orig_latent_raw_bytes": orig_latent_raw_bytes,
            "orig_mp4_bytes": orig_mp4_bytes,
            "orig_raw_rgb_bytes": raw_rgb_bytes(orig_video),
            "baseline_reencoded_mp4": str(baseline_mp4),
            "baseline_reencoded_mp4_bytes": baseline_mp4.stat().st_size,
            "schemes": {},
        }

        for scheme in args.schemes:
            family = "inter" if scheme.startswith("inter_") else "intra"
            container_path = container_root / scheme / f"{stem}.latz"
            recon_mp4_path = recon_root / scheme / f"{stem}.mp4"
            container_path.parent.mkdir(parents=True, exist_ok=True)
            recon_mp4_path.parent.mkdir(parents=True, exist_ok=True)

            encode_stats = encode_latents(
                orig_latents,
                container_path,
                scheme,
                zstd_level=args.zstd_level,
                extra_meta={
                    "prompt": meta["prompt"],
                    "seed": meta["seed"],
                    "fps": meta["native_fps"],
                    "frame_num": meta["frame_num"],
                    "size": meta["size"],
                },
            )
            recon_latents, header = decode_latents(container_path)
            latent_mse = torch.mean((orig_latents - recon_latents) ** 2).item()
            latent_mae = torch.mean(torch.abs(orig_latents - recon_latents)).item()
            latent_max_abs = torch.max(torch.abs(orig_latents - recon_latents)).item()

            recon_video = decode_with_vae(vae, recon_latents, args.device)
            recon_frames_u8 = to_uint8_frames(recon_video)
            raw_mse, raw_psnr = mse_and_psnr(orig_frames_u8, recon_frames_u8)
            save_mp4(recon_video, recon_mp4_path, fps=int(meta["native_fps"]))
            mp4_psnr = ffmpeg_metric(baseline_mp4, recon_mp4_path, "psnr")["mp4_psnr_db"]
            mp4_ssim = ffmpeg_metric(baseline_mp4, recon_mp4_path, "ssim")["mp4_ssim"]

            scheme_report = {
                "family": family,
                "scheme": scheme,
                "container_path": str(container_path),
                "recon_mp4_path": str(recon_mp4_path),
                "container_bytes": container_path.stat().st_size,
                "recon_mp4_bytes": recon_mp4_path.stat().st_size,
                "container_mb": round(container_path.stat().st_size / 1_000_000, 4),
                "recon_mp4_mb": round(recon_mp4_path.stat().st_size / 1_000_000, 4),
                "latent_compression_ratio_vs_raw": compression_ratio(orig_latent_raw_bytes, container_path.stat().st_size),
                "latent_compression_ratio_vs_pt": compression_ratio(orig_latent_pt_bytes, container_path.stat().st_size),
                "mp4_size_gap_mb_vs_original_mp4": round((orig_mp4_bytes - container_path.stat().st_size) / 1_000_000, 4),
                "latent_mse": latent_mse,
                "latent_mae": latent_mae,
                "latent_max_abs": latent_max_abs,
                "raw_frame_mse": raw_mse,
                "raw_frame_psnr_db": raw_psnr,
                "mp4_psnr_db": mp4_psnr,
                "mp4_ssim": mp4_ssim,
                "header": header,
                "encode_stats": encode_stats,
            }
            item_report["schemes"][scheme] = scheme_report

        report_path.write_text(json.dumps(item_report, indent=2, ensure_ascii=False), encoding="utf-8")
        per_sample_reports.append(item_report)
        print(f"done {stem}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    for scheme in args.schemes:
        scheme_reports = [r["schemes"][scheme] for r in per_sample_reports if scheme in r["schemes"]]
        if not scheme_reports:
            continue
        family = "inter" if scheme.startswith("inter_") else "intra"
        row = {
            "family": family,
            "scheme": scheme,
            "samples": len(scheme_reports),
            "mean_container_mb": sum(x["container_mb"] for x in scheme_reports) / len(scheme_reports),
            "mean_recon_mp4_mb": sum(x["recon_mp4_mb"] for x in scheme_reports) / len(scheme_reports),
            "mean_raw_psnr_db": sum(x["raw_frame_psnr_db"] for x in scheme_reports) / len(scheme_reports),
            "mean_mp4_psnr_db": sum((x["mp4_psnr_db"] or 0.0) for x in scheme_reports) / len(scheme_reports),
            "mean_mp4_ssim": sum((x["mp4_ssim"] or 0.0) for x in scheme_reports) / len(scheme_reports),
            "mean_mp4_vs_latent_size_gap_mb": sum(x["mp4_size_gap_mb_vs_original_mp4"] for x in scheme_reports) / len(scheme_reports),
        }
        summary_rows.append(row)

    best_by_family = summarize_best(summary_rows)
    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "samples": len(per_sample_reports),
        "schemes": summary_rows,
        "recommended": best_by_family,
        "notes": [
            "Quality metrics on raw decoded frames use uint8 RGB PSNR against baseline latent decode.",
            "MP4 quality metrics compare MP4 files re-encoded from baseline latent decode vs reconstructed latent decode using the same Wan save_video path.",
            "Container sizes should be compared directly against the original MP4 file sizes already produced by Wan generation.",
        ],
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
