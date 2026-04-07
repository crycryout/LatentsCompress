#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import io
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zstandard as zstd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure latent channel sensitivity using downstream reconstruction quality, "
            "then test a sensitivity-aware mixed-bit quantizer against uniform baselines."
        )
    )
    parser.add_argument(
        "--latent-path",
        type=Path,
        default=Path("/root/s3_probe/sample/latent.pt"),
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path("/root/s3_probe/sample/meta.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/LatentsCompress/examples/channel_sensitivity_wan22"),
    )
    parser.add_argument(
        "--helper-script",
        type=Path,
        default=Path("/root/LatentsCompress/scripts/experiment_wan22_downstream_loss_aware.py"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--zstd-level", type=int, default=19)
    parser.add_argument(
        "--sensitivity-family",
        choices=["intra", "inter"],
        default="inter",
        help="Family used when perturbing a single channel to measure sensitivity.",
    )
    parser.add_argument(
        "--sensitivity-bits",
        type=int,
        choices=[4, 6],
        default=4,
        help="Bit depth used for single-channel perturbation during sensitivity measurement.",
    )
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--high-channels",
        type=int,
        default=8,
        help="Number of most-sensitive channels to upgrade to 8 bits.",
    )
    parser.add_argument(
        "--low-channels",
        type=int,
        default=8,
        help="Number of least-sensitive channels to downgrade to 4 bits.",
    )
    parser.add_argument(
        "--mp4-validation-count",
        type=int,
        default=6,
        help="How many top and bottom channels to validate with final MP4 metrics.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_helper_module(path: Path):
    spec = importlib.util.spec_from_file_location("downstream_helper", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def channel_rank_from_scores(scores: list[dict[str, Any]], descending: bool = True) -> list[int]:
    ordered = sorted(scores, key=lambda row: row["score"], reverse=descending)
    return [int(row["channel"]) for row in ordered]


def seq_from_cthw(latents_cthw: torch.Tensor) -> np.ndarray:
    return np.transpose(latents_cthw.detach().cpu().numpy().astype(np.float32, copy=True), (1, 0, 2, 3))


def cthw_from_seq(seq_tchw: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.transpose(seq_tchw, (1, 0, 2, 3)).copy()).to(torch.float32)


def quantize_one_channel(
    seq_tchw: np.ndarray,
    channel_index: int,
    family: str,
    bits: int,
    keyframe_interval: int,
    helper: Any,
) -> np.ndarray:
    out = seq_tchw.copy()
    num_frames = seq_tchw.shape[0]
    kf_mask = helper.keyframe_mask(num_frames, keyframe_interval, family == "intra")
    recon_channel = np.zeros((num_frames, seq_tchw.shape[2], seq_tchw.shape[3]), dtype=np.float32)
    for t in range(num_frames):
        predictive = family == "inter" and kf_mask[t] == 0
        pred = recon_channel[t - 1] if predictive and t > 0 else 0.0
        target = seq_tchw[t, channel_index] - pred
        quant, _scale, rec_delta = helper.quantize_qint(target[None, ...], bits=bits)
        recon_channel[t] = pred + rec_delta[0]
        out[t, channel_index] = recon_channel[t]
    return out


def make_uniform_scheme(seq_tchw: np.ndarray, family: str, bits: int, keyframe_interval: int, helper: Any) -> tuple[np.ndarray, bytes]:
    codec_name = f"qint{bits}" if bits != 16 else "fp16"
    scheme = helper.Scheme(
        name=f"{family}_{codec_name}_k{keyframe_interval if family == 'inter' else 0}",
        family=family,
        value_codec=codec_name,
        keyframe_interval=keyframe_interval if family == "inter" else 0,
    )
    archive, recon_cthw, _meta = helper.encode_scheme(np.transpose(seq_tchw, (1, 0, 2, 3)), scheme, zstd_level=19)
    return np.transpose(recon_cthw, (1, 0, 2, 3)), archive


def build_mixed_bit_allocation(
    num_channels: int,
    ranked_channels: list[int],
    high_channels: int,
    low_channels: int,
) -> list[int]:
    bits = [6] * num_channels
    for channel in ranked_channels[:high_channels]:
        bits[channel] = 8
    if low_channels > 0:
        for channel in ranked_channels[-low_channels:]:
            bits[channel] = 4
    return bits


def encode_mixed_bits(
    seq_tchw: np.ndarray,
    channel_bits: list[int],
    family: str,
    keyframe_interval: int,
    helper: Any,
    zstd_level: int,
) -> tuple[np.ndarray, bytes, dict[str, Any]]:
    num_frames, num_channels, height, width = seq_tchw.shape
    kf_mask = helper.keyframe_mask(num_frames, keyframe_interval, family == "intra")
    recon = np.zeros_like(seq_tchw, dtype=np.float32)
    scales = np.zeros((num_frames, num_channels), dtype=np.float16)
    offsets: list[int] = []
    bit_depths: list[int] = []
    packed_parts: list[np.ndarray] = []

    for channel_index, bits in enumerate(channel_bits):
        offset = 0 if not packed_parts else int(offsets[-1] + packed_parts[-1].size)
        offsets.append(offset)
        bit_depths.append(bits)
        qmax = float((1 << (bits - 1)) - 1)
        qoffset = int(qmax)
        quant_values = np.empty((num_frames, height, width), dtype=np.uint8)
        recon_channel = np.zeros((num_frames, height, width), dtype=np.float32)
        for t in range(num_frames):
            predictive = family == "inter" and kf_mask[t] == 0
            pred = recon_channel[t - 1] if predictive and t > 0 else 0.0
            target = seq_tchw[t, channel_index] - pred
            amax = np.max(np.abs(target))
            scale = max(float(amax / qmax), 1e-8)
            signed = np.clip(np.rint(target / scale), -qmax, qmax).astype(np.int16)
            quant_values[t] = np.ascontiguousarray((signed + qoffset).astype(np.uint8))
            rec_delta = signed.astype(np.float32) * scale
            recon_channel[t] = pred + rec_delta
            scales[t, channel_index] = np.float16(scale)
        recon[:, channel_index] = recon_channel
        packed_parts.append(helper.pack_lowbit(quant_values.reshape(-1), bits))

    packed_concat = np.concatenate(packed_parts) if packed_parts else np.empty((0,), dtype=np.uint8)
    payload_buffer = io.BytesIO()
    np.savez(
        payload_buffer,
        scheme_name=np.asarray(["mixed_864"], dtype="<U9"),
        family=np.asarray([family], dtype="<U5"),
        latent_shape=np.asarray(seq_tchw.shape, dtype=np.int32),
        keyframe_mask=kf_mask,
        channel_bits=np.asarray(channel_bits, dtype=np.uint8),
        channel_offsets=np.asarray(offsets, dtype=np.int64),
        packed_payload=packed_concat,
        scales=scales,
    )
    archive = zstd.ZstdCompressor(level=zstd_level).compress(payload_buffer.getvalue())
    meta = {
        "scheme": "mixed_864",
        "family": family,
        "keyframe_interval": keyframe_interval,
        "channel_bits": channel_bits,
        "raw_payload_bytes": int(sum(part.size for part in packed_parts)),
    }
    return recon, archive, meta


def evaluate_mp4(
    helper: Any,
    baseline_frames: torch.Tensor,
    baseline_mp4_path: Path,
    recon_video: torch.Tensor,
    recon_mp4_path: Path,
    fps: int,
) -> dict[str, float]:
    recon_frames = helper.video_to_uint8_frames(recon_video)
    helper.ffmpeg_encode_rgb24(recon_frames.numpy(), recon_mp4_path, fps=fps)
    raw_mse, raw_psnr = helper.mse_and_psnr(baseline_frames, recon_frames)
    metrics = {
        "raw_frame_mse": raw_mse,
        "raw_frame_psnr_db": raw_psnr,
    }
    metrics.update(helper.ffmpeg_metric(baseline_mp4_path, recon_mp4_path, "psnr"))
    metrics.update(helper.ffmpeg_metric(baseline_mp4_path, recon_mp4_path, "ssim"))
    return metrics


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    summary_path = args.output_dir / "summary.md"
    channel_json_path = args.output_dir / "channel_sensitivity.json"

    if report_path.exists() and not args.overwrite:
        print(report_path.read_text(encoding="utf-8"))
        return

    helper = load_helper_module(args.helper_script)
    payload = torch.load(args.latent_path, map_location="cpu")
    latents = payload["latents"].to(torch.float32).cpu()
    meta = json.loads(args.meta_path.read_text(encoding="utf-8"))
    fps = int(meta["native_fps"])
    num_channels = int(latents.shape[0])

    Wan2_2_VAE = helper.load_wan_vae_class(helper.Path("/root/Wan2.2/wan/modules/vae2_2.py"))
    vae = Wan2_2_VAE(
        vae_pth=str(helper.Path("/root/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")),
        device=args.device,
        dtype=helper.torch_dtype_from_name(args.dtype),
    )

    baseline_video = helper.decode_with_vae(vae, latents, args.device)
    baseline_frames = helper.video_to_uint8_frames(baseline_video)
    baseline_mp4_path = args.output_dir / "baseline_vae.mp4"
    helper.ffmpeg_encode_rgb24(baseline_frames.numpy(), baseline_mp4_path, fps=fps)

    seq_tchw = seq_from_cthw(latents)
    sensitivity_rows: list[dict[str, Any]] = []

    for channel_index in range(num_channels):
        perturbed_seq = quantize_one_channel(
            seq_tchw,
            channel_index=channel_index,
            family=args.sensitivity_family,
            bits=args.sensitivity_bits,
            keyframe_interval=args.keyframe_interval,
            helper=helper,
        )
        perturbed_latents = cthw_from_seq(perturbed_seq)
        perturbed_video = helper.decode_with_vae(vae, perturbed_latents, args.device)
        perturbed_frames = helper.video_to_uint8_frames(perturbed_video)
        raw_mse, raw_psnr = helper.mse_and_psnr(baseline_frames, perturbed_frames)
        sensitivity_rows.append(
            {
                "channel": channel_index,
                "raw_frame_mse": raw_mse,
                "raw_frame_psnr_db": raw_psnr,
                "score": raw_mse,
            }
        )
        print(f"sensitivity channel={channel_index:02d} raw_psnr={raw_psnr:.3f}dB", flush=True)

    ranked_channels = channel_rank_from_scores(sensitivity_rows, descending=True)
    top_channels = ranked_channels[: args.mp4_validation_count]
    bottom_channels = ranked_channels[-args.mp4_validation_count :]
    validation_channels = top_channels + [c for c in bottom_channels if c not in top_channels]

    validation_rows: list[dict[str, Any]] = []
    for channel_index in validation_channels:
        perturbed_seq = quantize_one_channel(
            seq_tchw,
            channel_index=channel_index,
            family=args.sensitivity_family,
            bits=args.sensitivity_bits,
            keyframe_interval=args.keyframe_interval,
            helper=helper,
        )
        perturbed_latents = cthw_from_seq(perturbed_seq)
        perturbed_video = helper.decode_with_vae(vae, perturbed_latents, args.device)
        recon_mp4_path = args.output_dir / "mp4_channel_validation" / f"channel_{channel_index:02d}.mp4"
        metrics = evaluate_mp4(
            helper,
            baseline_frames=baseline_frames,
            baseline_mp4_path=baseline_mp4_path,
            recon_video=perturbed_video,
            recon_mp4_path=recon_mp4_path,
            fps=fps,
        )
        validation_rows.append(
            {
                "channel": channel_index,
                **metrics,
            }
        )
        print(
            f"mp4_validate channel={channel_index:02d} "
            f"mp4_psnr={metrics['mp4_psnr_db']:.3f} ssim={metrics['mp4_ssim']:.6f}",
            flush=True,
        )

    channel_bits = build_mixed_bit_allocation(
        num_channels=num_channels,
        ranked_channels=ranked_channels,
        high_channels=args.high_channels,
        low_channels=args.low_channels,
    )

    candidate_rows: list[dict[str, Any]] = []
    scheme_specs = [
        ("uniform_inter_q8", "uniform", {"family": "inter", "bits": 8}),
        ("uniform_inter_q6", "uniform", {"family": "inter", "bits": 6}),
        ("weighted_inter_mixed_864", "mixed", {"family": "inter"}),
    ]

    for scheme_name, kind, params in scheme_specs:
        if kind == "uniform":
            recon_seq, archive = make_uniform_scheme(
                seq_tchw=seq_tchw,
                family=params["family"],
                bits=params["bits"],
                keyframe_interval=args.keyframe_interval,
                helper=helper,
            )
            meta_row = {
                "kind": "uniform",
                "family": params["family"],
                "bits": params["bits"],
            }
        else:
            recon_seq, archive, mixed_meta = encode_mixed_bits(
                seq_tchw=seq_tchw,
                channel_bits=channel_bits,
                family=params["family"],
                keyframe_interval=args.keyframe_interval,
                helper=helper,
                zstd_level=args.zstd_level,
            )
            meta_row = {
                "kind": "mixed",
                **mixed_meta,
            }
        recon_latents = cthw_from_seq(recon_seq)
        recon_video = helper.decode_with_vae(vae, recon_latents, args.device)
        recon_mp4_path = args.output_dir / "scheme_mp4" / f"{scheme_name}.mp4"
        metrics = evaluate_mp4(
            helper,
            baseline_frames=baseline_frames,
            baseline_mp4_path=baseline_mp4_path,
            recon_video=recon_video,
            recon_mp4_path=recon_mp4_path,
            fps=fps,
        )
        latent_stats = helper.latent_metrics(latents, recon_latents)
        candidate_row = {
            "scheme": scheme_name,
            "archive_bytes": len(archive),
            "archive_mb": round(len(archive) / 1_000_000, 6),
            "archive_vs_pt_ratio": helper.compression_ratio(int(args.latent_path.stat().st_size), len(archive)),
            "archive_vs_raw_latent_ratio": helper.compression_ratio(int(latents.numel() * latents.element_size()), len(archive)),
            **metrics,
            **latent_stats,
            **meta_row,
        }
        candidate_rows.append(candidate_row)
        print(
            f"scheme={scheme_name} archive={candidate_row['archive_mb']:.3f}MB "
            f"mp4_psnr={candidate_row['mp4_psnr_db']:.3f} ssim={candidate_row['mp4_ssim']:.6f}",
            flush=True,
        )

    report = {
        "sample": args.latent_path.stem,
        "prompt": payload.get("prompt", ""),
        "seed": int(payload.get("seed", -1)),
        "latent_shape": list(latents.shape),
        "sensitivity_setup": {
            "family": args.sensitivity_family,
            "bits": args.sensitivity_bits,
            "keyframe_interval": args.keyframe_interval,
            "high_channels": args.high_channels,
            "low_channels": args.low_channels,
        },
        "ranked_channels": ranked_channels,
        "channel_bits": channel_bits,
        "channel_sensitivity": sensitivity_rows,
        "channel_mp4_validation": validation_rows,
        "scheme_results": candidate_rows,
    }

    channel_json_path.write_text(json.dumps(sensitivity_rows, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Channel Sensitivity Experiment",
        "",
        f"Sample: `{args.latent_path.stem}`",
        "",
        "## Scheme Comparison",
        "",
        "| Scheme | Archive MB | vs `.pt` | Latent MAE | Raw Frame PSNR | MP4 PSNR | MP4 SSIM |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(candidate_rows, key=lambda item: (-item["mp4_psnr_db"], item["archive_bytes"])):
        lines.append(
            "| "
            f"{row['scheme']} | "
            f"{row['archive_mb']:.3f} | "
            f"{row['archive_vs_pt_ratio']:.2f}x | "
            f"{row['latent_mae']:.4f} | "
            f"{row['raw_frame_psnr_db']:.3f} dB | "
            f"{row['mp4_psnr_db']:.3f} dB | "
            f"{row['mp4_ssim']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Most Sensitive Channels",
            "",
            "| Rank | Channel | Raw Frame PSNR After Single-Channel Q4 |",
            "|---|---:|---:|",
        ]
    )
    ordered_rows = sorted(sensitivity_rows, key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(ordered_rows[:10], start=1):
        lines.append(f"| {rank} | {row['channel']} | {row['raw_frame_psnr_db']:.3f} dB |")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
