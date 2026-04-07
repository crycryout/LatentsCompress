#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import io
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zstandard as zstd


@dataclass(frozen=True)
class Scheme:
    name: str
    family: str
    value_codec: str
    keyframe_interval: int


SCHEMES: list[Scheme] = [
    Scheme(name="intra_fp16", family="intra", value_codec="fp16", keyframe_interval=0),
    Scheme(name="inter_fp16_k8", family="inter", value_codec="fp16", keyframe_interval=8),
    Scheme(name="intra_q8", family="intra", value_codec="qint8", keyframe_interval=0),
    Scheme(name="inter_q8_k8", family="inter", value_codec="qint8", keyframe_interval=8),
    Scheme(name="intra_q6", family="intra", value_codec="qint6", keyframe_interval=0),
    Scheme(name="inter_q6_k8", family="inter", value_codec="qint6", keyframe_interval=8),
    Scheme(name="intra_q4", family="intra", value_codec="qint4", keyframe_interval=0),
    Scheme(name="inter_q4_k8", family="inter", value_codec="qint4", keyframe_interval=8),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether lossy latent compression can preserve the final MP4 "
            "even when latent-domain fidelity drops."
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
        default=Path("/root/LatentsCompress/examples/downstream_loss_aware_wan22"),
    )
    parser.add_argument(
        "--vae-path",
        type=Path,
        default=Path("/root/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),
    )
    parser.add_argument(
        "--vae-module-path",
        type=Path,
        default=Path("/root/Wan2.2/wan/modules/vae2_2.py"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--zstd-level", type=int, default=19)
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=[scheme.name for scheme in SCHEMES],
        choices=[scheme.name for scheme in SCHEMES],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def torch_dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def load_wan_vae_class(module_path: Path):
    spec = importlib.util.spec_from_file_location("wan_vae2_2_direct", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Wan VAE from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Wan2_2_VAE


def video_to_uint8_frames(video_cthw: torch.Tensor) -> torch.Tensor:
    frames = ((video_cthw.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return frames.permute(1, 2, 3, 0).contiguous()


def decode_with_vae(vae: Any, latents_cthw: torch.Tensor, device: str) -> torch.Tensor:
    with torch.inference_mode():
        video = vae.decode([latents_cthw.to(device=device, dtype=torch.float32)])[0]
    out = video.detach().cpu()
    del video
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def ffmpeg_encode_rgb24(
    frames_thwc_u8: np.ndarray,
    out_path: Path,
    fps: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.ascontiguousarray(frames_thwc_u8, dtype=np.uint8)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "17",
        str(out_path),
    ]
    proc = subprocess.run(cmd, input=frames.tobytes(order="C"), capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))


def ffmpeg_metric(ref_path: Path, dist_path: Path, metric: str) -> dict[str, float]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(ref_path),
        "-i",
        str(dist_path),
        "-lavfi",
        metric,
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    text = proc.stderr
    if metric == "psnr":
        match = re.search(r"average:([0-9.]+)", text)
        if not match:
            raise RuntimeError(f"Could not parse PSNR from ffmpeg output:\n{text}")
        return {"mp4_psnr_db": float(match.group(1))}
    if metric == "ssim":
        match = re.search(r"All:([0-9.]+)", text)
        if not match:
            raise RuntimeError(f"Could not parse SSIM from ffmpeg output:\n{text}")
        return {"mp4_ssim": float(match.group(1))}
    raise ValueError(metric)


def mse_and_psnr(ref: torch.Tensor, dist: torch.Tensor, max_val: float = 255.0) -> tuple[float, float]:
    ref_f = ref.to(torch.float32)
    dist_f = dist.to(torch.float32)
    mse = float(torch.mean((ref_f - dist_f) ** 2).item())
    if mse == 0.0:
        return 0.0, math.inf
    psnr = 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)
    return mse, psnr


def value_codec_bits(value_codec: str) -> int:
    if value_codec == "fp16":
        return 16
    if value_codec.startswith("qint"):
        return int(value_codec[4:])
    raise ValueError(f"Unsupported value codec: {value_codec}")


def pack_lowbit(values: np.ndarray, bits: int) -> np.ndarray:
    flat = np.ascontiguousarray(values, dtype=np.uint8).reshape(-1)
    if bits == 8:
        return flat.copy()
    mask = (1 << bits) - 1
    acc = 0
    acc_bits = 0
    out = bytearray()
    for value in flat:
        acc |= (int(value) & mask) << acc_bits
        acc_bits += bits
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        out.append(acc & 0xFF)
    return np.frombuffer(bytes(out), dtype=np.uint8)


def unpack_lowbit(buf: np.ndarray, bits: int, count: int) -> np.ndarray:
    packed = np.ascontiguousarray(buf, dtype=np.uint8).reshape(-1)
    if bits == 8:
        return packed[:count].copy()
    mask = (1 << bits) - 1
    acc = 0
    acc_bits = 0
    out = np.empty((count,), dtype=np.uint8)
    out_idx = 0
    for byte in packed:
        acc |= int(byte) << acc_bits
        acc_bits += 8
        while acc_bits >= bits and out_idx < count:
            out[out_idx] = acc & mask
            acc >>= bits
            acc_bits -= bits
            out_idx += 1
        if out_idx >= count:
            break
    if out_idx != count:
        raise ValueError(f"Low-bit unpack underflow: expected {count}, got {out_idx}")
    return out


def quantize_qint(target: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qmax = float((1 << (bits - 1)) - 1)
    offset = int(qmax)
    amax = np.max(np.abs(target), axis=(1, 2), keepdims=True)
    scale = np.maximum(amax / qmax, 1e-8).astype(np.float32)
    signed = np.clip(np.rint(target / scale), -qmax, qmax).astype(np.int16)
    encoded = np.ascontiguousarray((signed + offset).astype(np.uint8))
    recon = signed.astype(np.float32) * scale
    return encoded, scale.astype(np.float16), recon.astype(np.float32)


def quantize_fp16(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quant = target.astype(np.float16)
    recon = quant.astype(np.float32)
    return quant, recon


def byteshuffle(arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr)
    itemsize = arr.dtype.itemsize
    raw = arr.view(np.uint8).reshape(-1, itemsize)
    shuffled = raw.T.copy().reshape(-1)
    return shuffled


def unbyteshuffle(buf: np.ndarray, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize
    raw = np.ascontiguousarray(buf).reshape(itemsize, -1).T.copy().reshape(-1)
    return raw.view(dtype).reshape(shape)


def keyframe_mask(num_frames: int, interval: int, intra_only: bool) -> np.ndarray:
    if intra_only:
        return np.ones((num_frames,), dtype=np.uint8)
    mask = np.zeros((num_frames,), dtype=np.uint8)
    mask[0] = 1
    if interval and interval > 0:
        mask[::interval] = 1
        mask[0] = 1
    return mask


def encode_scheme(
    latents_cthw: np.ndarray,
    scheme: Scheme,
    zstd_level: int,
) -> tuple[bytes, np.ndarray, dict[str, Any]]:
    seq = np.transpose(latents_cthw, (1, 0, 2, 3)).astype(np.float32, copy=True)
    num_frames = seq.shape[0]
    intra_only = scheme.family == "intra"
    kf_mask = keyframe_mask(num_frames, scheme.keyframe_interval, intra_only)
    value_bits = value_codec_bits(scheme.value_codec)

    if scheme.value_codec.startswith("qint"):
        payload = np.empty_like(seq, dtype=np.uint8)
        scales = np.empty((num_frames, seq.shape[1], 1, 1), dtype=np.float16)
    elif scheme.value_codec == "fp16":
        payload = np.empty_like(seq, dtype=np.float16)
        scales = None
    else:
        raise ValueError(scheme.value_codec)

    recon = np.zeros_like(seq, dtype=np.float32)
    for idx in range(num_frames):
        predictive = (not intra_only) and (kf_mask[idx] == 0)
        pred = recon[idx - 1] if predictive and idx > 0 else 0.0
        target = seq[idx] - pred
        if scheme.value_codec.startswith("qint"):
            quant, scale, rec_delta = quantize_qint(target, bits=value_bits)
            payload[idx] = quant
            scales[idx] = scale
        else:
            quant, rec_delta = quantize_fp16(target)
            payload[idx] = quant
        recon[idx] = pred + rec_delta

    raw_buffer = io.BytesIO()
    npz_kwargs: dict[str, Any] = {
        "scheme_name": np.asarray([scheme.name]),
        "family": np.asarray([scheme.family]),
        "value_codec": np.asarray([scheme.value_codec]),
        "latent_shape": np.asarray(seq.shape, dtype=np.int32),
        "keyframe_mask": kf_mask,
    }
    if scheme.value_codec.startswith("qint"):
        npz_kwargs["payload_bits"] = np.asarray([value_bits], dtype=np.uint8)
        npz_kwargs["payload_num_values"] = np.asarray([payload.size], dtype=np.int64)
        npz_kwargs["payload_shape"] = np.asarray(payload.shape, dtype=np.int32)
        npz_kwargs["payload_packed"] = pack_lowbit(payload, value_bits)
        npz_kwargs["payload_scales"] = scales
    else:
        npz_kwargs["payload_dtype"] = np.asarray([str(payload.dtype)])
        npz_kwargs["payload_shape"] = np.asarray(payload.shape, dtype=np.int32)
        npz_kwargs["payload_byteshuffled"] = byteshuffle(payload)
    np.savez(raw_buffer, **npz_kwargs)
    archive = zstd.ZstdCompressor(level=zstd_level).compress(raw_buffer.getvalue())

    meta = {
        "scheme_name": scheme.name,
        "family": scheme.family,
        "value_codec": scheme.value_codec,
        "value_bits": value_bits,
        "keyframe_interval": scheme.keyframe_interval,
        "raw_payload_bytes": int(payload.nbytes),
    }
    return archive, np.transpose(recon, (1, 0, 2, 3)).copy(), meta


def decode_archive(archive_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    dctx = zstd.ZstdDecompressor()
    payload = dctx.decompress(archive_path.read_bytes())
    with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
        scheme_name = str(npz["scheme_name"][0])
        family = str(npz["family"][0])
        value_codec = str(npz["value_codec"][0])
        latent_shape = tuple(int(v) for v in npz["latent_shape"].tolist())
        kf_mask = np.asarray(npz["keyframe_mask"], dtype=np.uint8)
        value_bits = value_codec_bits(value_codec)

        if value_codec.startswith("qint"):
            shape = tuple(int(v) for v in npz["payload_shape"].tolist())
            count = int(np.asarray(npz["payload_num_values"], dtype=np.int64)[0])
            packed = np.asarray(npz["payload_packed"], dtype=np.uint8)
            quant = unpack_lowbit(packed, bits=value_bits, count=count).reshape(shape)
            scales = np.asarray(npz["payload_scales"], dtype=np.float16).astype(np.float32)
        else:
            dtype = np.dtype(str(npz["payload_dtype"][0]))
            shape = tuple(int(v) for v in npz["payload_shape"].tolist())
            shuffled = np.asarray(npz["payload_byteshuffled"], dtype=np.uint8)
            quant = unbyteshuffle(shuffled, dtype=dtype, shape=shape).astype(np.float16)
            scales = None

    recon = np.zeros(latent_shape, dtype=np.float32)
    offset = float((1 << (value_bits - 1)) - 1) if value_codec.startswith("qint") else 0.0
    for idx in range(latent_shape[0]):
        predictive = family == "inter" and kf_mask[idx] == 0
        pred = recon[idx - 1] if predictive and idx > 0 else 0.0
        if value_codec.startswith("qint"):
            rec_delta = (quant[idx].astype(np.float32) - offset) * scales[idx]
        else:
            rec_delta = quant[idx].astype(np.float32)
        recon[idx] = pred + rec_delta

    meta = {
        "scheme_name": scheme_name,
        "family": family,
        "value_codec": value_codec,
        "value_bits": value_bits,
        "num_keyframes": int(np.count_nonzero(kf_mask)),
    }
    return np.transpose(recon, (1, 0, 2, 3)).copy(), meta


def compression_ratio(raw_bytes: int, compressed_bytes: int) -> float:
    return float(raw_bytes / compressed_bytes) if compressed_bytes else math.inf


def latent_metrics(ref: torch.Tensor, dist: torch.Tensor) -> dict[str, float]:
    diff = (dist - ref).to(torch.float32)
    abs_diff = torch.abs(diff)
    return {
        "latent_mse": float(torch.mean(diff * diff).item()),
        "latent_mae": float(torch.mean(abs_diff).item()),
        "latent_max_abs": float(torch.max(abs_diff).item()),
        "latent_abs_le_0.01": float(torch.mean((abs_diff <= 0.01).to(torch.float32)).item()),
        "latent_abs_le_0.1": float(torch.mean((abs_diff <= 0.1).to(torch.float32)).item()),
    }


def find_scheme(name: str) -> Scheme:
    for scheme in SCHEMES:
        if scheme.name == name:
            return scheme
    raise KeyError(name)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_root = args.output_dir / "archives"
    mp4_root = args.output_dir / "reconstructed_mp4"
    report_path = args.output_dir / "report.json"
    summary_path = args.output_dir / "summary.md"

    if report_path.exists() and not args.overwrite:
        print(report_path.read_text(encoding="utf-8"))
        return

    payload = torch.load(args.latent_path, map_location="cpu")
    latents = payload["latents"].to(torch.float32).cpu()
    meta = json.loads(args.meta_path.read_text(encoding="utf-8"))
    fps = int(meta["native_fps"])

    Wan2_2_VAE = load_wan_vae_class(args.vae_module_path)
    vae = Wan2_2_VAE(
        vae_pth=str(args.vae_path),
        device=args.device,
        dtype=torch_dtype_from_name(args.dtype),
    )

    baseline_video = decode_with_vae(vae, latents, args.device)
    baseline_frames = video_to_uint8_frames(baseline_video)
    baseline_mp4_path = args.output_dir / "baseline_vae.mp4"
    ffmpeg_encode_rgb24(baseline_frames.numpy(), baseline_mp4_path, fps=fps)

    report: dict[str, Any] = {
        "sample": args.latent_path.stem,
        "prompt": payload.get("prompt", ""),
        "seed": int(payload.get("seed", -1)),
        "latent_shape": list(latents.shape),
        "frame_shape_thwc": list(baseline_frames.shape),
        "source_latent_path": str(args.latent_path),
        "baseline_mp4_path": str(baseline_mp4_path),
        "sizes": {
            "latent_pt_bytes": int(args.latent_path.stat().st_size),
            "latent_raw_bytes": int(latents.numel() * latents.element_size()),
            "baseline_mp4_bytes": int(baseline_mp4_path.stat().st_size),
            "raw_rgb_bytes": int(baseline_frames.numel() * baseline_frames.element_size()),
        },
        "schemes": {},
    }

    for scheme_name in args.schemes:
        scheme = find_scheme(scheme_name)
        archive_path = archive_root / f"{scheme.name}.latz"
        recon_mp4_path = mp4_root / f"{scheme.name}.mp4"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        recon_mp4_path.parent.mkdir(parents=True, exist_ok=True)

        archive, _, encode_meta = encode_scheme(latents.numpy(), scheme, zstd_level=args.zstd_level)
        archive_path.write_bytes(archive)
        recon_np, decode_meta = decode_archive(archive_path)
        recon_latents = torch.from_numpy(recon_np).to(torch.float32)

        recon_video = decode_with_vae(vae, recon_latents, args.device)
        recon_frames = video_to_uint8_frames(recon_video)
        ffmpeg_encode_rgb24(recon_frames.numpy(), recon_mp4_path, fps=fps)

        raw_mse, raw_psnr = mse_and_psnr(baseline_frames, recon_frames)
        mp4_metrics = {}
        mp4_metrics.update(ffmpeg_metric(baseline_mp4_path, recon_mp4_path, "psnr"))
        mp4_metrics.update(ffmpeg_metric(baseline_mp4_path, recon_mp4_path, "ssim"))

        row = {
            "scheme": scheme.name,
            "family": scheme.family,
            "value_codec": scheme.value_codec,
            "keyframe_interval": scheme.keyframe_interval,
            "archive_path": str(archive_path),
            "recon_mp4_path": str(recon_mp4_path),
            "archive_bytes": int(archive_path.stat().st_size),
            "recon_mp4_bytes": int(recon_mp4_path.stat().st_size),
            "archive_vs_raw_latent_ratio": compression_ratio(report["sizes"]["latent_raw_bytes"], archive_path.stat().st_size),
            "archive_vs_pt_ratio": compression_ratio(report["sizes"]["latent_pt_bytes"], archive_path.stat().st_size),
            "archive_vs_baseline_mp4_ratio": compression_ratio(report["sizes"]["baseline_mp4_bytes"], archive_path.stat().st_size),
            "raw_frame_mse": raw_mse,
            "raw_frame_psnr_db": raw_psnr,
            **mp4_metrics,
            **latent_metrics(latents, recon_latents),
            "encode_meta": encode_meta,
            "decode_meta": decode_meta,
        }
        report["schemes"][scheme.name] = row
        print(
            f"{scheme.name}: archive={row['archive_bytes']/1_000_000:.3f}MB "
            f"mp4_psnr={row['mp4_psnr_db']:.3f} ssim={row['mp4_ssim']:.6f}",
            flush=True,
        )

    rows = list(report["schemes"].values())
    rows_sorted = sorted(
        rows,
        key=lambda x: (
            -x["archive_vs_pt_ratio"],
            -x["mp4_psnr_db"],
            -x["mp4_ssim"],
        ),
    )
    report["recommended"] = [
        row["scheme"]
        for row in sorted(
            rows,
            key=lambda x: (
                -(x["mp4_psnr_db"] >= 45.0),
                -(x["mp4_ssim"] >= 0.99),
                -x["archive_vs_pt_ratio"],
                -x["mp4_psnr_db"],
            ),
        )[:3]
    ]

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Downstream-Loss-Aware Wan2.2 Experiment",
        "",
        f"Sample: `{report['sample']}`",
        "",
        "| Scheme | Archive MB | vs latent .pt | vs baseline MP4 | Latent MAE | Raw frame PSNR | MP4 PSNR | MP4 SSIM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_sorted:
        lines.append(
            "| "
            f"{row['scheme']} | "
            f"{row['archive_bytes']/1_000_000:.3f} | "
            f"{row['archive_vs_pt_ratio']:.2f}x | "
            f"{row['archive_vs_baseline_mp4_ratio']:.2f}x | "
            f"{row['latent_mae']:.4f} | "
            f"{row['raw_frame_psnr_db']:.3f} dB | "
            f"{row['mp4_psnr_db']:.3f} dB | "
            f"{row['mp4_ssim']:.6f} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
