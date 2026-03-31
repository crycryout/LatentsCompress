#!/usr/bin/env python3
import argparse
import gc
import io
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import zstandard as zstd

sys.path.insert(0, "/root/Wan2.2")

from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress Wan2.2 TI2V latents, decode them through the Wan VAE, and benchmark size and quality."
    )
    parser.add_argument(
        "--input-root",
        default="/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42",
        help="Root directory containing latents/ and native_16fps/.",
    )
    parser.add_argument(
        "--output-root",
        default="/workspace/video_bench/latent_codec/results",
        help="Directory for codec files, reconstructed MP4s, and reports.",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="/workspace/models/Wan2.2-TI2V-5B",
        help="Wan2.2-TI2V-5B checkpoint directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device used for VAE decode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of matched samples to process.",
    )
    parser.add_argument(
        "--zstd-level",
        type=int,
        default=19,
        help="Zstandard compression level.",
    )
    parser.add_argument(
        "--value-codec",
        choices=["qint8", "qint6", "qint4", "fp16"],
        default="qint8",
        help="Value transform before zstd. qint* uses symmetric per-frame/per-channel quantization.",
    )
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=8,
        help="Latent-frame keyframe interval for temporal predictive coding. 0 means only the first latent frame is a keyframe.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute existing outputs.",
    )
    return parser.parse_args()


@dataclass
class Sample:
    stem: str
    latent_path: Path
    native_mp4_path: Path
    fps: int
    prompt: str


def byteshuffle(arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr)
    itemsize = arr.dtype.itemsize
    raw = arr.view(np.uint8).reshape(-1, itemsize)
    shuffled = raw.T.copy().reshape(-1)
    return shuffled


def unbyteshuffle(buf: np.ndarray, dtype: np.dtype, shape: Tuple[int, ...]) -> np.ndarray:
    dtype = np.dtype(dtype)
    itemsize = dtype.itemsize
    raw = np.ascontiguousarray(buf).reshape(itemsize, -1).T.copy().reshape(-1)
    return raw.view(dtype).reshape(shape)


def value_codec_bits(value_codec: str) -> int:
    if value_codec == "fp16":
        return 16
    if value_codec.startswith("qint"):
        return int(value_codec[4:])
    raise ValueError(f"Unsupported value codec: {value_codec}")


def keyframe_mask(num_frames: int, interval: int, intra_only: bool) -> np.ndarray:
    if intra_only:
        return np.ones((num_frames,), dtype=np.uint8)
    mask = np.zeros((num_frames,), dtype=np.uint8)
    mask[0] = 1
    if interval and interval > 0:
        mask[::interval] = 1
        mask[0] = 1
    return mask


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


def quantize_qint(target: np.ndarray, bits: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    qmax = float((1 << (bits - 1)) - 1)
    offset = int(qmax)
    amax = np.max(np.abs(target), axis=(1, 2), keepdims=True)
    scale = np.maximum(amax / qmax, 1e-8).astype(np.float32)
    signed = np.clip(np.rint(target / scale), -qmax, qmax).astype(np.int16)
    encoded = np.ascontiguousarray((signed + offset).astype(np.uint8))
    recon = signed.astype(np.float32) * scale
    return encoded, scale.astype(np.float16), recon.astype(np.float32)


def quantize_fp16(target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    quant = target.astype(np.float16)
    recon = quant.astype(np.float32)
    return quant, recon


def build_codec_archive(
    latents_cthw: np.ndarray,
    scheme_name: str,
    intra_only: bool,
    value_codec: str,
    zstd_level: int,
    key_interval: int,
) -> Tuple[bytes, np.ndarray, Dict[str, object]]:
    seq = np.transpose(latents_cthw, (1, 0, 2, 3)).astype(np.float32, copy=True)
    num_frames = seq.shape[0]
    kf_mask = keyframe_mask(num_frames, key_interval, intra_only)
    value_bits = value_codec_bits(value_codec)

    if value_codec.startswith("qint"):
        payload = np.empty_like(seq, dtype=np.uint8)
        scales = np.empty((num_frames, seq.shape[1], 1, 1), dtype=np.float16)
    elif value_codec == "fp16":
        payload = np.empty_like(seq, dtype=np.float16)
        scales = None
    else:
        raise ValueError(f"Unsupported value codec: {value_codec}")

    recon = np.zeros_like(seq, dtype=np.float32)
    for idx in range(num_frames):
        predictive = (not intra_only) and (kf_mask[idx] == 0)
        pred = recon[idx - 1] if predictive and idx > 0 else 0.0
        target = seq[idx] - pred
        if value_codec.startswith("qint"):
            quant, scale, rec_delta = quantize_qint(target, bits=value_bits)
            payload[idx] = quant
            scales[idx] = scale
        else:
            quant, rec_delta = quantize_fp16(target)
            payload[idx] = quant
        recon[idx] = pred + rec_delta

    compressor = zstd.ZstdCompressor(level=zstd_level)
    raw_buffer = io.BytesIO()
    npz_kwargs = {
        "version": np.asarray([2], dtype=np.int32),
        "scheme_name": np.asarray([scheme_name]),
        "value_codec": np.asarray([value_codec]),
        "latent_shape": np.asarray(seq.shape, dtype=np.int32),
        "keyframe_mask": kf_mask,
    }
    if value_codec.startswith("qint"):
        npz_kwargs["payload_bits"] = np.asarray([value_bits], dtype=np.uint8)
        npz_kwargs["payload_num_values"] = np.asarray([payload.size], dtype=np.int64)
        npz_kwargs["payload_shape"] = np.asarray(payload.shape, dtype=np.int32)
        npz_kwargs["payload_packed"] = pack_lowbit(payload, value_bits)
        npz_kwargs["payload_scales"] = scales
    else:
        npz_kwargs["payload_byteshuffled"] = byteshuffle(payload)
        npz_kwargs["payload_dtype"] = np.asarray([str(payload.dtype)])
        npz_kwargs["payload_shape"] = np.asarray(payload.shape, dtype=np.int32)
    np.savez(raw_buffer, **npz_kwargs)
    archive = compressor.compress(raw_buffer.getvalue())

    meta = {
        "scheme_name": scheme_name,
        "value_codec": value_codec,
        "value_bits": value_bits,
        "zstd_level": zstd_level,
        "intra_only": intra_only,
        "keyframe_interval": key_interval,
        "latent_shape_cthw": list(latents_cthw.shape),
        "latent_shape_tchw": list(seq.shape),
        "archive_bytes": len(archive),
    }
    return archive, np.transpose(recon, (1, 0, 2, 3)).copy(), meta


def decode_codec_archive(archive_path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    dctx = zstd.ZstdDecompressor()
    payload = dctx.decompress(archive_path.read_bytes())
    with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
        scheme_name = str(npz["scheme_name"][0])
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
        elif value_codec == "fp16":
            dtype = np.dtype(str(npz["payload_dtype"][0]))
            shape = tuple(int(v) for v in npz["payload_shape"].tolist())
            shuffled = np.asarray(npz["payload_byteshuffled"], dtype=np.uint8)
            quant = unbyteshuffle(shuffled, dtype=dtype, shape=shape).astype(np.float16)
            scales = None
        else:
            raise ValueError(f"Unsupported value codec in archive: {value_codec}")

    recon = np.zeros(latent_shape, dtype=np.float32)
    offset = float((1 << (value_bits - 1)) - 1) if value_codec.startswith("qint") else 0.0
    for idx in range(latent_shape[0]):
        predictive = kf_mask[idx] == 0
        pred = recon[idx - 1] if predictive and idx > 0 else 0.0
        if value_codec.startswith("qint"):
            rec_delta = (quant[idx].astype(np.float32) - offset) * scales[idx]
        else:
            rec_delta = quant[idx].astype(np.float32)
        recon[idx] = pred + rec_delta

    meta = {
        "scheme_name": scheme_name,
        "value_codec": value_codec,
        "value_bits": value_bits,
        "latent_shape_tchw": list(latent_shape),
        "keyframe_interval_effective": int(np.where(kf_mask == 1)[0][1] - np.where(kf_mask == 1)[0][0])
        if np.count_nonzero(kf_mask) > 1
        else 0,
        "num_keyframes": int(np.count_nonzero(kf_mask)),
    }
    return np.transpose(recon, (1, 0, 2, 3)).copy(), meta


def load_samples(input_root: Path, limit: int | None) -> List[Sample]:
    lat_dir = input_root / "latents"
    video_dir = input_root / "native_16fps"
    samples: List[Sample] = []
    for latent_path in sorted(lat_dir.glob("*.pt")):
        stem = latent_path.stem
        mp4_path = video_dir / f"{stem}.mp4"
        if not mp4_path.exists():
            continue
        payload = torch.load(latent_path, map_location="cpu")
        samples.append(
            Sample(
                stem=stem,
                latent_path=latent_path,
                native_mp4_path=mp4_path,
                fps=int(payload.get("fps", 24)),
                prompt=str(payload.get("prompt", "")),
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


class WanVaeDecoder:
    def __init__(self, ckpt_dir: Path, device: str) -> None:
        cfg = WAN_CONFIGS["ti2v-5B"]
        self.device = torch.device(device)
        self.vae = Wan2_2_VAE(
            vae_pth=str(ckpt_dir / cfg.vae_checkpoint),
            device=self.device,
        )

    def decode_one(self, latents: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(latents).to(self.device)
        with torch.no_grad():
            video = self.vae.decode([tensor])[0]
        decoded = video.detach().to("cpu", copy=True)
        del tensor, video
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return decoded


def video_tensor_metrics(ref: torch.Tensor, test: torch.Tensor) -> Dict[str, float]:
    ref01 = ((ref.float() + 1.0) * 0.5).clamp(0, 1)
    test01 = ((test.float() + 1.0) * 0.5).clamp(0, 1)
    diff = test01 - ref01
    mse = float(torch.mean(diff * diff).item())
    mae = float(torch.mean(torch.abs(diff)).item())
    max_abs = float(torch.max(torch.abs(diff)).item())
    psnr = 100.0 if mse == 0 else float(10.0 * math.log10(1.0 / mse))
    ref_centered = ref01.reshape(-1) - ref01.mean()
    test_centered = test01.reshape(-1) - test01.mean()
    denom = torch.sqrt(torch.sum(ref_centered * ref_centered) * torch.sum(test_centered * test_centered))
    corr = float(torch.sum(ref_centered * test_centered) / denom) if float(denom) > 0 else 1.0
    return {
        "raw_mse": mse,
        "raw_mae": mae,
        "raw_max_abs": max_abs,
        "raw_psnr_db": psnr,
        "raw_pearson": corr,
    }


def run_ffmpeg_metric(metric: str, ref_path: Path, dist_path: Path) -> Dict[str, float]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-i",
        str(ref_path),
        "-i",
        str(dist_path),
        "-lavfi",
        f"[0:v][1:v]{metric}",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    stderr = proc.stderr
    if metric == "psnr":
        match = re.search(r"average:([0-9.]+)", stderr)
        if not match:
            raise RuntimeError(f"Could not parse PSNR from ffmpeg output:\n{stderr}")
        return {"mp4_psnr_db": float(match.group(1))}
    if metric == "ssim":
        match = re.search(r"All:([0-9.]+)", stderr)
        if not match:
            raise RuntimeError(f"Could not parse SSIM from ffmpeg output:\n{stderr}")
        return {"mp4_ssim": float(match.group(1))}
    raise ValueError(metric)


def save_reconstructed_mp4(video: torch.Tensor, out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video.unsqueeze(0), save_file=str(out_path), fps=fps)


def raw_rgb_bytes(video_cthw: torch.Tensor) -> int:
    c, t, h, w = video_cthw.shape
    return int(c * t * h * w)


def process_sample(
    sample: Sample,
    decoder: WanVaeDecoder,
    output_root: Path,
    zstd_level: int,
    value_codec: str,
    key_interval: int,
    overwrite: bool,
) -> Dict[str, object]:
    report_path = output_root / "reports" / f"{sample.stem}.json"
    if report_path.exists() and not overwrite:
        return json.loads(report_path.read_text())

    payload = torch.load(sample.latent_path, map_location="cpu")
    latents = payload["latents"].cpu().numpy().astype(np.float32, copy=False)
    raw_latent_bytes = int(payload["latents"].numel() * payload["latents"].element_size())

    codec_root = output_root / "codecs"
    intra_path = codec_root / "intra" / f"{sample.stem}.latent.zst"
    inter_path = codec_root / "inter" / f"{sample.stem}.latent.zst"
    intra_path.parent.mkdir(parents=True, exist_ok=True)
    inter_path.parent.mkdir(parents=True, exist_ok=True)

    intra_archive, _, intra_meta = build_codec_archive(
        latents,
        scheme_name="intra",
        intra_only=True,
        value_codec=value_codec,
        zstd_level=zstd_level,
        key_interval=key_interval,
    )
    inter_archive, _, inter_meta = build_codec_archive(
        latents,
        scheme_name="inter",
        intra_only=False,
        value_codec=value_codec,
        zstd_level=zstd_level,
        key_interval=key_interval,
    )
    intra_path.write_bytes(intra_archive)
    inter_path.write_bytes(inter_archive)

    intra_recon, intra_loaded_meta = decode_codec_archive(intra_path)
    inter_recon, inter_loaded_meta = decode_codec_archive(inter_path)
    if intra_recon.shape != latents.shape or inter_recon.shape != latents.shape:
        raise RuntimeError(f"Reconstructed latent shape mismatch for {sample.stem}")

    orig_video = decoder.decode_one(latents)
    intra_video = decoder.decode_one(intra_recon)
    inter_video = decoder.decode_one(inter_recon)

    intra_mp4_path = output_root / "reconstructed_mp4" / "intra" / f"{sample.stem}.mp4"
    inter_mp4_path = output_root / "reconstructed_mp4" / "inter" / f"{sample.stem}.mp4"
    save_reconstructed_mp4(intra_video, intra_mp4_path, sample.fps)
    save_reconstructed_mp4(inter_video, inter_mp4_path, sample.fps)

    intra_metrics = video_tensor_metrics(orig_video, intra_video)
    inter_metrics = video_tensor_metrics(orig_video, inter_video)
    intra_metrics.update(run_ffmpeg_metric("psnr", sample.native_mp4_path, intra_mp4_path))
    intra_metrics.update(run_ffmpeg_metric("ssim", sample.native_mp4_path, intra_mp4_path))
    inter_metrics.update(run_ffmpeg_metric("psnr", sample.native_mp4_path, inter_mp4_path))
    inter_metrics.update(run_ffmpeg_metric("ssim", sample.native_mp4_path, inter_mp4_path))

    raw_video_bytes = raw_rgb_bytes(orig_video)
    report = {
        "stem": sample.stem,
        "prompt": sample.prompt,
        "fps": sample.fps,
        "latent_path": str(sample.latent_path),
        "native_mp4_path": str(sample.native_mp4_path),
        "native_mp4_bytes": sample.native_mp4_path.stat().st_size,
        "raw_latent_bytes": raw_latent_bytes,
        "original_pt_bytes": sample.latent_path.stat().st_size,
        "raw_rgb_bytes": raw_video_bytes,
        "latent_shape": list(latents.shape),
        "intra": {
            **intra_meta,
            **intra_loaded_meta,
            **intra_metrics,
            "archive_path": str(intra_path),
            "archive_bytes": intra_path.stat().st_size,
            "archive_vs_native_mp4_ratio": intra_path.stat().st_size / sample.native_mp4_path.stat().st_size,
            "archive_vs_pt_ratio": intra_path.stat().st_size / sample.latent_path.stat().st_size,
            "archive_vs_raw_latent_ratio": intra_path.stat().st_size / raw_latent_bytes,
            "reconstructed_mp4_path": str(intra_mp4_path),
            "reconstructed_mp4_bytes": intra_mp4_path.stat().st_size,
        },
        "inter": {
            **inter_meta,
            **inter_loaded_meta,
            **inter_metrics,
            "archive_path": str(inter_path),
            "archive_bytes": inter_path.stat().st_size,
            "archive_vs_native_mp4_ratio": inter_path.stat().st_size / sample.native_mp4_path.stat().st_size,
            "archive_vs_pt_ratio": inter_path.stat().st_size / sample.latent_path.stat().st_size,
            "archive_vs_raw_latent_ratio": inter_path.stat().st_size / raw_latent_bytes,
            "reconstructed_mp4_path": str(inter_mp4_path),
            "reconstructed_mp4_bytes": inter_mp4_path.stat().st_size,
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def summarize_reports(reports: List[Dict[str, object]], output_root: Path, args: argparse.Namespace) -> None:
    summary_rows = []
    for report in reports:
        summary_rows.append(
            {
                "stem": report["stem"],
                "native_mp4_mb": round(report["native_mp4_bytes"] / 1e6, 4),
                "latent_pt_mb": round(report["original_pt_bytes"] / 1e6, 4),
                "raw_latent_mb": round(report["raw_latent_bytes"] / 1e6, 4),
                "intra_archive_mb": round(report["intra"]["archive_bytes"] / 1e6, 4),
                "intra_vs_native_mp4_ratio": round(report["intra"]["archive_vs_native_mp4_ratio"], 4),
                "intra_mp4_mb": round(report["intra"]["reconstructed_mp4_bytes"] / 1e6, 4),
                "intra_raw_psnr_db": round(report["intra"]["raw_psnr_db"], 4),
                "intra_mp4_psnr_db": round(report["intra"]["mp4_psnr_db"], 4),
                "intra_mp4_ssim": round(report["intra"]["mp4_ssim"], 6),
                "inter_archive_mb": round(report["inter"]["archive_bytes"] / 1e6, 4),
                "inter_vs_native_mp4_ratio": round(report["inter"]["archive_vs_native_mp4_ratio"], 4),
                "inter_mp4_mb": round(report["inter"]["reconstructed_mp4_bytes"] / 1e6, 4),
                "inter_raw_psnr_db": round(report["inter"]["raw_psnr_db"], 4),
                "inter_mp4_psnr_db": round(report["inter"]["mp4_psnr_db"], 4),
                "inter_mp4_ssim": round(report["inter"]["mp4_ssim"], 6),
            }
        )

    recommendation = None
    if reports:
        intra_score = (
            float(np.mean([r["intra"]["archive_vs_native_mp4_ratio"] for r in reports])),
            -float(np.mean([r["intra"]["raw_psnr_db"] for r in reports])),
            -float(np.mean([r["intra"]["mp4_ssim"] for r in reports])),
        )
        inter_score = (
            float(np.mean([r["inter"]["archive_vs_native_mp4_ratio"] for r in reports])),
            -float(np.mean([r["inter"]["raw_psnr_db"] for r in reports])),
            -float(np.mean([r["inter"]["mp4_ssim"] for r in reports])),
        )
        recommendation = "inter" if inter_score < intra_score else "intra"

    aggregate = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "num_reports": len(reports),
        "mean_native_mp4_mb": float(np.mean([r["native_mp4_bytes"] / 1e6 for r in reports])) if reports else 0.0,
        "mean_latent_pt_mb": float(np.mean([r["original_pt_bytes"] / 1e6 for r in reports])) if reports else 0.0,
        "mean_intra_archive_mb": float(np.mean([r["intra"]["archive_bytes"] / 1e6 for r in reports])) if reports else 0.0,
        "mean_inter_archive_mb": float(np.mean([r["inter"]["archive_bytes"] / 1e6 for r in reports])) if reports else 0.0,
        "mean_intra_vs_native_mp4_ratio": float(np.mean([r["intra"]["archive_vs_native_mp4_ratio"] for r in reports])) if reports else 0.0,
        "mean_inter_vs_native_mp4_ratio": float(np.mean([r["inter"]["archive_vs_native_mp4_ratio"] for r in reports])) if reports else 0.0,
        "mean_intra_raw_psnr_db": float(np.mean([r["intra"]["raw_psnr_db"] for r in reports])) if reports else 0.0,
        "mean_inter_raw_psnr_db": float(np.mean([r["inter"]["raw_psnr_db"] for r in reports])) if reports else 0.0,
        "mean_intra_mp4_psnr_db": float(np.mean([r["intra"]["mp4_psnr_db"] for r in reports])) if reports else 0.0,
        "mean_inter_mp4_psnr_db": float(np.mean([r["inter"]["mp4_psnr_db"] for r in reports])) if reports else 0.0,
        "mean_intra_mp4_ssim": float(np.mean([r["intra"]["mp4_ssim"] for r in reports])) if reports else 0.0,
        "mean_inter_mp4_ssim": float(np.mean([r["inter"]["mp4_ssim"] for r in reports])) if reports else 0.0,
        "recommended_family": recommendation,
        "notes": [
            "intra = each latent frame coded independently inside one sample.",
            "inter = temporal predictive coding with periodic keyframes over adjacent latent frames.",
            "archive_vs_native_mp4_ratio < 1 means the compressed latent container is smaller than the original MP4.",
        ],
        "rows": summary_rows,
    }
    (output_root / "summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = load_samples(input_root, args.limit)
    if not samples:
        raise SystemExit(f"No matched latent/mp4 samples found under {input_root}")

    print(f"Found {len(samples)} matched samples under {input_root}")
    decoder = WanVaeDecoder(Path(args.ckpt_dir), args.device)

    reports = []
    for idx, sample in enumerate(samples, start=1):
        print(f"[{idx}/{len(samples)}] Processing {sample.stem}")
        report = process_sample(
            sample=sample,
            decoder=decoder,
            output_root=output_root,
            zstd_level=args.zstd_level,
            value_codec=args.value_codec,
            key_interval=args.keyframe_interval,
            overwrite=args.overwrite,
        )
        reports.append(report)
        print(
            f"  intra={report['intra']['archive_bytes'] / 1e6:.2f} MB "
            f"inter={report['inter']['archive_bytes'] / 1e6:.2f} MB "
            f"native_mp4={report['native_mp4_bytes'] / 1e6:.2f} MB"
        )

    summarize_reports(reports, output_root, args)
    print(f"Wrote summary to {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
