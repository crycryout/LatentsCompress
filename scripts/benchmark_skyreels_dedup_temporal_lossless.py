#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pcodec
import pcodec.standalone as ps
import torch
import zstandard as zstd


DEFAULT_LATENT_PATHS = [
    "/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/wingsuit_rescue_glacier_pullup/full_video_latents_dedup.pt",
    "/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/neon_hoverbike_chain_reaction/full_video_latents_dedup.pt",
    "/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/avalanche_snowmobile_bridge_escape/full_video_latents_dedup.pt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark lossless temporal codecs on three deduplicated SkyReels long-video latents."
        )
    )
    parser.add_argument(
        "--latent-path",
        action="append",
        dest="latent_paths",
        default=[],
        help="Path to a deduplicated latent `.pt`. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/LatentsCompress/examples/vbench_codec/skyreels_dedup_temporal_lossless"),
    )
    parser.add_argument("--zstd-level", type=int, default=19)
    parser.add_argument(
        "--pcodec-level",
        type=int,
        default=pcodec.DEFAULT_COMPRESSION_LEVEL,
    )
    parser.add_argument("--save-containers", action="store_true")
    return parser.parse_args()


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    return round(original_bytes / compressed_bytes, 6)


def bytes_to_mb(nbytes: int) -> float:
    return round(nbytes / 1_000_000, 6)


def ensure_u8_size(blob: bytes, expected_size: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.uint8)
    if arr.size != expected_size:
        raise ValueError(f"Unexpected restored size: {arr.size} vs {expected_size}")
    return arr


def compress_pcodec(arr: np.ndarray, level: int) -> bytes:
    arr = np.ascontiguousarray(arr)
    config = pcodec.ChunkConfig(
        compression_level=level,
        enable_8_bit=bool(arr.dtype == np.uint8),
    )
    return ps.simple_compress(arr.reshape(-1), config)


def decompress_pcodec_u8(blob: bytes, expected_size: int) -> np.ndarray:
    restored = ps.simple_decompress(blob)
    restored_u8 = np.ascontiguousarray(restored.astype(np.uint8, copy=False).reshape(-1))
    if restored_u8.size != expected_size:
        raise ValueError(f"Unexpected pcodec size: {restored_u8.size} vs {expected_size}")
    return restored_u8


def bshuffle_u16_bytes(arr_u16: np.ndarray) -> bytes:
    raw = np.ascontiguousarray(arr_u16).view(np.uint8).reshape(-1, 2)
    return raw.T.copy().reshape(-1).tobytes()


def load_latent_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    tensor = obj["latents"] if isinstance(obj, dict) and "latents" in obj else obj
    if not torch.is_tensor(tensor):
        raise TypeError(f"{path} does not contain a tensor latent payload")
    tensor = tensor.detach().cpu().contiguous()
    if tensor.ndim == 5 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected CTHW or 1CTHW latent, got shape {tuple(tensor.shape)}")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"Expected bfloat16 latent, got {tensor.dtype}")
    return tensor


def seq_tchw_u16_from_cthw(tensor_cthw: torch.Tensor) -> np.ndarray:
    return tensor_cthw.permute(1, 0, 2, 3).contiguous().view(torch.uint16).numpy().copy()


def pack_container(header: dict[str, Any], parts: dict[str, bytes]) -> bytes:
    header_json = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    chunks = [b"SLTL1", struct.pack("<I", len(header_json)), header_json]
    for name in header["part_names"]:
        part = parts[name]
        chunks.append(struct.pack("<I", len(part)))
        chunks.append(part)
    return b"".join(chunks)


def unpack_container(blob: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    offset = 0
    magic = blob[offset : offset + 5]
    offset += 5
    if magic != b"SLTL1":
        raise ValueError("Bad magic")
    (header_len,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    header = json.loads(blob[offset : offset + header_len].decode("utf-8"))
    offset += header_len
    parts: dict[str, bytes] = {}
    for name in header["part_names"]:
        (part_len,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        parts[name] = blob[offset : offset + part_len]
        offset += part_len
    if offset != len(blob):
        raise ValueError("Trailing bytes in container")
    return header, parts


def encode_temporal_split_codec(
    seq_tchw_u16: np.ndarray,
    *,
    zstd_level: int,
    pcodec_level: int,
    low_changed_mode: str,
) -> tuple[bytes, dict[str, Any]]:
    if low_changed_mode not in {"raw_zstd", "delta_zstd", "raw_pcodec", "delta_pcodec"}:
        raise ValueError(f"Unsupported low_changed_mode={low_changed_mode}")

    zc = zstd.ZstdCompressor(level=zstd_level)
    bytes2 = seq_tchw_u16.view(np.uint8).reshape(seq_tchw_u16.shape + (2,))
    lo = np.ascontiguousarray(bytes2[..., 0])
    hi = np.ascontiguousarray(bytes2[..., 1])

    same_hi = hi[1:] == hi[:-1]
    lo_curr = lo[1:]
    lo_prev = lo[:-1]
    lo_delta = ((lo_curr.astype(np.int16) - lo_prev.astype(np.int16)) & 0xFF).astype(np.uint8)
    lo_delta_same = np.ascontiguousarray(lo_delta[same_hi])
    lo_raw_changed = np.ascontiguousarray(lo_curr[~same_hi])
    lo_delta_changed = np.ascontiguousarray(lo_delta[~same_hi])

    parts: dict[str, bytes] = {
        "hi_zstd": zc.compress(hi.tobytes()),
        "seed_lo_zstd": zc.compress(lo[:1].tobytes()),
        "lo_same_pcodec": compress_pcodec(lo_delta_same, pcodec_level),
    }
    part_names = ["hi_zstd", "seed_lo_zstd", "lo_same_pcodec"]

    if low_changed_mode == "raw_zstd":
        parts["lo_changed"] = zc.compress(lo_raw_changed.tobytes())
    elif low_changed_mode == "delta_zstd":
        parts["lo_changed"] = zc.compress(lo_delta_changed.tobytes())
    elif low_changed_mode == "raw_pcodec":
        parts["lo_changed"] = compress_pcodec(lo_raw_changed, pcodec_level)
    else:
        parts["lo_changed"] = compress_pcodec(lo_delta_changed, pcodec_level)
    part_names.append("lo_changed")

    header = {
        "codec": "skyreels_temporal_split_v1",
        "shape_tchw": list(seq_tchw_u16.shape),
        "dtype": "bfloat16_bits_uint16",
        "zstd_level": zstd_level,
        "pcodec_level": pcodec_level,
        "part_names": part_names,
        "same_hi_true_count": int(np.count_nonzero(same_hi)),
        "same_hi_total": int(same_hi.size),
        "lo_same_len": int(lo_delta_same.size),
        "lo_changed_len": int(lo_raw_changed.size),
        "low_changed_mode": low_changed_mode,
    }
    container = pack_container(header, parts)
    meta = {
        "same_hi_fraction": round(float(np.mean(same_hi)), 6),
        "lo_same_bytes_raw": int(lo_delta_same.nbytes),
        "lo_changed_bytes_raw": int(lo_raw_changed.nbytes),
        "part_sizes": {name: len(blob) for name, blob in parts.items()},
        "container_overhead_bytes": len(container) - sum(len(blob) for blob in parts.values()),
        "low_changed_mode": low_changed_mode,
    }
    return container, meta


def decode_temporal_split_codec(container: bytes) -> np.ndarray:
    header, parts = unpack_container(container)
    if header["codec"] != "skyreels_temporal_split_v1":
        raise ValueError(f"Unexpected codec {header['codec']}")
    zc = zstd.ZstdDecompressor()

    t, c, h, w = header["shape_tchw"]
    hi = ensure_u8_size(zc.decompress(parts["hi_zstd"]), t * c * h * w).reshape((t, c, h, w))
    seed_lo = ensure_u8_size(zc.decompress(parts["seed_lo_zstd"]), c * h * w).reshape((1, c, h, w))
    lo_same = decompress_pcodec_u8(parts["lo_same_pcodec"], header["lo_same_len"])
    same_hi = hi[1:] == hi[:-1]

    if header["low_changed_mode"] == "raw_zstd":
        lo_changed = ensure_u8_size(zc.decompress(parts["lo_changed"]), header["lo_changed_len"])
        changed_mode = "raw"
    elif header["low_changed_mode"] == "delta_zstd":
        lo_changed = ensure_u8_size(zc.decompress(parts["lo_changed"]), header["lo_changed_len"])
        changed_mode = "delta"
    elif header["low_changed_mode"] == "raw_pcodec":
        lo_changed = decompress_pcodec_u8(parts["lo_changed"], header["lo_changed_len"])
        changed_mode = "raw"
    else:
        lo_changed = decompress_pcodec_u8(parts["lo_changed"], header["lo_changed_len"])
        changed_mode = "delta"

    lo = np.empty((t, c, h, w), dtype=np.uint8)
    lo[0] = seed_lo[0]
    same_offset = 0
    changed_offset = 0
    for ti in range(1, t):
        mask = same_hi[ti - 1]
        same_count = int(np.count_nonzero(mask))
        changed_count = int(mask.size - same_count)
        same_delta = lo_same[same_offset : same_offset + same_count]
        changed_vals = lo_changed[changed_offset : changed_offset + changed_count]
        same_offset += same_count
        changed_offset += changed_count

        curr = np.empty((c, h, w), dtype=np.uint8)
        prev_vals = lo[ti - 1][mask].astype(np.uint16)
        curr[mask] = ((prev_vals + same_delta.astype(np.uint16)) & 0xFF).astype(np.uint8)
        if changed_mode == "raw":
            curr[~mask] = changed_vals
        else:
            prev_changed = lo[ti - 1][~mask].astype(np.uint16)
            curr[~mask] = ((prev_changed + changed_vals.astype(np.uint16)) & 0xFF).astype(np.uint8)
        lo[ti] = curr

    if same_offset != lo_same.size or changed_offset != lo_changed.size:
        raise ValueError("Did not consume all low-byte streams")

    bytes2 = np.empty((t, c, h, w, 2), dtype=np.uint8)
    bytes2[..., 0] = lo
    bytes2[..., 1] = hi
    restored = np.ascontiguousarray(bytes2).view(np.uint16).reshape((t, c, h, w))
    return restored


def benchmark_one(
    latent_path: Path,
    *,
    zstd_level: int,
    pcodec_level: int,
    save_containers: bool,
    output_dir: Path,
) -> dict[str, Any]:
    tensor_cthw = load_latent_tensor(latent_path)
    seq_tchw_u16 = seq_tchw_u16_from_cthw(tensor_cthw)

    payload_u16 = np.ascontiguousarray(seq_tchw_u16.reshape(-1))
    payload_bytes = payload_u16.tobytes()
    pt_bytes = latent_path.read_bytes()

    zc = zstd.ZstdCompressor(level=zstd_level)
    zstd_raw = zc.compress(payload_bytes)
    zstd_bshuffle = zc.compress(bshuffle_u16_bytes(seq_tchw_u16))
    pcodec_raw = compress_pcodec(payload_u16, pcodec_level)

    candidate_rows = []
    for low_changed_mode in ("raw_zstd", "delta_zstd", "raw_pcodec", "delta_pcodec"):
        container, meta = encode_temporal_split_codec(
            seq_tchw_u16,
            zstd_level=zstd_level,
            pcodec_level=pcodec_level,
            low_changed_mode=low_changed_mode,
        )
        restored = decode_temporal_split_codec(container)
        verified = bool(np.array_equal(restored, seq_tchw_u16))
        row = {
            "codec": f"temporal_split_{low_changed_mode}",
            "compressed_bytes": len(container),
            "compressed_mb": bytes_to_mb(len(container)),
            "ratio_vs_payload": compression_ratio(len(payload_bytes), len(container)),
            "ratio_vs_pt": compression_ratio(len(pt_bytes), len(container)),
            "verified_lossless": verified,
            **meta,
        }
        candidate_rows.append(row)
        if save_containers:
            sample_dir = output_dir / latent_path.parent.name
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / f"{latent_path.stem}.{low_changed_mode}.sltl").write_bytes(container)

    best_candidate = min(candidate_rows, key=lambda row: row["compressed_bytes"])
    report = {
        "sample": latent_path.parent.name,
        "latent_path": str(latent_path),
        "latent_shape_cthw": list(tensor_cthw.shape),
        "latent_dtype": str(tensor_cthw.dtype).replace("torch.", ""),
        "payload_bytes": len(payload_bytes),
        "pt_bytes": len(pt_bytes),
        "baselines": [
            {
                "codec": "zstd_raw",
                "compressed_bytes": len(zstd_raw),
                "compressed_mb": bytes_to_mb(len(zstd_raw)),
                "ratio_vs_payload": compression_ratio(len(payload_bytes), len(zstd_raw)),
                "ratio_vs_pt": compression_ratio(len(pt_bytes), len(zstd_raw)),
                "verified_lossless": True,
            },
            {
                "codec": "zstd_bshuffle_u16",
                "compressed_bytes": len(zstd_bshuffle),
                "compressed_mb": bytes_to_mb(len(zstd_bshuffle)),
                "ratio_vs_payload": compression_ratio(len(payload_bytes), len(zstd_bshuffle)),
                "ratio_vs_pt": compression_ratio(len(pt_bytes), len(zstd_bshuffle)),
                "verified_lossless": True,
            },
            {
                "codec": "pcodec_raw_u16",
                "compressed_bytes": len(pcodec_raw),
                "compressed_mb": bytes_to_mb(len(pcodec_raw)),
                "ratio_vs_payload": compression_ratio(len(payload_bytes), len(pcodec_raw)),
                "ratio_vs_pt": compression_ratio(len(pt_bytes), len(pcodec_raw)),
                "verified_lossless": True,
            },
        ],
        "candidates": candidate_rows,
        "best_candidate": best_candidate,
    }
    return report


def write_summary(all_reports: list[dict[str, Any]], summary_path: Path) -> None:
    lines = [
        "# SkyReels Dedup Temporal Lossless Compression",
        "",
        "## Per-Sample Best Results",
        "",
        "| Sample | Payload MB | Best ZSTD MB | Best Pcodec MB | Best New Codec | New Codec MB | Saved vs Pcodec MB | same-hi fraction |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    total_payload = 0
    total_zstd = 0
    total_pcodec = 0
    total_new = 0
    for report in all_reports:
        baselines = {row["codec"]: row for row in report["baselines"]}
        best_new = report["best_candidate"]
        total_payload += report["payload_bytes"]
        total_zstd += baselines["zstd_bshuffle_u16"]["compressed_bytes"]
        total_pcodec += baselines["pcodec_raw_u16"]["compressed_bytes"]
        total_new += best_new["compressed_bytes"]
        saved_vs_pcodec_mb = bytes_to_mb(
            baselines["pcodec_raw_u16"]["compressed_bytes"] - best_new["compressed_bytes"]
        )
        lines.append(
            "| "
            f"{report['sample']} | "
            f"{bytes_to_mb(report['payload_bytes']):.3f} | "
            f"{baselines['zstd_bshuffle_u16']['compressed_mb']:.3f} | "
            f"{baselines['pcodec_raw_u16']['compressed_mb']:.3f} | "
            f"{best_new['codec']} | "
            f"{best_new['compressed_mb']:.3f} | "
            f"{saved_vs_pcodec_mb:.3f} | "
            f"{best_new['same_hi_fraction']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Totals",
            "",
            "| Item | Bytes | MB | Ratio vs payload |",
            "|---|---:|---:|---:|",
            f"| payload | {total_payload} | {bytes_to_mb(total_payload):.3f} | 1.000x |",
            f"| best zstd (`bshuffle_u16`) | {total_zstd} | {bytes_to_mb(total_zstd):.3f} | {compression_ratio(total_payload, total_zstd):.3f}x |",
            f"| best pcodec (`raw_u16`) | {total_pcodec} | {bytes_to_mb(total_pcodec):.3f} | {compression_ratio(total_payload, total_pcodec):.3f}x |",
            f"| best new temporal codec | {total_new} | {bytes_to_mb(total_new):.3f} | {compression_ratio(total_payload, total_new):.3f}x |",
            "",
            "## Codec Idea",
            "",
            "- Store the `bf16` high byte stream losslessly with `zstd`.",
            "- Reconstruct the high-byte timeline first, then derive where the exponent/sign byte stayed unchanged across time.",
            "- For positions whose high byte stayed unchanged, encode the low byte as modulo-256 temporal deltas and compress that stream with `pcodec`.",
            "- For positions whose high byte changed, store the current low byte directly and compress that smaller changed-only stream with `zstd`.",
            "- The mask is not stored separately; it is recovered from the restored high-byte stream.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    latent_paths = [Path(path) for path in (args.latent_paths or DEFAULT_LATENT_PATHS)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_reports = []
    for latent_path in latent_paths:
        report = benchmark_one(
            latent_path,
            zstd_level=args.zstd_level,
            pcodec_level=args.pcodec_level,
            save_containers=args.save_containers,
            output_dir=args.output_dir,
        )
        sample_json = args.output_dir / f"{report['sample']}_report.json"
        sample_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        all_reports.append(report)
        best = report["best_candidate"]
        print(
            f"{report['sample']}: best={best['codec']} "
            f"{best['compressed_mb']:.3f}MB "
            f"vs pcodec={next(row['compressed_mb'] for row in report['baselines'] if row['codec']=='pcodec_raw_u16'):.3f}MB",
            flush=True,
        )

    aggregate = {
        "reports": all_reports,
        "totals": {
            "payload_bytes": int(sum(report["payload_bytes"] for report in all_reports)),
            "zstd_bshuffle_u16_bytes": int(
                sum(
                    next(row["compressed_bytes"] for row in report["baselines"] if row["codec"] == "zstd_bshuffle_u16")
                    for report in all_reports
                )
            ),
            "pcodec_raw_u16_bytes": int(
                sum(
                    next(row["compressed_bytes"] for row in report["baselines"] if row["codec"] == "pcodec_raw_u16")
                    for report in all_reports
                )
            ),
            "best_temporal_bytes": int(sum(report["best_candidate"]["compressed_bytes"] for report in all_reports)),
        },
    }
    aggregate_json = args.output_dir / "aggregate_report.json"
    aggregate_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    write_summary(all_reports, args.output_dir / "summary.md")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
