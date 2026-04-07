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
            "Benchmark a second-generation lossless codec for deduplicated SkyReels long-video latents. "
            "This codec adaptively chooses a per-channel mode based on temporal continuity."
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
        default=Path("/root/LatentsCompress/examples/vbench_codec/skyreels_dedup_temporal_codec_v2"),
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


def load_latent_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    tensor = obj["latents"] if isinstance(obj, dict) and "latents" in obj else obj
    if not torch.is_tensor(tensor):
        raise TypeError(f"{path} does not contain a tensor latent payload")
    tensor = tensor.detach().cpu().contiguous()
    if tensor.ndim == 5 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected CTHW tensor, got shape {tuple(tensor.shape)}")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"Expected bfloat16 tensor, got {tensor.dtype}")
    return tensor


def compress_pcodec(arr: np.ndarray, level: int) -> bytes:
    arr = np.ascontiguousarray(arr)
    config = pcodec.ChunkConfig(
        compression_level=level,
        enable_8_bit=bool(arr.dtype == np.uint8),
    )
    return ps.simple_compress(arr.reshape(-1), config)


def decompress_pcodec(blob: bytes, dtype: np.dtype, expected_size: int) -> np.ndarray:
    restored = ps.simple_decompress(blob)
    restored = np.ascontiguousarray(restored.astype(dtype, copy=False).reshape(-1))
    if restored.size != expected_size:
        raise ValueError(f"Unexpected restored size: {restored.size} vs {expected_size}")
    return restored


def ensure_u8_size(blob: bytes, expected_size: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.uint8)
    if arr.size != expected_size:
        raise ValueError(f"Unexpected restored size: {arr.size} vs {expected_size}")
    return arr


def bshuffle_u16_bytes(arr_u16: np.ndarray) -> bytes:
    raw = np.ascontiguousarray(arr_u16).view(np.uint8).reshape(-1, 2)
    return raw.T.copy().reshape(-1).tobytes()


def pack_container(header: dict[str, Any], parts: list[bytes]) -> bytes:
    header_json = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    chunks = [b"SLT2", struct.pack("<I", len(header_json)), header_json]
    for blob in parts:
        chunks.append(struct.pack("<I", len(blob)))
        chunks.append(blob)
    return b"".join(chunks)


def unpack_container(blob: bytes) -> tuple[dict[str, Any], list[bytes]]:
    offset = 0
    magic = blob[offset : offset + 4]
    offset += 4
    if magic != b"SLT2":
        raise ValueError("Bad magic")
    (header_len,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    header = json.loads(blob[offset : offset + header_len].decode("utf-8"))
    offset += header_len
    parts: list[bytes] = []
    for _ in range(header["num_parts"]):
        (part_len,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        parts.append(blob[offset : offset + part_len])
        offset += part_len
    if offset != len(blob):
        raise ValueError("Trailing bytes in container")
    return header, parts


def encode_channel_raw_pcodec(channel_thw_u16: np.ndarray, pcodec_level: int) -> tuple[bytes, dict[str, Any]]:
    payload = compress_pcodec(channel_thw_u16.reshape(-1), pcodec_level)
    meta = {
        "mode": "raw_pcodec",
        "shape_thw": list(channel_thw_u16.shape),
        "parts": [
            {
                "kind": "pcodec",
                "dtype": "uint16",
                "length": int(channel_thw_u16.size),
            }
        ],
    }
    return payload, meta


def decode_channel_raw_pcodec(blob: bytes, meta: dict[str, Any]) -> np.ndarray:
    shape = tuple(meta["shape_thw"])
    payload = decompress_pcodec(blob, np.uint16, int(np.prod(shape)))
    return payload.reshape(shape)


def encode_channel_temporal_split(
    channel_thw_u16: np.ndarray,
    *,
    zstd_level: int,
    pcodec_level: int,
    low_changed_mode: str,
) -> tuple[list[bytes], dict[str, Any]]:
    zc = zstd.ZstdCompressor(level=zstd_level)
    bytes2 = channel_thw_u16.view(np.uint8).reshape(channel_thw_u16.shape + (2,))
    lo = np.ascontiguousarray(bytes2[..., 0])
    hi = np.ascontiguousarray(bytes2[..., 1])

    same = hi[1:] == hi[:-1]
    lo_curr = lo[1:]
    lo_prev = lo[:-1]
    lo_delta = ((lo_curr.astype(np.int16) - lo_prev.astype(np.int16)) & 0xFF).astype(np.uint8)
    lo_same = np.ascontiguousarray(lo_delta[same])
    lo_changed_raw = np.ascontiguousarray(lo_curr[~same])
    lo_changed_delta = np.ascontiguousarray(lo_delta[~same])

    parts: list[bytes] = [
        zc.compress(hi.tobytes()),
        zc.compress(lo[:1].tobytes()),
        compress_pcodec(lo_same, pcodec_level),
    ]
    part_meta = [
        {"kind": "zstd", "stream": "hi_raw", "dtype": "uint8", "length": int(hi.size)},
        {"kind": "zstd", "stream": "lo_seed", "dtype": "uint8", "length": int(lo[:1].size)},
        {"kind": "pcodec", "stream": "lo_same_delta", "dtype": "uint8", "length": int(lo_same.size)},
    ]
    if low_changed_mode == "raw_zstd":
        parts.append(zc.compress(lo_changed_raw.tobytes()))
        part_meta.append({"kind": "zstd", "stream": "lo_changed_raw", "dtype": "uint8", "length": int(lo_changed_raw.size)})
    elif low_changed_mode == "raw_pcodec":
        parts.append(compress_pcodec(lo_changed_raw, pcodec_level))
        part_meta.append({"kind": "pcodec", "stream": "lo_changed_raw", "dtype": "uint8", "length": int(lo_changed_raw.size)})
    elif low_changed_mode == "delta_zstd":
        parts.append(zc.compress(lo_changed_delta.tobytes()))
        part_meta.append({"kind": "zstd", "stream": "lo_changed_delta", "dtype": "uint8", "length": int(lo_changed_delta.size)})
    elif low_changed_mode == "delta_pcodec":
        parts.append(compress_pcodec(lo_changed_delta, pcodec_level))
        part_meta.append({"kind": "pcodec", "stream": "lo_changed_delta", "dtype": "uint8", "length": int(lo_changed_delta.size)})
    else:
        raise ValueError(f"Unsupported low_changed_mode={low_changed_mode}")

    meta = {
        "mode": f"temporal_split_{low_changed_mode}",
        "shape_thw": list(channel_thw_u16.shape),
        "same_hi_fraction": round(float(np.mean(same)), 6),
        "low_changed_mode": low_changed_mode,
        "parts": part_meta,
    }
    return parts, meta


def decode_channel_temporal_split(parts: list[bytes], meta: dict[str, Any], zstd_level: int) -> np.ndarray:
    _ = zstd_level
    zd = zstd.ZstdDecompressor()
    shape = tuple(meta["shape_thw"])
    t, h, w = shape

    hi = ensure_u8_size(zd.decompress(parts[0]), t * h * w).reshape(shape)
    lo_seed = ensure_u8_size(zd.decompress(parts[1]), h * w).reshape((1, h, w))
    lo_same_len = meta["parts"][2]["length"]
    lo_same = decompress_pcodec(parts[2], np.uint8, lo_same_len)
    same_hi = hi[1:] == hi[:-1]

    changed_meta = meta["parts"][3]
    if changed_meta["kind"] == "zstd":
        lo_changed = ensure_u8_size(zd.decompress(parts[3]), changed_meta["length"])
    else:
        lo_changed = decompress_pcodec(parts[3], np.uint8, changed_meta["length"])

    lo = np.empty(shape, dtype=np.uint8)
    lo[0] = lo_seed[0]
    same_offset = 0
    changed_offset = 0
    changed_is_raw = changed_meta["stream"] == "lo_changed_raw"
    for ti in range(1, t):
        mask = same_hi[ti - 1]
        same_count = int(np.count_nonzero(mask))
        changed_count = int(mask.size - same_count)
        same_delta = lo_same[same_offset : same_offset + same_count]
        changed_vals = lo_changed[changed_offset : changed_offset + changed_count]
        same_offset += same_count
        changed_offset += changed_count

        curr = np.empty((h, w), dtype=np.uint8)
        prev_same = lo[ti - 1][mask].astype(np.uint16)
        curr[mask] = ((prev_same + same_delta.astype(np.uint16)) & 0xFF).astype(np.uint8)
        if changed_is_raw:
            curr[~mask] = changed_vals
        else:
            prev_changed = lo[ti - 1][~mask].astype(np.uint16)
            curr[~mask] = ((prev_changed + changed_vals.astype(np.uint16)) & 0xFF).astype(np.uint8)
        lo[ti] = curr

    bytes2 = np.empty(shape + (2,), dtype=np.uint8)
    bytes2[..., 0] = lo
    bytes2[..., 1] = hi
    return np.ascontiguousarray(bytes2).view(np.uint16).reshape(shape)


def encode_adaptive_channels(
    tensor_cthw: torch.Tensor,
    *,
    zstd_level: int,
    pcodec_level: int,
) -> tuple[bytes, dict[str, Any]]:
    zc = zstd.ZstdCompressor(level=zstd_level)
    seq_tchw_u16 = tensor_cthw.view(torch.uint16).permute(1, 0, 2, 3).contiguous().numpy().copy()

    channel_rows = []
    all_parts: list[bytes] = []
    for channel in range(seq_tchw_u16.shape[1]):
        channel_thw = np.ascontiguousarray(seq_tchw_u16[:, channel, :, :])
        candidates: list[tuple[int, str, list[bytes], dict[str, Any]]] = []

        raw_blob, raw_meta = encode_channel_raw_pcodec(channel_thw, pcodec_level)
        candidates.append((len(raw_blob), "raw_pcodec", [raw_blob], raw_meta))

        for low_changed_mode in ("raw_zstd", "raw_pcodec", "delta_zstd", "delta_pcodec"):
            parts, meta = encode_channel_temporal_split(
                channel_thw,
                zstd_level=zstd_level,
                pcodec_level=pcodec_level,
                low_changed_mode=low_changed_mode,
            )
            candidates.append((sum(len(part) for part in parts), meta["mode"], parts, meta))

        best_bytes, best_mode, best_parts, best_meta = min(candidates, key=lambda item: item[0])
        best_meta = {"channel": channel, **best_meta}
        best_meta["compressed_bytes"] = best_bytes
        best_meta["num_parts"] = len(best_parts)
        channel_rows.append(best_meta)
        all_parts.extend(best_parts)

    header = {
        "codec": "skyreels_temporal_codec_v2",
        "shape_cthw": list(tensor_cthw.shape),
        "dtype": "bfloat16",
        "zstd_level": zstd_level,
        "pcodec_level": pcodec_level,
        "num_parts": len(all_parts),
        "channels": channel_rows,
    }
    container = pack_container(header, all_parts)
    return container, header


def decode_adaptive_channels(container: bytes) -> np.ndarray:
    header, parts = unpack_container(container)
    if header["codec"] != "skyreels_temporal_codec_v2":
        raise ValueError(f"Unexpected codec {header['codec']}")

    c, t, h, w = header["shape_cthw"]
    seq_tchw_u16 = np.empty((t, c, h, w), dtype=np.uint16)
    part_offset = 0
    for channel_meta in header["channels"]:
        channel = channel_meta["channel"]
        num_parts = channel_meta["num_parts"]
        channel_parts = parts[part_offset : part_offset + num_parts]
        part_offset += num_parts
        if channel_meta["mode"] == "raw_pcodec":
            restored = decode_channel_raw_pcodec(channel_parts[0], channel_meta)
        else:
            restored = decode_channel_temporal_split(channel_parts, channel_meta, header["zstd_level"])
        seq_tchw_u16[:, channel, :, :] = restored

    if part_offset != len(parts):
        raise ValueError("Unconsumed parts")
    return seq_tchw_u16


def benchmark_one(
    latent_path: Path,
    *,
    zstd_level: int,
    pcodec_level: int,
    save_containers: bool,
    output_dir: Path,
) -> dict[str, Any]:
    tensor_cthw = load_latent_tensor(latent_path)
    seq_tchw_u16 = tensor_cthw.view(torch.uint16).permute(1, 0, 2, 3).contiguous().numpy().copy()
    payload_bytes = seq_tchw_u16.reshape(-1).tobytes()
    pt_bytes = latent_path.read_bytes()
    zc = zstd.ZstdCompressor(level=zstd_level)

    zstd_raw = zc.compress(payload_bytes)
    zstd_bshuffle = zc.compress(bshuffle_u16_bytes(seq_tchw_u16))
    pcodec_global = compress_pcodec(seq_tchw_u16.reshape(-1), pcodec_level)

    per_channel_pcodec = 0
    for channel in range(seq_tchw_u16.shape[1]):
        channel_thw = np.ascontiguousarray(seq_tchw_u16[:, channel, :, :])
        blob = compress_pcodec(channel_thw.reshape(-1), pcodec_level)
        per_channel_pcodec += len(blob)

    container, header = encode_adaptive_channels(
        tensor_cthw,
        zstd_level=zstd_level,
        pcodec_level=pcodec_level,
    )
    restored = decode_adaptive_channels(container)
    verified = bool(np.array_equal(restored, seq_tchw_u16))

    chosen_mode_counts: dict[str, int] = {}
    top_savings: list[dict[str, Any]] = []
    for row in header["channels"]:
        chosen_mode_counts[row["mode"]] = chosen_mode_counts.get(row["mode"], 0) + 1
        channel_thw = np.ascontiguousarray(seq_tchw_u16[:, row["channel"], :, :])
        raw_pcodec_bytes = len(compress_pcodec(channel_thw.reshape(-1), pcodec_level))
        top_savings.append(
            {
                "channel": row["channel"],
                "mode": row["mode"],
                "raw_pcodec_bytes": raw_pcodec_bytes,
                "chosen_bytes": row["compressed_bytes"],
                "saved_vs_raw_pcodec_bytes": raw_pcodec_bytes - row["compressed_bytes"],
                "same_hi_fraction": row.get("same_hi_fraction"),
            }
        )
    top_savings.sort(key=lambda item: item["saved_vs_raw_pcodec_bytes"], reverse=True)

    report = {
        "sample": latent_path.parent.name,
        "latent_path": str(latent_path),
        "latent_shape_cthw": list(tensor_cthw.shape),
        "latent_dtype": "bfloat16",
        "payload_bytes": len(payload_bytes),
        "pt_bytes": len(pt_bytes),
        "baselines": [
            {
                "codec": "zstd_raw",
                "compressed_bytes": len(zstd_raw),
                "compressed_mb": bytes_to_mb(len(zstd_raw)),
                "ratio_vs_payload": compression_ratio(len(payload_bytes), len(zstd_raw)),
                "ratio_vs_pt": compression_ratio(len(pt_bytes), len(zstd_raw)),
            },
            {
                "codec": "zstd_bshuffle_u16",
                "compressed_bytes": len(zstd_bshuffle),
                "compressed_mb": bytes_to_mb(len(zstd_bshuffle)),
                "ratio_vs_payload": compression_ratio(len(payload_bytes), len(zstd_bshuffle)),
                "ratio_vs_pt": compression_ratio(len(pt_bytes), len(zstd_bshuffle)),
            },
            {
                "codec": "pcodec_raw_global",
                "compressed_bytes": len(pcodec_global),
                "compressed_mb": bytes_to_mb(len(pcodec_global)),
                "ratio_vs_payload": compression_ratio(len(payload_bytes), len(pcodec_global)),
                "ratio_vs_pt": compression_ratio(len(pt_bytes), len(pcodec_global)),
            },
            {
                "codec": "pcodec_raw_per_channel_sum",
                "compressed_bytes": per_channel_pcodec,
                "compressed_mb": bytes_to_mb(per_channel_pcodec),
                "ratio_vs_payload": compression_ratio(len(payload_bytes), per_channel_pcodec),
                "ratio_vs_pt": compression_ratio(len(pt_bytes), per_channel_pcodec),
            },
        ],
        "codec_v2": {
            "compressed_bytes": len(container),
            "compressed_mb": bytes_to_mb(len(container)),
            "ratio_vs_payload": compression_ratio(len(payload_bytes), len(container)),
            "ratio_vs_pt": compression_ratio(len(pt_bytes), len(container)),
            "verified_lossless": verified,
            "container_overhead_bytes": len(container) - sum(
                row["compressed_bytes"] for row in header["channels"]
            ),
            "chosen_mode_counts": chosen_mode_counts,
            "channels": header["channels"],
            "top_savings_vs_raw_pcodec": top_savings[:8],
        },
    }

    if save_containers:
        sample_dir = output_dir / latent_path.parent.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / f"{latent_path.stem}.slt2").write_bytes(container)

    return report


def write_summary(reports: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# SkyReels Dedup Temporal Codec V2",
        "",
        "## Per-Sample Results",
        "",
        "| Sample | Global Pcodec MB | Per-Channel Pcodec MB | Codec V2 MB | Saved vs Global Pcodec MB | Saved vs Per-Channel Pcodec MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    total_payload = 0
    total_global_pcodec = 0
    total_per_channel_pcodec = 0
    total_v2 = 0
    for report in reports:
        baselines = {row["codec"]: row for row in report["baselines"]}
        v2 = report["codec_v2"]
        total_payload += report["payload_bytes"]
        total_global_pcodec += baselines["pcodec_raw_global"]["compressed_bytes"]
        total_per_channel_pcodec += baselines["pcodec_raw_per_channel_sum"]["compressed_bytes"]
        total_v2 += v2["compressed_bytes"]
        lines.append(
            "| "
            f"{report['sample']} | "
            f"{baselines['pcodec_raw_global']['compressed_mb']:.3f} | "
            f"{baselines['pcodec_raw_per_channel_sum']['compressed_mb']:.3f} | "
            f"{v2['compressed_mb']:.3f} | "
            f"{bytes_to_mb(baselines['pcodec_raw_global']['compressed_bytes'] - v2['compressed_bytes']):.3f} | "
            f"{bytes_to_mb(baselines['pcodec_raw_per_channel_sum']['compressed_bytes'] - v2['compressed_bytes']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate Totals",
            "",
            "| Item | Bytes | MB | Ratio vs payload |",
            "|---|---:|---:|---:|",
            f"| payload | {total_payload} | {bytes_to_mb(total_payload):.3f} | 1.000x |",
            f"| global pcodec | {total_global_pcodec} | {bytes_to_mb(total_global_pcodec):.3f} | {compression_ratio(total_payload, total_global_pcodec):.3f}x |",
            f"| per-channel pcodec sum | {total_per_channel_pcodec} | {bytes_to_mb(total_per_channel_pcodec):.3f} | {compression_ratio(total_payload, total_per_channel_pcodec):.3f}x |",
            f"| codec v2 | {total_v2} | {bytes_to_mb(total_v2):.3f} | {compression_ratio(total_payload, total_v2):.3f}x |",
            "",
            "## Codec Idea",
            "",
            "- Split the latent by channel and choose a mode independently for each channel.",
            "- Candidate modes are `raw_pcodec` and several `temporal_split_*` variants.",
            "- `temporal_split_*` stores the high byte directly, then uses high-byte continuity to split low-byte data into a stable stream and a changed stream.",
            "- Stable low-byte deltas go through `pcodec`; changed positions go through `zstd` or `pcodec`, depending on which is smaller for that channel.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    latent_paths = [Path(path) for path in (args.latent_paths or DEFAULT_LATENT_PATHS)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for latent_path in latent_paths:
        report = benchmark_one(
            latent_path,
            zstd_level=args.zstd_level,
            pcodec_level=args.pcodec_level,
            save_containers=args.save_containers,
            output_dir=args.output_dir,
        )
        reports.append(report)
        (args.output_dir / f"{report['sample']}_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            f"{report['sample']}: codec_v2={report['codec_v2']['compressed_mb']:.3f}MB "
            f"vs global_pcodec={next(row['compressed_mb'] for row in report['baselines'] if row['codec']=='pcodec_raw_global'):.3f}MB",
            flush=True,
        )

    aggregate = {
        "reports": reports,
        "totals": {
            "payload_bytes": int(sum(report["payload_bytes"] for report in reports)),
            "global_pcodec_bytes": int(
                sum(
                    next(row["compressed_bytes"] for row in report["baselines"] if row["codec"] == "pcodec_raw_global")
                    for report in reports
                )
            ),
            "per_channel_pcodec_bytes": int(
                sum(
                    next(row["compressed_bytes"] for row in report["baselines"] if row["codec"] == "pcodec_raw_per_channel_sum")
                    for report in reports
                )
            ),
            "codec_v2_bytes": int(sum(report["codec_v2"]["compressed_bytes"] for report in reports)),
        },
    }
    (args.output_dir / "aggregate_report.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    write_summary(reports, args.output_dir / "summary.md")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
