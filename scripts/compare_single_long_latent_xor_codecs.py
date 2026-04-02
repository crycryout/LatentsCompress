#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pcodec
import pcodec.standalone as ps
import torch
import zstandard as zstd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate block-wise base+XOR reversible transforms for one long-video latent,"
            " then compress the transformed payload with zstd and pcodec."
        )
    )
    parser.add_argument(
        "--latent-path",
        type=Path,
        default=Path(
            "/workspace/video_bench/skyreels_v2_vbench2_60s_720p/latents/subject_consistency/subject_consistency_000_5eaae1c7.pt"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/video_bench/codec_compare/skyreels_long_xor"),
    )
    parser.add_argument("--zstd-level", type=int, default=19)
    parser.add_argument("--pcodec-level", type=int, default=pcodec.DEFAULT_COMPRESSION_LEVEL)
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64],
    )
    return parser.parse_args()


def bytes_to_mb(n: int) -> float:
    return round(n / 1_000_000, 6)


def bytes_to_mib(n: int) -> float:
    return round(n / (1024 * 1024), 6)


def bshuffle_u16(arr: np.ndarray) -> bytes:
    raw = np.ascontiguousarray(arr).view(np.uint8).reshape(-1, 2)
    return raw.T.copy().reshape(-1).tobytes()


def xor_with_block_base_u16(seq_tchw_u16: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(seq_tchw_u16, dtype=np.uint16)
    t = seq_tchw_u16.shape[0]
    for start in range(0, t, window):
        end = min(start + window, t)
        base = seq_tchw_u16[start]
        out[start] = base
        if end - start > 1:
            out[start + 1 : end] = np.bitwise_xor(seq_tchw_u16[start + 1 : end], base)
    return out


def inverse_xor_with_block_base_u16(seq_tchw_u16: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(seq_tchw_u16, dtype=np.uint16)
    t = seq_tchw_u16.shape[0]
    for start in range(0, t, window):
        end = min(start + window, t)
        base = seq_tchw_u16[start]
        out[start] = base
        if end - start > 1:
            out[start + 1 : end] = np.bitwise_xor(seq_tchw_u16[start + 1 : end], base)
    return out


def write_md(report: dict, path: Path) -> None:
    lines = [
        "# SkyReels Long Latent XOR Compression Experiment",
        "",
        f"- latent: `{report['latent_path']}`",
        f"- latent shape: `{report['latent_shape']}`",
        f"- dtype: `{report['latent_dtype']}`",
        "",
        "## Baselines",
        "",
    ]
    lines.append("| method | bytes | MB | MiB |")
    lines.append("|---|---:|---:|---:|")
    for row in report["baselines"]:
        lines.append(
            f"| `{row['method']}` | `{row['bytes']}` | `{row['mb']}` | `{row['mib']}` |"
        )
    lines.append("")
    lines.append("## XOR Windows")
    lines.append("")
    lines.append(
        "| window | zstd_xor | zstd_xor_bshuffle | pcodec_xor | best | best_delta_vs_best_baseline_bytes | verified |"
    )
    lines.append("|---:|---:|---:|---:|---|---:|---:|")
    for row in report["xor_windows"]:
        lines.append(
            f"| `{row['window']}` | `{row['zstd_xor_bytes']}` | `{row['zstd_xor_bshuffle_bytes']}` | "
            f"`{row['pcodec_xor_bytes']}` | `{row['best_method']}` | "
            f"`{row['best_delta_vs_best_baseline_bytes']}` | `{row['verified_lossless']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    latent_obj = torch.load(args.latent_path, map_location="cpu")
    tensor = (
        latent_obj["latents"]
        if isinstance(latent_obj, dict) and "latents" in latent_obj
        else latent_obj
    )
    tensor = tensor.detach().cpu().contiguous()

    # Work on exact bfloat16 bit patterns through a uint16 view.
    seq_tchw_u16 = tensor.permute(1, 0, 2, 3).contiguous().view(torch.uint16).numpy()
    flat_u16 = seq_tchw_u16.reshape(-1)
    payload_bytes = flat_u16.tobytes()

    zc = zstd.ZstdCompressor(level=args.zstd_level)
    best_baseline = {
        "zstd_raw": len(zc.compress(payload_bytes)),
        "zstd_bshuffle": len(zc.compress(bshuffle_u16(seq_tchw_u16))),
        "pcodec_raw": len(
            ps.simple_compress(
                flat_u16,
                pcodec.ChunkConfig(compression_level=args.pcodec_level),
            )
        ),
    }

    baseline_rows = [
        {"method": name, "bytes": value, "mb": bytes_to_mb(value), "mib": bytes_to_mib(value)}
        for name, value in best_baseline.items()
    ]
    best_baseline_bytes = min(best_baseline.values())

    xor_rows = []
    for window in args.window_sizes:
        xored = xor_with_block_base_u16(seq_tchw_u16, window)
        restored = inverse_xor_with_block_base_u16(xored, window)
        verified = np.array_equal(restored, seq_tchw_u16)

        zstd_xor = len(zc.compress(xored.reshape(-1).tobytes()))
        zstd_xor_bshuffle = len(zc.compress(bshuffle_u16(xored)))
        pcodec_xor = len(
            ps.simple_compress(
                xored.reshape(-1),
                pcodec.ChunkConfig(compression_level=args.pcodec_level),
            )
        )

        methods = {
            "zstd_xor": zstd_xor,
            "zstd_xor_bshuffle": zstd_xor_bshuffle,
            "pcodec_xor": pcodec_xor,
        }
        best_method = min(methods, key=methods.get)
        best_bytes = methods[best_method]
        xor_rows.append(
            {
                "window": window,
                "zstd_xor_bytes": zstd_xor,
                "zstd_xor_bshuffle_bytes": zstd_xor_bshuffle,
                "pcodec_xor_bytes": pcodec_xor,
                "best_method": best_method,
                "best_bytes": best_bytes,
                "best_delta_vs_best_baseline_bytes": best_bytes - best_baseline_bytes,
                "verified_lossless": bool(verified),
            }
        )

    report = {
        "latent_path": str(args.latent_path),
        "latent_shape": list(tensor.shape),
        "latent_dtype": str(tensor.dtype).replace("torch.", ""),
        "payload_bytes": len(payload_bytes),
        "zstd_level": args.zstd_level,
        "pcodec_level": args.pcodec_level,
        "baselines": baseline_rows,
        "xor_windows": xor_rows,
    }

    out_dir = args.output_root / args.latent_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_md(report, md_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
