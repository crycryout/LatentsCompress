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


TORCH_UINT_VIEW_BY_ITEMSIZE = {
    1: torch.uint8,
    2: torch.uint16,
    4: torch.uint32,
    8: torch.uint64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate reversible block-wise base+XOR transforms on the 64 Wan2.2 5s latents,"
            " then compress with zstd and pcodec."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("/workspace/video_bench/codec_compare/wan64_xor_codecs.json"),
    )
    parser.add_argument("--zstd-level", type=int, default=19)
    parser.add_argument("--pcodec-level", type=int, default=pcodec.DEFAULT_COMPRESSION_LEVEL)
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[2, 4, 8, 16])
    return parser.parse_args()


def bshuffle_words(arr: np.ndarray) -> bytes:
    itemsize = arr.dtype.itemsize
    raw = np.ascontiguousarray(arr).view(np.uint8).reshape(-1, itemsize)
    return raw.T.copy().reshape(-1).tobytes()


def xor_with_block_base(seq_tchw_u: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(seq_tchw_u, dtype=seq_tchw_u.dtype)
    t = seq_tchw_u.shape[0]
    for start in range(0, t, window):
        end = min(start + window, t)
        base = seq_tchw_u[start]
        out[start] = base
        if end - start > 1:
            out[start + 1 : end] = np.bitwise_xor(seq_tchw_u[start + 1 : end], base)
    return out


def inverse_xor_with_block_base(seq_tchw_u: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(seq_tchw_u, dtype=seq_tchw_u.dtype)
    t = seq_tchw_u.shape[0]
    for start in range(0, t, window):
        end = min(start + window, t)
        base = seq_tchw_u[start]
        out[start] = base
        if end - start > 1:
            out[start + 1 : end] = np.bitwise_xor(seq_tchw_u[start + 1 : end], base)
    return out


def init_compare_table(windows: list[int]) -> dict[str, dict[int, dict[str, int]]]:
    return {
        "zstd_xor": {w: {"better": 0, "equal": 0, "worse": 0} for w in windows},
        "zstd_xor_bshuffle": {w: {"better": 0, "equal": 0, "worse": 0} for w in windows},
        "pcodec_xor": {w: {"better": 0, "equal": 0, "worse": 0} for w in windows},
    }


def update_compare(counter: dict[str, dict[int, dict[str, int]]], method: str, window: int, value: int, baseline: int) -> None:
    if value < baseline:
        counter[method][window]["better"] += 1
    elif value > baseline:
        counter[method][window]["worse"] += 1
    else:
        counter[method][window]["equal"] += 1


def main() -> None:
    args = parse_args()
    latents = sorted(args.input_dir.glob("*.pt"))
    if not latents:
        raise SystemExit(f"No latent files found in {args.input_dir}")

    zc = zstd.ZstdCompressor(level=args.zstd_level)
    pcodec_cfg = pcodec.ChunkConfig(compression_level=args.pcodec_level)

    totals = {
        "baseline": {"zstd_raw": 0, "zstd_bshuffle": 0, "pcodec_raw": 0},
        "xor": {
            str(w): {"zstd_xor": 0, "zstd_xor_bshuffle": 0, "pcodec_xor": 0}
            for w in args.window_sizes
        },
    }
    comparisons = init_compare_table(args.window_sizes)
    rows = []

    for idx, latent_path in enumerate(latents, start=1):
        latent_obj = torch.load(latent_path, map_location="cpu")
        tensor = (
            latent_obj["latents"]
            if isinstance(latent_obj, dict) and "latents" in latent_obj
            else latent_obj
        )
        tensor = tensor.detach().cpu().contiguous()
        itemsize = tensor.element_size()
        uint_view_dtype = TORCH_UINT_VIEW_BY_ITEMSIZE[itemsize]
        seq = tensor.permute(1, 0, 2, 3).contiguous().view(uint_view_dtype).numpy()
        flat = seq.reshape(-1)

        zstd_raw = len(zc.compress(flat.tobytes()))
        zstd_bshuffle = len(zc.compress(bshuffle_words(seq)))
        pcodec_raw = len(ps.simple_compress(flat, pcodec_cfg))

        totals["baseline"]["zstd_raw"] += zstd_raw
        totals["baseline"]["zstd_bshuffle"] += zstd_bshuffle
        totals["baseline"]["pcodec_raw"] += pcodec_raw

        row = {
            "name": latent_path.stem,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "time_major_word_dtype": str(seq.dtype),
            "baselines": {
                "zstd_raw": zstd_raw,
                "zstd_bshuffle": zstd_bshuffle,
                "pcodec_raw": pcodec_raw,
            },
            "windows": {},
        }

        for window in args.window_sizes:
            xored = xor_with_block_base(seq, window)
            restored = inverse_xor_with_block_base(xored, window)
            verified = bool(np.array_equal(restored, seq))

            zstd_xor = len(zc.compress(xored.reshape(-1).tobytes()))
            zstd_xor_bshuffle = len(zc.compress(bshuffle_words(xored)))
            pcodec_xor = len(ps.simple_compress(xored.reshape(-1), pcodec_cfg))

            totals["xor"][str(window)]["zstd_xor"] += zstd_xor
            totals["xor"][str(window)]["zstd_xor_bshuffle"] += zstd_xor_bshuffle
            totals["xor"][str(window)]["pcodec_xor"] += pcodec_xor

            update_compare(comparisons, "zstd_xor", window, zstd_xor, zstd_raw)
            update_compare(comparisons, "zstd_xor_bshuffle", window, zstd_xor_bshuffle, zstd_bshuffle)
            update_compare(comparisons, "pcodec_xor", window, pcodec_xor, pcodec_raw)

            row["windows"][str(window)] = {
                "zstd_xor": zstd_xor,
                "zstd_xor_bshuffle": zstd_xor_bshuffle,
                "pcodec_xor": pcodec_xor,
                "verified_lossless": verified,
            }

        rows.append(row)
        if idx % 8 == 0:
            print(f"done {idx}/{len(latents)}", flush=True)

    summary = {
        "input_dir": str(args.input_dir),
        "sample_count": len(rows),
        "zstd_level": args.zstd_level,
        "pcodec_level": args.pcodec_level,
        "window_sizes": args.window_sizes,
        "baseline_totals": totals["baseline"],
        "xor_totals": totals["xor"],
        "comparisons_vs_baseline": comparisons,
        "xor_total_delta_vs_baseline": {
            str(w): {
                "zstd_xor_vs_zstd_raw": totals["xor"][str(w)]["zstd_xor"] - totals["baseline"]["zstd_raw"],
                "zstd_xor_bshuffle_vs_zstd_bshuffle": totals["xor"][str(w)]["zstd_xor_bshuffle"] - totals["baseline"]["zstd_bshuffle"],
                "pcodec_xor_vs_pcodec_raw": totals["xor"][str(w)]["pcodec_xor"] - totals["baseline"]["pcodec_raw"],
            }
            for w in args.window_sizes
        },
        "all_verified_lossless": all(
            row["windows"][str(w)]["verified_lossless"] for row in rows for w in args.window_sizes
        ),
    }

    report = {"summary": summary, "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()
