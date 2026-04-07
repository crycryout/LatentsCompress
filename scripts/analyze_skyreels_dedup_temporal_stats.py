#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_LATENT_PATHS = [
    "/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/wingsuit_rescue_glacier_pullup/full_video_latents_dedup.pt",
    "/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/neon_hoverbike_chain_reaction/full_video_latents_dedup.pt",
    "/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/avalanche_snowmobile_bridge_escape/full_video_latents_dedup.pt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze temporal continuity of deduplicated SkyReels long-video latents, "
            "including bf16 component stability and channel/block-level continuity."
        )
    )
    parser.add_argument(
        "--latent-path",
        action="append",
        dest="latent_paths",
        default=[],
        help="Path to a deduplicated long-video latent `.pt`. Can be passed multiple times.",
    )
    parser.add_argument("--block-h", type=int, default=10)
    parser.add_argument("--block-w", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/LatentsCompress/examples/vbench_codec/skyreels_dedup_temporal_stats"),
    )
    return parser.parse_args()


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


def quantiles(arr: np.ndarray, ps: list[float]) -> dict[str, float]:
    return {f"p{int(p * 1000) / 10:g}": float(np.quantile(arr, p)) for p in ps}


def top_entries(items: list[dict[str, Any]], key: str, k: int, reverse: bool) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: item[key], reverse=reverse)[:k]


def bf16_component_stats(u16_tchw: np.ndarray) -> dict[str, Any]:
    sign = ((u16_tchw >> 15) & 0x1).astype(np.uint8)
    exponent = ((u16_tchw >> 7) & 0xFF).astype(np.uint8)
    mantissa = (u16_tchw & 0x7F).astype(np.uint8)

    sign_equal = sign[1:] == sign[:-1]
    exponent_equal = exponent[1:] == exponent[:-1]
    mantissa_equal = mantissa[1:] == mantissa[:-1]

    exponent_delta = exponent[1:].astype(np.int16) - exponent[:-1].astype(np.int16)
    mantissa_delta = mantissa[1:].astype(np.int16) - mantissa[:-1].astype(np.int16)
    mantissa_delta_same_exp = mantissa_delta[exponent_equal]
    mantissa_delta_changed_exp = mantissa_delta[~exponent_equal]

    return {
        "sign_equal_fraction": float(np.mean(sign_equal)),
        "exponent_equal_fraction": float(np.mean(exponent_equal)),
        "mantissa_equal_fraction": float(np.mean(mantissa_equal)),
        "exponent_change_fraction": float(np.mean(~exponent_equal)),
        "mantissa_change_fraction": float(np.mean(~mantissa_equal)),
        "sign_flip_fraction": float(np.mean(~sign_equal)),
        "exponent_delta_abs_mean": float(np.mean(np.abs(exponent_delta))),
        "mantissa_delta_abs_mean": float(np.mean(np.abs(mantissa_delta))),
        "mantissa_delta_abs_mean_given_same_exponent": float(np.mean(np.abs(mantissa_delta_same_exp))),
        "mantissa_delta_abs_mean_given_changed_exponent": float(np.mean(np.abs(mantissa_delta_changed_exp))),
        "exponent_delta_quantiles": quantiles(np.abs(exponent_delta).reshape(-1), [0.5, 0.9, 0.99]),
        "mantissa_delta_quantiles": quantiles(np.abs(mantissa_delta).reshape(-1), [0.5, 0.9, 0.99]),
        "mantissa_delta_same_exponent_quantiles": quantiles(
            np.abs(mantissa_delta_same_exp).reshape(-1), [0.5, 0.9, 0.99]
        ),
        "mantissa_delta_changed_exponent_quantiles": quantiles(
            np.abs(mantissa_delta_changed_exp).reshape(-1), [0.5, 0.9, 0.99]
        ),
        "component_raw_ranges": {
            "sign_unique": [int(v) for v in np.unique(sign)],
            "exponent_min": int(exponent.min()),
            "exponent_max": int(exponent.max()),
            "mantissa_min": int(mantissa.min()),
            "mantissa_max": int(mantissa.max()),
        },
    }


def block_stats(
    seq_tchw_f32: np.ndarray,
    u16_tchw: np.ndarray,
    *,
    block_h: int,
    block_w: int,
    top_k: int,
) -> dict[str, Any]:
    t, c, h, w = seq_tchw_f32.shape
    sign = ((u16_tchw >> 15) & 0x1).astype(np.uint8)
    exponent = ((u16_tchw >> 7) & 0xFF).astype(np.uint8)
    mantissa = (u16_tchw & 0x7F).astype(np.uint8)
    entries: list[dict[str, Any]] = []

    for channel in range(c):
        for y0 in range(0, h, block_h):
            y1 = min(y0 + block_h, h)
            for x0 in range(0, w, block_w):
                x1 = min(x0 + block_w, w)
                block = seq_tchw_f32[:, channel, y0:y1, x0:x1]
                block_u16 = u16_tchw[:, channel, y0:y1, x0:x1]
                block_sign = sign[:, channel, y0:y1, x0:x1]
                block_exp = exponent[:, channel, y0:y1, x0:x1]
                block_man = mantissa[:, channel, y0:y1, x0:x1]

                diff = block[1:] - block[:-1]
                abs_diff = np.abs(diff)
                a = block[:-1].reshape(t - 1, -1)
                b = block[1:].reshape(t - 1, -1)
                dot = np.sum(a * b, axis=1)
                na = np.linalg.norm(a, axis=1)
                nb = np.linalg.norm(b, axis=1)
                cosine = dot / np.clip(na * nb, 1e-12, None)

                entries.append(
                    {
                        "channel": channel,
                        "y0": y0,
                        "y1": y1,
                        "x0": x0,
                        "x1": x1,
                        "exact_u16_equal_fraction": float(np.mean(block_u16[1:] == block_u16[:-1])),
                        "sign_equal_fraction": float(np.mean(block_sign[1:] == block_sign[:-1])),
                        "exponent_equal_fraction": float(np.mean(block_exp[1:] == block_exp[:-1])),
                        "mantissa_equal_fraction": float(np.mean(block_man[1:] == block_man[:-1])),
                        "delta_mae": float(np.mean(abs_diff)),
                        "delta_rmse": float(np.sqrt(np.mean(diff * diff))),
                        "cosine_mean": float(np.mean(cosine)),
                        "cosine_min": float(np.min(cosine)),
                    }
                )

    return {
        "block_shape_hw": [block_h, block_w],
        "num_blocks_total": len(entries),
        "most_stable_by_delta_mae": top_entries(entries, "delta_mae", top_k, reverse=False),
        "least_stable_by_delta_mae": top_entries(entries, "delta_mae", top_k, reverse=True),
        "most_stable_by_exponent_equal": top_entries(entries, "exponent_equal_fraction", top_k, reverse=True),
        "least_stable_by_exponent_equal": top_entries(entries, "exponent_equal_fraction", top_k, reverse=False),
        "most_stable_by_exact_equal": top_entries(entries, "exact_u16_equal_fraction", top_k, reverse=True),
        "least_stable_by_exact_equal": top_entries(entries, "exact_u16_equal_fraction", top_k, reverse=False),
    }


def analyze_one(path: Path, *, block_h: int, block_w: int, top_k: int) -> dict[str, Any]:
    tensor_cthw = load_latent_tensor(path)
    seq_tchw_f32 = tensor_cthw.float().permute(1, 0, 2, 3).contiguous().numpy()
    u16_tchw = tensor_cthw.view(torch.uint16).permute(1, 0, 2, 3).contiguous().numpy().copy()

    diff = seq_tchw_f32[1:] - seq_tchw_f32[:-1]
    abs_diff = np.abs(diff)

    report = {
        "sample": path.parent.name,
        "latent_path": str(path),
        "shape_tchw": list(seq_tchw_f32.shape),
        "dtype": "bfloat16",
        "global": {
            "mean": float(seq_tchw_f32.mean()),
            "std": float(seq_tchw_f32.std()),
            "min": float(seq_tchw_f32.min()),
            "max": float(seq_tchw_f32.max()),
        },
        "adjacent_float": {
            "delta_mae_mean": float(np.mean(np.mean(abs_diff, axis=(1, 2, 3)))),
            "delta_rmse_mean": float(np.mean(np.sqrt(np.mean(diff * diff, axis=(1, 2, 3))))),
            "delta_abs_quantiles": quantiles(abs_diff.reshape(-1), [0.5, 0.9, 0.95, 0.99]),
        },
        "bf16_components": bf16_component_stats(u16_tchw),
        "channel_block_stats": block_stats(
            seq_tchw_f32,
            u16_tchw,
            block_h=block_h,
            block_w=block_w,
            top_k=top_k,
        ),
    }
    return report


def write_summary(reports: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# SkyReels Dedup Temporal Deep Stats",
        "",
        "## BF16 Component Stability",
        "",
        "| Sample | sign equal | exponent equal | mantissa equal | sign flip | exponent abs-delta mean | mantissa abs-delta mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        comp = report["bf16_components"]
        lines.append(
            "| "
            f"{report['sample']} | "
            f"{comp['sign_equal_fraction']:.4f} | "
            f"{comp['exponent_equal_fraction']:.4f} | "
            f"{comp['mantissa_equal_fraction']:.4f} | "
            f"{comp['sign_flip_fraction']:.4f} | "
            f"{comp['exponent_delta_abs_mean']:.4f} | "
            f"{comp['mantissa_delta_abs_mean']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Key Interpretation",
            "",
            "- `sign` and especially `exponent` are much more stable over time than the full 16-bit value.",
            "- `mantissa` changes much more often, so literal duplication is rare even when semantic continuity is high.",
            "- The useful compression target is therefore conditional continuity: predict mantissa or low-byte behavior only when higher-level state stays stable.",
            "",
            "## Most Stable Blocks By Delta MAE",
            "",
        ]
    )

    for report in reports:
        lines.append(f"### {report['sample']}")
        lines.append("")
        lines.append("| channel | y0:y1 | x0:x1 | delta MAE | exponent equal | exact equal | cosine mean |")
        lines.append("|---:|---|---|---:|---:|---:|---:|")
        for row in report["channel_block_stats"]["most_stable_by_delta_mae"]:
            lines.append(
                "| "
                f"{row['channel']} | "
                f"{row['y0']}:{row['y1']} | "
                f"{row['x0']}:{row['x1']} | "
                f"{row['delta_mae']:.4f} | "
                f"{row['exponent_equal_fraction']:.4f} | "
                f"{row['exact_u16_equal_fraction']:.4f} | "
                f"{row['cosine_mean']:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Least Stable Blocks By Delta MAE",
            "",
        ]
    )
    for report in reports:
        lines.append(f"### {report['sample']}")
        lines.append("")
        lines.append("| channel | y0:y1 | x0:x1 | delta MAE | exponent equal | exact equal | cosine mean |")
        lines.append("|---:|---|---|---:|---:|---:|---:|")
        for row in report["channel_block_stats"]["least_stable_by_delta_mae"]:
            lines.append(
                "| "
                f"{row['channel']} | "
                f"{row['y0']}:{row['y1']} | "
                f"{row['x0']}:{row['x1']} | "
                f"{row['delta_mae']:.4f} | "
                f"{row['exponent_equal_fraction']:.4f} | "
                f"{row['exact_u16_equal_fraction']:.4f} | "
                f"{row['cosine_mean']:.4f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    latent_paths = [Path(path) for path in (args.latent_paths or DEFAULT_LATENT_PATHS)]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for latent_path in latent_paths:
        report = analyze_one(
            latent_path,
            block_h=args.block_h,
            block_w=args.block_w,
            top_k=args.top_k,
        )
        reports.append(report)
        (args.output_dir / f"{report['sample']}_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            f"{report['sample']}: exponent_equal={report['bf16_components']['exponent_equal_fraction']:.4f} "
            f"mantissa_equal={report['bf16_components']['mantissa_equal_fraction']:.4f}",
            flush=True,
        )

    aggregate = {
        "block_shape_hw": [args.block_h, args.block_w],
        "top_k": args.top_k,
        "reports": reports,
    }
    (args.output_dir / "aggregate_report.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    write_summary(reports, args.output_dir / "summary.md")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
