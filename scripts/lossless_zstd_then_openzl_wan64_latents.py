#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import openzl.ext as zl
import zstandard as zstd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compress Wan2.2 latent .pt files with lossless zstd, then compress the zstd"
            " payload again with lossless openzl generic serial compression."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/lossless_zstd_then_openzl_wan64"),
    )
    parser.add_argument(
        "--repo-report-root",
        type=Path,
        default=Path("/root/LatentsCompress/examples/vbench_codec/lossless_zstd_then_openzl_wan64"),
    )
    parser.add_argument("--zstd-level", type=int, default=19)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mb_decimal(nbytes: int) -> float:
    return round(nbytes / 1_000_000, 6)


def mib_binary(nbytes: int) -> float:
    return round(nbytes / (1024 * 1024), 6)


def build_openzl_contexts() -> tuple[zl.CCtx, zl.DCtx]:
    compressor = zl.Compressor()
    graph = zl.graphs.Compress()(compressor)
    compressor.select_starting_graph(graph)
    cctx = zl.CCtx()
    cctx.ref_compressor(compressor)
    cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    dctx = zl.DCtx()
    return cctx, dctx


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(summary: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Wan2.2 64 Latents Lossless Zstd Then OpenZL Report")
    lines.append("")
    lines.append(
        "This report first compresses each Wan2.2 latent `.pt` file with lossless `zstd`,"
        " then compresses the resulting zstd byte stream again with lossless `openzl`"
        " generic serial compression."
    )
    lines.append("")
    lines.append("Properties:")
    lines.append("")
    lines.append("- stage 1: lossless `zstd` on original `.pt` bytes")
    lines.append("- stage 2: lossless `openzl` on the `zstd` payload bytes")
    lines.append("- verification: `openzl -> zstd -> original` round-trip checked for every sample")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key in (
        "sample_count",
        "zstd_level",
        "total_original_mb",
        "total_zstd_mb",
        "total_zstd_then_openzl_mb",
        "total_saved_vs_original_mb",
        "total_saved_vs_zstd_mb",
        "total_ratio_original_to_zstd_then_openzl",
        "mean_original_mb",
        "mean_zstd_mb",
        "mean_zstd_then_openzl_mb",
        "mean_delta_vs_zstd_mb",
        "all_verified_lossless",
    ):
        lines.append(f"- {key}: `{summary[key]}`")
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    lines.append(
        "| name | original_mb | zstd_mb | zstd_then_openzl_mb | delta_vs_zstd_mb | "
        "saved_vs_original_mb | ratio_original_to_final | verified_lossless |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| `{row['name']}` | `{row['original_mb']}` | `{row['zstd_mb']}` | "
            f"`{row['zstd_then_openzl_mb']}` | `{row['delta_vs_zstd_mb']}` | "
            f"`{row['saved_vs_original_mb']}` | `{row['ratio_original_to_final']}` | "
            f"`{row['verified_lossless']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    latents = sorted(args.input_dir.glob("*.pt"))
    if args.limit is not None:
        latents = latents[: args.limit]
    if not latents:
        raise SystemExit(f"No latent files found in {args.input_dir}")

    report_root = args.output_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    args.repo_report_root.mkdir(parents=True, exist_ok=True)

    zstd_cctx = zstd.ZstdCompressor(level=args.zstd_level)
    zstd_dctx = zstd.ZstdDecompressor()

    rows: list[dict[str, Any]] = []
    total_original = 0
    total_zstd = 0
    total_final = 0
    all_verified = True

    for latent_path in latents:
        cctx, dctx = build_openzl_contexts()
        raw_blob = latent_path.read_bytes()
        zstd_blob = zstd_cctx.compress(raw_blob)
        openzl_blob = cctx.compress([zl.Input(zl.Type.Serial, zstd_blob)])
        restored_zstd_blob = dctx.decompress(openzl_blob)[0].content.as_bytes()
        restored_raw_blob = zstd_dctx.decompress(restored_zstd_blob)

        verified = restored_zstd_blob == zstd_blob and restored_raw_blob == raw_blob
        all_verified = all_verified and verified

        original_bytes = len(raw_blob)
        zstd_bytes = len(zstd_blob)
        final_bytes = len(openzl_blob)
        total_original += original_bytes
        total_zstd += zstd_bytes
        total_final += final_bytes

        rows.append(
            {
                "name": latent_path.stem,
                "latent_path": str(latent_path),
                "original_bytes": original_bytes,
                "zstd_bytes": zstd_bytes,
                "zstd_then_openzl_bytes": final_bytes,
                "delta_vs_zstd_bytes": final_bytes - zstd_bytes,
                "saved_vs_original_bytes": original_bytes - final_bytes,
                "original_mb": mb_decimal(original_bytes),
                "zstd_mb": mb_decimal(zstd_bytes),
                "zstd_then_openzl_mb": mb_decimal(final_bytes),
                "delta_vs_zstd_mb": mb_decimal(final_bytes - zstd_bytes),
                "saved_vs_original_mb": mb_decimal(original_bytes - final_bytes),
                "original_mib": mib_binary(original_bytes),
                "zstd_mib": mib_binary(zstd_bytes),
                "zstd_then_openzl_mib": mib_binary(final_bytes),
                "ratio_original_to_zstd": round(original_bytes / zstd_bytes, 6),
                "ratio_original_to_final": round(original_bytes / final_bytes, 6),
                "delta_vs_zstd_percent": round(((final_bytes - zstd_bytes) / zstd_bytes) * 100.0, 6),
                "saved_vs_original_percent": round(((original_bytes - final_bytes) / original_bytes) * 100.0, 6),
                "original_sha256": sha256_bytes(raw_blob),
                "restored_sha256": sha256_bytes(restored_raw_blob),
                "verified_lossless": verified,
            }
        )

    summary = {
        "input_dir": str(args.input_dir),
        "output_root": str(args.output_root),
        "sample_count": len(rows),
        "zstd_level": args.zstd_level,
        "total_original_bytes": total_original,
        "total_zstd_bytes": total_zstd,
        "total_zstd_then_openzl_bytes": total_final,
        "total_original_mb": mb_decimal(total_original),
        "total_zstd_mb": mb_decimal(total_zstd),
        "total_zstd_then_openzl_mb": mb_decimal(total_final),
        "total_saved_vs_original_bytes": total_original - total_final,
        "total_saved_vs_original_mb": mb_decimal(total_original - total_final),
        "total_saved_vs_zstd_bytes": total_zstd - total_final,
        "total_saved_vs_zstd_mb": mb_decimal(total_zstd - total_final),
        "total_ratio_original_to_zstd_then_openzl": round(total_original / total_final, 6),
        "mean_original_mb": round(sum(r["original_mb"] for r in rows) / len(rows), 6),
        "mean_zstd_mb": round(sum(r["zstd_mb"] for r in rows) / len(rows), 6),
        "mean_zstd_then_openzl_mb": round(sum(r["zstd_then_openzl_mb"] for r in rows) / len(rows), 6),
        "mean_delta_vs_zstd_mb": round(sum(r["delta_vs_zstd_mb"] for r in rows) / len(rows), 6),
        "all_verified_lossless": all_verified,
    }

    json_report = {"summary": summary, "rows": rows}
    json_path = report_root / "wan64_lossless_zstd_then_openzl_report.json"
    csv_path = report_root / "wan64_lossless_zstd_then_openzl_report.csv"
    md_path = report_root / "wan64_lossless_zstd_then_openzl_report.md"
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_md(summary, rows, md_path)

    for src in (json_path, csv_path, md_path):
        dst = args.repo_report_root / src.name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
