#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import openzl.ext as zl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress Wan2.2 latent .pt files directly with lossless openzl and compare against zstd."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/lossless_openzl_wan64"),
    )
    parser.add_argument(
        "--repo-report-root",
        type=Path,
        default=Path("/root/LatentsCompress/examples/vbench_codec/lossless_openzl_wan64"),
    )
    parser.add_argument(
        "--zstd-report-json",
        type=Path,
        default=Path("/root/LatentsCompress/examples/vbench_codec/lossless_zstd_wan64/wan64_lossless_zstd_report.json"),
    )
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
    lines.append("# Wan2.2 64 Latents Lossless OpenZL Report")
    lines.append("")
    lines.append(
        "This report compresses the original Wan2.2 latent `.pt` bytes directly with lossless"
        " `openzl` generic serial compression, and compares the resulting container size against"
        " the previously measured lossless `zstd` baseline."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key in (
        "sample_count",
        "total_original_mb",
        "total_openzl_mb",
        "total_zstd_mb",
        "total_saved_vs_original_mb",
        "total_delta_vs_zstd_mb",
        "total_ratio_original_to_openzl",
        "mean_original_mb",
        "mean_openzl_mb",
        "mean_zstd_mb",
        "mean_delta_vs_zstd_mb",
        "count_openzl_smaller_than_zstd",
        "count_openzl_larger_than_zstd",
        "count_openzl_equal_to_zstd",
        "all_verified_lossless",
    ):
        lines.append(f"- {key}: `{summary[key]}`")
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    lines.append(
        "| name | original_mb | openzl_mb | zstd_mb | delta_vs_zstd_mb | "
        "saved_vs_original_mb | ratio_original_to_openzl | verified_lossless |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| `{row['name']}` | `{row['original_mb']}` | `{row['openzl_mb']}` | "
            f"`{row['zstd_mb']}` | `{row['delta_vs_zstd_mb']}` | `{row['saved_vs_original_mb']}` | "
            f"`{row['ratio_original_to_openzl']}` | `{row['verified_lossless']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    latents = sorted(args.input_dir.glob("*.pt"))
    if args.limit is not None:
        latents = latents[: args.limit]
    if not latents:
        raise SystemExit(f"No latent files found in {args.input_dir}")

    if not args.zstd_report_json.exists():
        raise SystemExit(f"Missing zstd baseline report: {args.zstd_report_json}")
    zstd_report = json.loads(args.zstd_report_json.read_text(encoding="utf-8"))
    zstd_by_name = {row["name"]: row for row in zstd_report["rows"]}

    report_root = args.output_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    args.repo_report_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total_original = 0
    total_openzl = 0
    total_zstd = 0
    all_verified = True
    smaller = 0
    larger = 0
    equal = 0

    for latent_path in latents:
        name = latent_path.stem
        if name not in zstd_by_name:
            raise SystemExit(f"Missing zstd baseline row for {name}")

        raw_blob = latent_path.read_bytes()
        cctx, dctx = build_openzl_contexts()
        openzl_blob = cctx.compress([zl.Input(zl.Type.Serial, raw_blob)])
        restored_blob = dctx.decompress(openzl_blob)[0].content.as_bytes()

        verified = restored_blob == raw_blob
        all_verified = all_verified and verified

        original_bytes = len(raw_blob)
        openzl_bytes = len(openzl_blob)
        zstd_bytes = int(zstd_by_name[name]["compressed_bytes"])
        delta_vs_zstd = openzl_bytes - zstd_bytes

        total_original += original_bytes
        total_openzl += openzl_bytes
        total_zstd += zstd_bytes

        if delta_vs_zstd < 0:
            smaller += 1
        elif delta_vs_zstd > 0:
            larger += 1
        else:
            equal += 1

        rows.append(
            {
                "name": name,
                "latent_path": str(latent_path),
                "original_bytes": original_bytes,
                "openzl_bytes": openzl_bytes,
                "zstd_bytes": zstd_bytes,
                "delta_vs_zstd_bytes": delta_vs_zstd,
                "saved_vs_original_bytes": original_bytes - openzl_bytes,
                "original_mb": mb_decimal(original_bytes),
                "openzl_mb": mb_decimal(openzl_bytes),
                "zstd_mb": mb_decimal(zstd_bytes),
                "delta_vs_zstd_mb": mb_decimal(delta_vs_zstd),
                "saved_vs_original_mb": mb_decimal(original_bytes - openzl_bytes),
                "original_mib": mib_binary(original_bytes),
                "openzl_mib": mib_binary(openzl_bytes),
                "zstd_mib": mib_binary(zstd_bytes),
                "ratio_original_to_openzl": round(original_bytes / openzl_bytes, 6),
                "delta_vs_zstd_percent": round((delta_vs_zstd / zstd_bytes) * 100.0, 6),
                "saved_vs_original_percent": round(((original_bytes - openzl_bytes) / original_bytes) * 100.0, 6),
                "original_sha256": sha256_bytes(raw_blob),
                "restored_sha256": sha256_bytes(restored_blob),
                "verified_lossless": verified,
            }
        )

    summary = {
        "input_dir": str(args.input_dir),
        "output_root": str(args.output_root),
        "sample_count": len(rows),
        "total_original_bytes": total_original,
        "total_openzl_bytes": total_openzl,
        "total_zstd_bytes": total_zstd,
        "total_original_mb": mb_decimal(total_original),
        "total_openzl_mb": mb_decimal(total_openzl),
        "total_zstd_mb": mb_decimal(total_zstd),
        "total_saved_vs_original_bytes": total_original - total_openzl,
        "total_saved_vs_original_mb": mb_decimal(total_original - total_openzl),
        "total_delta_vs_zstd_bytes": total_openzl - total_zstd,
        "total_delta_vs_zstd_mb": mb_decimal(total_openzl - total_zstd),
        "total_ratio_original_to_openzl": round(total_original / total_openzl, 6),
        "mean_original_mb": round(sum(r["original_mb"] for r in rows) / len(rows), 6),
        "mean_openzl_mb": round(sum(r["openzl_mb"] for r in rows) / len(rows), 6),
        "mean_zstd_mb": round(sum(r["zstd_mb"] for r in rows) / len(rows), 6),
        "mean_delta_vs_zstd_mb": round(sum(r["delta_vs_zstd_mb"] for r in rows) / len(rows), 6),
        "count_openzl_smaller_than_zstd": smaller,
        "count_openzl_larger_than_zstd": larger,
        "count_openzl_equal_to_zstd": equal,
        "all_verified_lossless": all_verified,
    }

    json_report = {"summary": summary, "rows": rows}
    json_path = report_root / "wan64_lossless_openzl_report.json"
    csv_path = report_root / "wan64_lossless_openzl_report.csv"
    md_path = report_root / "wan64_lossless_openzl_report.md"
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_md(summary, rows, md_path)

    for src in (json_path, csv_path, md_path):
        dst = args.repo_report_root / src.name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
