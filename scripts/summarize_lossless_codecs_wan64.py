#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Wan64 lossless codec results across zstd, openzl, and pcodec."
    )
    parser.add_argument(
        "--zstd-report-json",
        type=Path,
        default=Path("/root/LatentsCompress/examples/vbench_codec/lossless_zstd_wan64/wan64_lossless_zstd_report.json"),
    )
    parser.add_argument(
        "--openzl-report-json",
        type=Path,
        default=Path("/root/LatentsCompress/examples/vbench_codec/lossless_openzl_wan64/wan64_lossless_openzl_report.json"),
    )
    parser.add_argument(
        "--pcodec-report-json",
        type=Path,
        default=Path("/root/LatentsCompress/examples/vbench_codec/lossless_pcodec_wan64/wan64_lossless_pcodec_report.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/lossless_codec_comparison_wan64"),
    )
    parser.add_argument(
        "--repo-report-root",
        type=Path,
        default=Path("/root/LatentsCompress/examples/vbench_codec/lossless_codec_comparison_wan64"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(summary: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Wan2.2 64 Latents Lossless Codec Comparison")
    lines.append("")
    lines.append("This report compares three lossless latent compression routes:")
    lines.append("")
    lines.append("- `zstd` on the serialized `.pt` bytes")
    lines.append("- `openzl` on the original latent `.pt` bytes")
    lines.append("- `pcodec` on the numeric latent tensor payload")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Per-sample quick table")
    lines.append("")
    lines.append("| name | zstd_mb | openzl_mb | pcodec_mb | best_codec |")
    lines.append("|---|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| `{row['name']}` | `{row['zstd_mb']}` | `{row['openzl_mb']}` | `{row['pcodec_mb']}` | `{row['best_codec']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    zstd_report = read_json(args.zstd_report_json)
    openzl_report = read_json(args.openzl_report_json)
    pcodec_report = read_json(args.pcodec_report_json)

    zstd_rows = {r["name"]: r for r in zstd_report["rows"]}
    openzl_rows = {r["name"]: r for r in openzl_report["rows"]}
    pcodec_rows = {r["name"]: r for r in pcodec_report["rows"]}

    common_names = sorted(set(zstd_rows) & set(openzl_rows) & set(pcodec_rows))
    rows: list[dict[str, Any]] = []
    best_counts = {"zstd": 0, "openzl": 0, "pcodec": 0}

    for name in common_names:
        zstd_mb = float(zstd_rows[name]["compressed_mb"])
        openzl_mb = float(openzl_rows[name]["openzl_mb"])
        pcodec_mb = float(pcodec_rows[name]["pcodec_mb"])
        best_codec = min(
            [("zstd", zstd_mb), ("openzl", openzl_mb), ("pcodec", pcodec_mb)],
            key=lambda x: x[1],
        )[0]
        best_counts[best_codec] += 1
        rows.append(
            {
                "name": name,
                "zstd_mb": zstd_mb,
                "openzl_mb": openzl_mb,
                "pcodec_mb": pcodec_mb,
                "best_codec": best_codec,
            }
        )

    zstd_total = float(zstd_report["summary"]["total_compressed_mb"])
    openzl_total = float(openzl_report["summary"]["total_openzl_mb"])
    pcodec_total = float(pcodec_report["summary"]["total_pcodec_mb"])
    summary = {
        "sample_count": len(rows),
        "zstd_total_mb": zstd_total,
        "openzl_total_mb": openzl_total,
        "pcodec_total_mb": pcodec_total,
        "pcodec_minus_zstd_mb": round(pcodec_total - zstd_total, 6),
        "pcodec_minus_openzl_mb": round(pcodec_total - openzl_total, 6),
        "best_codec_counts": best_counts,
        "all_pcodec_lossless": pcodec_report["summary"]["all_verified_lossless"],
        "all_zstd_lossless": zstd_report["summary"]["all_verified_lossless"],
        "all_openzl_lossless": openzl_report["summary"]["all_verified_lossless"],
    }

    report_root = args.output_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    args.repo_report_root.mkdir(parents=True, exist_ok=True)
    json_path = report_root / "wan64_lossless_codec_comparison_report.json"
    csv_path = report_root / "wan64_lossless_codec_comparison_report.csv"
    md_path = report_root / "wan64_lossless_codec_comparison_report.md"
    json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_md(summary, rows, md_path)

    for src in (json_path, csv_path, md_path):
        dst = args.repo_report_root / src.name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
