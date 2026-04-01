#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Losslessly compress Wan2.2 saved latent .pt files with zstd and report size comparisons."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/lossless_zstd_wan64"),
    )
    parser.add_argument(
        "--repo-report-root",
        type=Path,
        default=Path("/root/LatentsCompress/examples/vbench_codec/lossless_zstd_wan64"),
    )
    parser.add_argument("--zstd-level", type=int, default=19)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-containers", action="store_true")
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def zstd_compress_bytes(raw: bytes, level: int) -> bytes:
    proc = subprocess.run(
        ["zstd", "-q", f"-{level}", "-c"],
        input=raw,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def maybe_save_container(blob: bytes, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(blob)


def zstd_decompress_bytes(src: Path) -> bytes:
    proc = subprocess.run(
        ["zstd", "-q", "-d", "-c", str(src)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def mb_decimal(nbytes: int) -> float:
    return round(nbytes / 1_000_000, 6)


def mib_binary(nbytes: int) -> float:
    return round(nbytes / (1024 * 1024), 6)


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
    lines.append("# Wan2.2 64 Latents Lossless Zstd Report")
    lines.append("")
    lines.append("This report compresses the saved Wan2.2 latent `.pt` files directly with `zstd`.")
    lines.append("")
    lines.append("Properties:")
    lines.append("")
    lines.append("- compression target: original `.pt` latent file bytes")
    lines.append("- compression type: lossless `zstd`")
    lines.append("- verification: decompressed bytes checked against original bytes")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key in (
        "sample_count",
        "zstd_level",
        "total_original_bytes",
        "total_compressed_bytes",
        "total_original_mb",
        "total_compressed_mb",
        "total_bytes_saved",
        "total_mb_saved",
        "total_compression_ratio",
        "mean_original_mb",
        "mean_compressed_mb",
        "mean_bytes_saved_mb",
        "mean_compression_ratio",
        "all_verified_lossless",
    ):
        lines.append(f"- {key}: `{summary[key]}`")
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    lines.append("| name | original_mb | compressed_mb | saved_mb | compression_ratio | saved_percent | verified_lossless |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| `{row['name']}` | `{row['original_mb']}` | `{row['compressed_mb']}` | `{row['saved_mb']}` | "
            f"`{row['compression_ratio']}` | `{row['saved_percent']}` | `{row['verified_lossless']}` |"
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
    container_root = args.output_root / "containers"
    if args.save_containers:
        container_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total_original = 0
    total_compressed = 0
    all_verified = True

    for latent_path in latents:
        compressed_path = container_root / f"{latent_path.name}.zst"
        original_bytes = latent_path.stat().st_size
        original_blob = latent_path.read_bytes()
        compressed_blob = zstd_compress_bytes(original_blob, args.zstd_level)
        if args.save_containers and not (args.skip_existing and compressed_path.exists()):
            maybe_save_container(compressed_blob, compressed_path)
        compressed_bytes = len(compressed_blob)
        total_original += original_bytes
        total_compressed += compressed_bytes
        restored_blob = subprocess.run(
            ["zstd", "-q", "-d", "-c"],
            input=compressed_blob,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout
        verified = restored_blob == original_blob
        all_verified = all_verified and verified

        saved_bytes = original_bytes - compressed_bytes
        ratio = round(original_bytes / compressed_bytes, 6) if compressed_bytes else None
        saved_percent = round((saved_bytes / original_bytes) * 100.0, 4) if original_bytes else 0.0

        rows.append(
            {
                "name": latent_path.stem,
                "latent_path": str(latent_path),
                "compressed_path": str(compressed_path) if args.save_containers else None,
                "original_bytes": original_bytes,
                "compressed_bytes": compressed_bytes,
                "saved_bytes": saved_bytes,
                "original_mb": mb_decimal(original_bytes),
                "compressed_mb": mb_decimal(compressed_bytes),
                "saved_mb": mb_decimal(saved_bytes),
                "original_mib": mib_binary(original_bytes),
                "compressed_mib": mib_binary(compressed_bytes),
                "saved_mib": mib_binary(saved_bytes),
                "compression_ratio": ratio,
                "saved_percent": saved_percent,
                "original_sha256": sha256_bytes(original_blob),
                "restored_sha256": sha256_bytes(restored_blob),
                "verified_lossless": verified,
            }
        )

    summary = {
        "input_dir": str(args.input_dir),
        "output_root": str(args.output_root),
        "sample_count": len(rows),
        "zstd_level": args.zstd_level,
        "total_original_bytes": total_original,
        "total_compressed_bytes": total_compressed,
        "total_original_mb": mb_decimal(total_original),
        "total_compressed_mb": mb_decimal(total_compressed),
        "total_bytes_saved": total_original - total_compressed,
        "total_mb_saved": mb_decimal(total_original - total_compressed),
        "total_compression_ratio": round(total_original / total_compressed, 6),
        "mean_original_mb": round(sum(r["original_mb"] for r in rows) / len(rows), 6),
        "mean_compressed_mb": round(sum(r["compressed_mb"] for r in rows) / len(rows), 6),
        "mean_bytes_saved_mb": round(sum(r["saved_mb"] for r in rows) / len(rows), 6),
        "mean_compression_ratio": round(sum(r["compression_ratio"] for r in rows) / len(rows), 6),
        "all_verified_lossless": all_verified,
    }

    json_report = {
        "summary": summary,
        "rows": rows,
    }

    json_path = report_root / "wan64_lossless_zstd_report.json"
    csv_path = report_root / "wan64_lossless_zstd_report.csv"
    md_path = report_root / "wan64_lossless_zstd_report.md"
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_md(summary, rows, md_path)

    # Copy lightweight reports into repo for GitHub.
    repo_json = args.repo_report_root / json_path.name
    repo_csv = args.repo_report_root / csv_path.name
    repo_md = args.repo_report_root / md_path.name
    repo_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    repo_csv.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    repo_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
