#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pcodec
import pcodec.wrapped as pw
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compress Wan2.2 latent tensors directly with lossless pcodec and compare against"
            " existing zstd/openzl baselines."
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
        default=Path("/tmp/lossless_pcodec_wan64"),
    )
    parser.add_argument(
        "--repo-report-root",
        type=Path,
        default=Path("/root/LatentsCompress/examples/vbench_codec/lossless_pcodec_wan64"),
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
    parser.add_argument("--compression-level", type=int, default=pcodec.DEFAULT_COMPRESSION_LEVEL)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    lines.append("# Wan2.2 64 Latents Lossless PCodec Report")
    lines.append("")
    lines.append(
        "This report compresses the original Wan2.2 latent tensor payload directly with lossless"
        " `pcodec`, then compares the resulting size against the previously measured lossless"
        " `zstd` and `openzl` baselines."
    )
    lines.append("")
    lines.append("Important note:")
    lines.append("")
    lines.append(
        "- `pcodec` operates on the numeric latent tensor payload (`latents`) rather than the full"
        " serialized `.pt` container."
    )
    lines.append(
        "- For these files the `.pt` metadata overhead is tiny, so the comparison is still useful,"
        " but it is not a byte-identical container-vs-container benchmark."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key in (
        "sample_count",
        "compression_level",
        "total_original_pt_mb",
        "total_original_tensor_mb",
        "total_pcodec_mb",
        "total_zstd_mb",
        "total_openzl_mb",
        "total_pcodec_saved_vs_tensor_mb",
        "total_pcodec_delta_vs_zstd_mb",
        "total_pcodec_delta_vs_openzl_mb",
        "mean_original_tensor_mb",
        "mean_pcodec_mb",
        "mean_zstd_mb",
        "mean_openzl_mb",
        "mean_pcodec_delta_vs_zstd_mb",
        "mean_pcodec_delta_vs_openzl_mb",
        "count_pcodec_smaller_than_zstd",
        "count_pcodec_smaller_than_openzl",
        "all_verified_lossless",
    ):
        lines.append(f"- {key}: `{summary[key]}`")
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    lines.append(
        "| name | tensor_mb | pcodec_mb | zstd_mb | openzl_mb | "
        "pcodec_vs_zstd_mb | pcodec_vs_openzl_mb | verified_lossless |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| `{row['name']}` | `{row['original_tensor_mb']}` | `{row['pcodec_mb']}` | "
            f"`{row['zstd_mb']}` | `{row['openzl_mb']}` | `{row['pcodec_delta_vs_zstd_mb']}` | "
            f"`{row['pcodec_delta_vs_openzl_mb']}` | `{row['verified_lossless']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pcodec_dtype_of(np_dtype: np.dtype) -> str:
    kind = np_dtype.kind
    item_bits = np_dtype.itemsize * 8
    if kind == "f":
        return f"f{item_bits}"
    if kind == "i":
        return f"i{item_bits}"
    if kind == "u":
        return f"u{item_bits}"
    raise ValueError(f"Unsupported dtype for pcodec: {np_dtype}")


def compress_pcodec(arr: np.ndarray, compression_level: int) -> tuple[bytes, list[int]]:
    file_compressor = pw.FileCompressor()
    config = pcodec.ChunkConfig(compression_level=compression_level)
    chunk_compressor = file_compressor.chunk_compressor(arr, config)
    pages = chunk_compressor.n_per_page()
    parts = [file_compressor.write_header(), chunk_compressor.write_meta()]
    for page_idx in range(len(pages)):
        parts.append(chunk_compressor.write_page(page_idx))
    return b"".join(parts), pages


def decompress_pcodec(blob: bytes, dtype_str: str, page_counts: list[int]) -> np.ndarray:
    file_decompressor, header_len = pw.FileDecompressor.new(blob)
    chunk_decompressor, meta_len = file_decompressor.chunk_decompressor(blob[header_len:], dtype_str)
    offset = header_len + meta_len
    np_dtype = np.dtype(dtype_str.replace("f", "float").replace("i", "int").replace("u", "uint"))
    pages: list[np.ndarray] = []
    for page_n in page_counts:
        dst = np.empty(page_n, dtype=np_dtype)
        _progress, nread = chunk_decompressor.read_page_into(blob[offset:], page_n, dst)
        pages.append(dst)
        offset += nread
    return np.concatenate(pages)


def load_baseline_rows(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row[key_field]: row for row in data["rows"]}


def main() -> None:
    args = parse_args()
    latents = sorted(args.input_dir.glob("*.pt"))
    if args.limit is not None:
        latents = latents[: args.limit]
    if not latents:
        raise SystemExit(f"No latent files found in {args.input_dir}")

    zstd_by_name = load_baseline_rows(args.zstd_report_json, "name")
    openzl_by_name = load_baseline_rows(args.openzl_report_json, "name")

    report_root = args.output_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    args.repo_report_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    total_original_pt = 0
    total_original_tensor = 0
    total_pcodec = 0
    total_zstd = 0
    total_openzl = 0
    smaller_than_zstd = 0
    smaller_than_openzl = 0
    all_verified = True

    for latent_path in latents:
        name = latent_path.stem
        if name not in zstd_by_name or name not in openzl_by_name:
            raise SystemExit(f"Missing baseline row for {name}")

        obj = torch.load(latent_path, map_location="cpu")
        tensor = obj["latents"] if isinstance(obj, dict) and "latents" in obj else obj
        if not torch.is_tensor(tensor):
            raise SystemExit(f"No tensor latent payload found in {latent_path}")

        arr = tensor.detach().cpu().contiguous().view(-1).numpy()
        dtype_str = pcodec_dtype_of(arr.dtype)
        pcodec_blob, page_counts = compress_pcodec(arr, args.compression_level)
        restored = decompress_pcodec(pcodec_blob, dtype_str, page_counts)
        verified = np.array_equal(restored, arr)
        all_verified = all_verified and verified

        original_pt_bytes = latent_path.stat().st_size
        original_tensor_bytes = arr.nbytes
        pcodec_bytes = len(pcodec_blob)
        zstd_bytes = int(zstd_by_name[name]["compressed_bytes"])
        openzl_bytes = int(openzl_by_name[name]["openzl_bytes"])

        total_original_pt += original_pt_bytes
        total_original_tensor += original_tensor_bytes
        total_pcodec += pcodec_bytes
        total_zstd += zstd_bytes
        total_openzl += openzl_bytes

        if pcodec_bytes < zstd_bytes:
            smaller_than_zstd += 1
        if pcodec_bytes < openzl_bytes:
            smaller_than_openzl += 1

        rows.append(
            {
                "name": name,
                "latent_path": str(latent_path),
                "latent_shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "original_pt_bytes": original_pt_bytes,
                "original_tensor_bytes": original_tensor_bytes,
                "pt_metadata_overhead_bytes": original_pt_bytes - original_tensor_bytes,
                "pcodec_bytes": pcodec_bytes,
                "zstd_bytes": zstd_bytes,
                "openzl_bytes": openzl_bytes,
                "original_pt_mb": mb_decimal(original_pt_bytes),
                "original_tensor_mb": mb_decimal(original_tensor_bytes),
                "pcodec_mb": mb_decimal(pcodec_bytes),
                "zstd_mb": mb_decimal(zstd_bytes),
                "openzl_mb": mb_decimal(openzl_bytes),
                "pcodec_saved_vs_tensor_mb": mb_decimal(original_tensor_bytes - pcodec_bytes),
                "pcodec_delta_vs_zstd_mb": mb_decimal(pcodec_bytes - zstd_bytes),
                "pcodec_delta_vs_openzl_mb": mb_decimal(pcodec_bytes - openzl_bytes),
                "ratio_tensor_to_pcodec": round(original_tensor_bytes / pcodec_bytes, 6),
                "verified_lossless": verified,
                "original_tensor_sha256": sha256_bytes(arr.tobytes()),
                "restored_tensor_sha256": sha256_bytes(restored.tobytes()),
            }
        )

    summary = {
        "input_dir": str(args.input_dir),
        "output_root": str(args.output_root),
        "sample_count": len(rows),
        "compression_level": args.compression_level,
        "total_original_pt_mb": mb_decimal(total_original_pt),
        "total_original_tensor_mb": mb_decimal(total_original_tensor),
        "total_pcodec_mb": mb_decimal(total_pcodec),
        "total_zstd_mb": mb_decimal(total_zstd),
        "total_openzl_mb": mb_decimal(total_openzl),
        "total_pcodec_saved_vs_tensor_mb": mb_decimal(total_original_tensor - total_pcodec),
        "total_pcodec_delta_vs_zstd_mb": mb_decimal(total_pcodec - total_zstd),
        "total_pcodec_delta_vs_openzl_mb": mb_decimal(total_pcodec - total_openzl),
        "mean_original_tensor_mb": round(sum(r["original_tensor_mb"] for r in rows) / len(rows), 6),
        "mean_pcodec_mb": round(sum(r["pcodec_mb"] for r in rows) / len(rows), 6),
        "mean_zstd_mb": round(sum(r["zstd_mb"] for r in rows) / len(rows), 6),
        "mean_openzl_mb": round(sum(r["openzl_mb"] for r in rows) / len(rows), 6),
        "mean_pcodec_delta_vs_zstd_mb": round(sum(r["pcodec_delta_vs_zstd_mb"] for r in rows) / len(rows), 6),
        "mean_pcodec_delta_vs_openzl_mb": round(sum(r["pcodec_delta_vs_openzl_mb"] for r in rows) / len(rows), 6),
        "count_pcodec_smaller_than_zstd": smaller_than_zstd,
        "count_pcodec_smaller_than_openzl": smaller_than_openzl,
        "all_verified_lossless": all_verified,
    }

    json_report = {"summary": summary, "rows": rows}
    json_path = report_root / "wan64_lossless_pcodec_report.json"
    csv_path = report_root / "wan64_lossless_pcodec_report.csv"
    md_path = report_root / "wan64_lossless_pcodec_report.md"
    json_path.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    write_md(summary, rows, md_path)

    for src in (json_path, csv_path, md_path):
        dst = args.repo_report_root / src.name
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
