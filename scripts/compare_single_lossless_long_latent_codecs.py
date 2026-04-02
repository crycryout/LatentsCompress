#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import openzl.ext as zl
import pcodec
import pcodec.standalone as ps
import torch
import zstandard as zstd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare lossless ZSTD/OpenZL/Pcodec compression on one saved long-video latent."
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
        "--mp4-path",
        type=Path,
        default=Path(
            "/workspace/video_bench/skyreels_v2_vbench2_60s_720p/videos/subject_consistency/subject_consistency_000_5eaae1c7.mp4"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/video_bench/codec_compare/skyreels_long_lossless"),
    )
    parser.add_argument("--zstd-level", type=int, default=19)
    parser.add_argument(
        "--pcodec-level", type=int, default=pcodec.DEFAULT_COMPRESSION_LEVEL
    )
    parser.add_argument("--save-containers", action="store_true")
    return parser.parse_args()


def bytes_to_mb(nbytes: int) -> float:
    return round(nbytes / 1_000_000, 6)


def bytes_to_mib(nbytes: int) -> float:
    return round(nbytes / (1024 * 1024), 6)


def ratio(original: int, compressed: int) -> float:
    return round(original / compressed, 6)


def ffprobe_video(path: Path) -> dict[str, int | float]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    data = json.loads(proc.stdout)
    stream = data["streams"][0]
    fmt = data["format"]
    fps_num, fps_den = stream["r_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "nb_frames": int(stream["nb_frames"]),
        "fps": float(fps_num) / float(fps_den),
        "duration": float(fmt["duration"]),
        "size_bytes": int(fmt["size"]),
    }


def build_openzl_contexts() -> tuple[zl.CCtx, zl.DCtx]:
    compressor = zl.Compressor()
    graph = zl.graphs.Compress()(compressor)
    compressor.select_starting_graph(graph)
    cctx = zl.CCtx()
    cctx.ref_compressor(compressor)
    cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
    dctx = zl.DCtx()
    return cctx, dctx


def write_md(report: dict, path: Path) -> None:
    codec_rows = report["codec_rows"]
    sizes = report["sizes"]
    video = report["video"]
    lines: list[str] = []
    lines.append("# SkyReels Long Latent Lossless Codec Comparison")
    lines.append("")
    lines.append(f"- latent: `{report['latent_path']}`")
    lines.append(f"- mp4: `{report['mp4_path']}`")
    lines.append(f"- latent shape: `{report['latent_shape']}`")
    lines.append(f"- latent dtype: `{report['latent_dtype']}`")
    lines.append("")
    lines.append("## Size Summary")
    lines.append("")
    lines.append("| item | bytes | MB | MiB |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| original latent `.pt` | `{sizes['latent_pt_bytes']}` | `{sizes['latent_pt_mb']}` | `{sizes['latent_pt_mib']}` |"
    )
    lines.append(
        f"| original latent payload | `{sizes['latent_payload_bytes']}` | `{sizes['latent_payload_mb']}` | `{sizes['latent_payload_mib']}` |"
    )
    lines.append(
        f"| raw video RGB24 | `{sizes['raw_rgb24_bytes']}` | `{sizes['raw_rgb24_mb']}` | `{sizes['raw_rgb24_mib']}` |"
    )
    lines.append(
        f"| raw video YUV420 | `{sizes['raw_yuv420_bytes']}` | `{sizes['raw_yuv420_mb']}` | `{sizes['raw_yuv420_mib']}` |"
    )
    lines.append(
        f"| generated MP4 | `{sizes['mp4_bytes']}` | `{sizes['mp4_mb']}` | `{sizes['mp4_mib']}` |"
    )
    lines.append("")
    lines.append("## Lossless Codec Results")
    lines.append("")
    lines.append(
        "| codec | target | compressed bytes | MB | MiB | ratio vs payload | ratio vs `.pt` | verified lossless |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in codec_rows:
        lines.append(
            f"| `{row['codec']}` | `{row['target']}` | `{row['compressed_bytes']}` | "
            f"`{row['compressed_mb']}` | `{row['compressed_mib']}` | "
            f"`{row['ratio_vs_payload']}` | `{row['ratio_vs_pt']}` | "
            f"`{row['verified_lossless']}` |"
        )
    lines.append("")
    lines.append("## Video Context")
    lines.append("")
    lines.append(
        f"- resolution: `{video['width']}x{video['height']}` at `{video['fps']}` fps"
    )
    lines.append(f"- frames: `{video['nb_frames']}`")
    lines.append(f"- duration: `{video['duration']}` s")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- ZSTD and OpenZL are run on the exact latent payload bytes extracted from the saved tensor."
    )
    lines.append(
        "- Pcodec is run on a `uint16` view of the same `bfloat16` payload, so the bit-pattern is preserved exactly."
    )
    lines.append(
        "- The saved `.pt` container overhead is tiny, so payload-vs-file comparisons differ only slightly."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.latent_path.exists():
        raise SystemExit(f"Missing latent: {args.latent_path}")
    if not args.mp4_path.exists():
        raise SystemExit(f"Missing mp4: {args.mp4_path}")

    latent_obj = torch.load(args.latent_path, map_location="cpu")
    tensor = (
        latent_obj["latents"]
        if isinstance(latent_obj, dict) and "latents" in latent_obj
        else latent_obj
    )
    if not torch.is_tensor(tensor):
        raise SystemExit("Latent payload is not a torch tensor.")

    tensor = tensor.detach().cpu().contiguous()
    payload_u8 = tensor.view(torch.uint8).contiguous().reshape(-1).numpy()
    payload_u16 = tensor.view(torch.uint16).contiguous().reshape(-1).numpy()
    payload_bytes = payload_u8.tobytes()
    pt_bytes = args.latent_path.read_bytes()

    output_dir = args.output_root / args.latent_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    zstd_cctx = zstd.ZstdCompressor(level=args.zstd_level)
    zstd_dctx = zstd.ZstdDecompressor()
    zstd_blob = zstd_cctx.compress(payload_bytes)
    zstd_restored = zstd_dctx.decompress(zstd_blob)

    openzl_cctx, openzl_dctx = build_openzl_contexts()
    openzl_blob = openzl_cctx.compress([zl.Input(zl.Type.Serial, payload_bytes)])
    openzl_restored = openzl_dctx.decompress(openzl_blob)[0].content.as_bytes()

    pcodec_blob = ps.simple_compress(
        payload_u16,
        pcodec.ChunkConfig(compression_level=args.pcodec_level),
    )
    pcodec_restored = ps.simple_decompress(pcodec_blob)
    pcodec_restored_bytes = pcodec_restored.astype(payload_u16.dtype, copy=False).tobytes()

    codec_rows = []
    for codec, blob, restored_ok in (
        ("zstd", zstd_blob, zstd_restored == payload_bytes),
        ("openzl", openzl_blob, openzl_restored == payload_bytes),
        ("pcodec", pcodec_blob, pcodec_restored_bytes == payload_bytes),
    ):
        compressed_bytes = len(blob)
        codec_rows.append(
            {
                "codec": codec,
                "target": "latent_payload",
                "compressed_bytes": compressed_bytes,
                "compressed_mb": bytes_to_mb(compressed_bytes),
                "compressed_mib": bytes_to_mib(compressed_bytes),
                "ratio_vs_payload": ratio(len(payload_bytes), compressed_bytes),
                "ratio_vs_pt": ratio(len(pt_bytes), compressed_bytes),
                "verified_lossless": restored_ok,
            }
        )

    if args.save_containers:
        (output_dir / f"{args.latent_path.stem}.payload.zst").write_bytes(zstd_blob)
        (output_dir / f"{args.latent_path.stem}.payload.ozl").write_bytes(openzl_blob)
        (output_dir / f"{args.latent_path.stem}.payload.pco").write_bytes(pcodec_blob)

    video = ffprobe_video(args.mp4_path)
    raw_rgb24_bytes = video["width"] * video["height"] * 3 * video["nb_frames"]
    raw_yuv420_bytes = int(video["width"] * video["height"] * 1.5 * video["nb_frames"])

    report = {
        "latent_path": str(args.latent_path),
        "mp4_path": str(args.mp4_path),
        "latent_shape": list(tensor.shape),
        "latent_dtype": str(tensor.dtype).replace("torch.", ""),
        "zstd_level": args.zstd_level,
        "pcodec_level": args.pcodec_level,
        "sizes": {
            "latent_pt_bytes": len(pt_bytes),
            "latent_pt_mb": bytes_to_mb(len(pt_bytes)),
            "latent_pt_mib": bytes_to_mib(len(pt_bytes)),
            "latent_payload_bytes": len(payload_bytes),
            "latent_payload_mb": bytes_to_mb(len(payload_bytes)),
            "latent_payload_mib": bytes_to_mib(len(payload_bytes)),
            "pt_metadata_overhead_bytes": len(pt_bytes) - len(payload_bytes),
            "raw_rgb24_bytes": raw_rgb24_bytes,
            "raw_rgb24_mb": bytes_to_mb(raw_rgb24_bytes),
            "raw_rgb24_mib": bytes_to_mib(raw_rgb24_bytes),
            "raw_yuv420_bytes": raw_yuv420_bytes,
            "raw_yuv420_mb": bytes_to_mb(raw_yuv420_bytes),
            "raw_yuv420_mib": bytes_to_mib(raw_yuv420_bytes),
            "mp4_bytes": video["size_bytes"],
            "mp4_mb": bytes_to_mb(video["size_bytes"]),
            "mp4_mib": bytes_to_mib(video["size_bytes"]),
            "mp4_ratio_vs_raw_rgb24": ratio(raw_rgb24_bytes, video["size_bytes"]),
            "mp4_ratio_vs_raw_yuv420": ratio(raw_yuv420_bytes, video["size_bytes"]),
        },
        "video": video,
        "codec_rows": codec_rows,
    }

    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_md(report, md_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
