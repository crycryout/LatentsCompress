#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def slugify(text: str, limit: int = 80) -> str:
    safe = []
    for ch in text.lower():
        if ch.isalnum():
            safe.append(ch)
        elif ch in (" ", "-", "_"):
            safe.append("_")
    slug = "".join(safe)
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")
    return slug[:limit] or "prompt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="t2v-A14B",
        help="Wan task name, e.g. t2v-A14B or ti2v-5B.",
    )
    parser.add_argument(
        "--manifest",
        default="/workspace/video_bench/manifests/wan22_vbench_16x20_seed42.json",
    )
    parser.add_argument(
        "--wan-root",
        default="/root/Wan2.2",
    )
    parser.add_argument(
        "--ckpt-dir",
        default="/workspace/models/Wan2.2-T2V-A14B",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/video_bench/wan22_vbench_16x20_seed42",
    )
    parser.add_argument(
        "--size",
        default="1280*720",
    )
    parser.add_argument(
        "--frame-num",
        type=int,
        default=81,
        help="Wan native frame count. 81 frames at 16 fps is about 5.06s.",
    )
    parser.add_argument(
        "--native-fps",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--export-fps",
        type=int,
        default=24,
        help="Final MP4 fps. Uses ffmpeg fps filter to preserve duration.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--sample-shift",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--sample-guide-scale",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--save-latents",
        action="store_true",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
    )
    parser.add_argument(
        "--offload-model",
        action="store_true",
    )
    parser.add_argument(
        "--t5-cpu",
        action="store_true",
    )
    parser.add_argument(
        "--convert-model-dtype",
        action="store_true",
    )
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    end_index = args.end_index if args.end_index is not None else len(manifest)
    items = manifest[args.start_index:end_index]

    out_root = Path(args.output_dir)
    native_dir = out_root / "native_16fps"
    final_dir = out_root / "final_24fps"
    latent_dir = out_root / "latents"
    logs_dir = out_root / "logs"
    for path in (native_dir, final_dir, latent_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.wan_root}:{env.get('PYTHONPATH', '')}".rstrip(":")

    for global_idx, item in enumerate(items, start=args.start_index):
        prompt = item["prompt"]
        dimension = item["dimension"]
        sample_rank = item["sample_rank"]
        seed = item["generation_seed"]
        prompt_idx = item["prompt_index_in_suite"]
        stem = (
            f"{global_idx:03d}_{dimension}_r{sample_rank:02d}_"
            f"p{prompt_idx:03d}_{slugify(prompt)}"
        )

        native_mp4 = native_dir / f"{stem}.mp4"
        final_mp4 = final_dir / f"{stem}.mp4"
        latents_path = latent_dir / f"{stem}.pt"
        meta_path = final_dir / f"{stem}.json"

        if args.skip_existing and final_mp4.exists() and meta_path.exists():
            print(f"skip {global_idx}: {final_mp4}")
            continue

        cmd = [
            "python3",
            str(Path(args.wan_root) / "generate.py"),
            "--task",
            args.task,
            "--size",
            args.size,
            "--frame_num",
            str(args.frame_num),
            "--ckpt_dir",
            args.ckpt_dir,
            "--prompt",
            prompt,
            "--base_seed",
            str(seed),
            "--sample_steps",
            str(args.sample_steps),
            "--sample_shift",
            str(args.sample_shift),
            "--sample_guide_scale",
            str(args.sample_guide_scale),
            "--save_file",
            str(native_mp4),
        ]
        if args.save_latents:
            cmd.extend(["--save_latents_file", str(latents_path)])
        if args.offload_model:
            cmd.extend(["--offload_model", "True"])
        if args.t5_cpu:
            cmd.extend(["--t5_cpu"])
        if args.convert_model_dtype:
            cmd.extend(["--convert_model_dtype"])

        run(cmd, env=env)

        if args.export_fps != args.native_fps:
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(native_mp4),
                "-vf",
                f"fps={args.export_fps}",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(final_mp4),
            ]
            run(ffmpeg_cmd)
        else:
            final_mp4.write_bytes(native_mp4.read_bytes())

        meta = {
            "global_index": global_idx,
            "dimension": dimension,
            "prompt_suite": item["prompt_suite"],
            "sample_rank": sample_rank,
            "prompt_index_in_suite": prompt_idx,
            "prompt": prompt,
            "seed": seed,
            "native_video_path": str(native_mp4),
            "final_video_path": str(final_mp4),
            "latents_path": str(latents_path) if args.save_latents else None,
            "task": args.task,
            "size": args.size,
            "native_fps": args.native_fps,
            "export_fps": args.export_fps,
            "frame_num": args.frame_num,
            "offload_model": args.offload_model,
            "t5_cpu": args.t5_cpu,
            "convert_model_dtype": args.convert_model_dtype,
            "native_duration_sec": args.frame_num / args.native_fps,
            "export_duration_sec": args.frame_num / args.native_fps,
            "note": (
                "Wan2.2-T2V-A14B natively samples at 16 fps. "
                "The final 24 fps file is exported with ffmpeg fps filter "
                "to preserve duration."
            ),
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
