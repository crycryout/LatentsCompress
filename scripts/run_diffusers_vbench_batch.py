#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
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


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid(pid: int, poll_interval: int) -> None:
    while pid_exists(pid):
        print(f"waiting for pid {pid} to finish ...", flush=True)
        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["cog", "hunyuan", "mochi", "ltx"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--transformer-subfolder")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--script-path",
        default="/workspace/video_bench/scripts/run_diffusers_video_bench.py",
    )
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--negative-prompt", default="low quality, blurry, static shot, deformed anatomy, duplicated subject, artifacts")
    parser.add_argument("--max-sequence-length", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed-override", type=int)
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--wait-poll-seconds", type=int, default=60)
    args = parser.parse_args()

    if args.wait_for_pid is not None:
        wait_for_pid(args.wait_for_pid, args.wait_poll_seconds)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    end_index = args.end_index if args.end_index is not None else len(manifest)
    items = manifest[args.start_index:end_index]

    out_root = Path(args.output_dir)
    videos_dir = out_root / "videos"
    latents_dir = out_root / "latents"
    stats_dir = out_root / "stats"
    meta_dir = out_root / "meta"
    for path in (videos_dir, latents_dir, stats_dir, meta_dir):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    for global_idx, item in enumerate(items, start=args.start_index):
        prompt = item["prompt"]
        dimension = item["dimension"]
        sample_rank = item["sample_rank"]
        prompt_idx = item["prompt_index_in_suite"]
        seed = args.seed_override if args.seed_override is not None else item["generation_seed"]
        stem = (
            f"{global_idx:03d}_{dimension}_r{sample_rank:02d}_"
            f"p{prompt_idx:03d}_{slugify(prompt)}"
        )
        out_prefix = videos_dir / stem
        mp4_path = out_prefix.with_suffix(".mp4")
        latent_path = out_prefix.with_name(out_prefix.name + "_latents.pt")
        stats_path = out_prefix.with_name(out_prefix.name + "_stats.json")
        final_latent_path = latents_dir / latent_path.name
        final_stats_path = stats_dir / stats_path.name
        meta_path = meta_dir / f"{stem}.json"

        if args.skip_existing and mp4_path.exists() and final_latent_path.exists() and final_stats_path.exists():
            print(f"skip {global_idx}: {mp4_path}", flush=True)
            continue

        cmd = [
            sys.executable,
            args.script_path,
            "--family",
            args.family,
            "--model-id",
            args.model_id,
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--num-frames",
            str(args.num_frames),
            "--fps",
            str(args.fps),
            "--steps",
            str(args.steps),
            "--guidance-scale",
            str(args.guidance_scale),
            "--seed",
            str(seed),
            "--out-prefix",
            str(out_prefix),
            "--prompt",
            prompt,
            "--negative-prompt",
            args.negative_prompt,
        ]
        if args.transformer_subfolder:
            cmd.extend(["--transformer-subfolder", args.transformer_subfolder])
        if args.max_sequence_length is not None:
            cmd.extend(["--max-sequence-length", str(args.max_sequence_length)])

        run(cmd, env=env)

        latent_path.replace(final_latent_path)
        stats_path.replace(final_stats_path)

        meta = {
            "global_index": global_idx,
            "dimension": dimension,
            "prompt_suite": item["prompt_suite"],
            "sample_rank": sample_rank,
            "prompt_index_in_suite": prompt_idx,
            "prompt": prompt,
            "seed": seed,
            "family": args.family,
            "model_id": args.model_id,
            "transformer_subfolder": args.transformer_subfolder,
            "video_path": str(mp4_path),
            "latents_path": str(final_latent_path),
            "stats_path": str(final_stats_path),
            "width": args.width,
            "height": args.height,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "negative_prompt": args.negative_prompt,
            "max_sequence_length": args.max_sequence_length,
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
