#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_MANIFEST = "/workspace/video_bench/manifests/wan22_vbench_16x2_seed42.json"
DEFAULT_BATCH_SCRIPT = "/workspace/video_bench/scripts/run_diffusers_vbench_batch.py"
DEFAULT_WAN_PID = 30316


def build_jobs(root_output_dir: str) -> list[dict[str, object]]:
    return [
        {
            "name": "hunyuanvideo15_720p",
            "family": "hunyuan",
            "model_id": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
            "width": 1280,
            "height": 720,
            "num_frames": 121,
            "fps": 24,
            "steps": 30,
            "guidance_scale": 1.0,
            "output_dir": str(Path(root_output_dir) / "hunyuanvideo15_vbench_16x2_seed42"),
        },
        {
            "name": "mochi1_preview",
            "family": "mochi",
            "model_id": "genmo/mochi-1-preview",
            "width": 848,
            "height": 480,
            "num_frames": 121,
            "fps": 24,
            "steps": 64,
            "guidance_scale": 4.5,
            "max_sequence_length": 256,
            "output_dir": str(Path(root_output_dir) / "mochi1_vbench_16x2_seed42"),
        },
        {
            "name": "ltx_video",
            "family": "ltx",
            "model_id": "Lightricks/LTX-Video",
            "width": 1216,
            "height": 704,
            "num_frames": 121,
            "fps": 24,
            "steps": 40,
            "guidance_scale": 3.0,
            "output_dir": str(Path(root_output_dir) / "ltxvideo_vbench_16x2_seed42"),
        },
    ]


def build_command(
    job: dict[str, object],
    manifest: str,
    batch_script: str,
    wait_for_pid: int | None,
    skip_existing: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        batch_script,
        "--family",
        str(job["family"]),
        "--model-id",
        str(job["model_id"]),
        "--manifest",
        manifest,
        "--output-dir",
        str(job["output_dir"]),
        "--width",
        str(job["width"]),
        "--height",
        str(job["height"]),
        "--num-frames",
        str(job["num_frames"]),
        "--fps",
        str(job["fps"]),
        "--steps",
        str(job["steps"]),
        "--guidance-scale",
        str(job["guidance_scale"]),
    ]
    if "max_sequence_length" in job:
        cmd.extend(["--max-sequence-length", str(job["max_sequence_length"])])
    if wait_for_pid is not None:
        cmd.extend(["--wait-for-pid", str(wait_for_pid)])
    if skip_existing:
        cmd.append("--skip-existing")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--batch-script", default=DEFAULT_BATCH_SCRIPT)
    parser.add_argument("--root-output-dir", default="/workspace/video_bench")
    parser.add_argument("--wait-for-pid", type=int, default=DEFAULT_WAN_PID)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute the queued jobs. Without this flag the script only prints commands.",
    )
    args = parser.parse_args()

    jobs = build_jobs(args.root_output_dir)
    commands = [
        build_command(job, args.manifest, args.batch_script, args.wait_for_pid, args.skip_existing)
        for job in jobs
    ]

    for job, cmd in zip(jobs, commands, strict=True):
        print(f"# {job['name']}")
        print(shlex.join(cmd))

    if not args.execute:
        return

    for cmd in commands:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
