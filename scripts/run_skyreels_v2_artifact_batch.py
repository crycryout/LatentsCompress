#!/usr/bin/env python3
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skyreels-root", default="/root/SkyReels-V2")
    parser.add_argument(
        "--manifest",
        default="/root/LatentsCompress/examples/skyreels_generation/skyreels_dynamic10_720p24_async_manifest.json",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Defaults to /root/SkyReels-V2/result/<timestamped-batch-dir>.",
    )
    parser.add_argument("--model-id", default="Skywork/SkyReels-V2-DF-14B-720P")
    parser.add_argument("--resolution", choices=["540P", "720P"], default="720P")
    parser.add_argument("--num-frames", type=int, default=737)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--base-num-frames", type=int, default=57)
    parser.add_argument("--overlap-history", type=int, default=17)
    parser.add_argument("--ar-step", type=int, default=5)
    parser.add_argument("--causal-block-size", type=int, default=5)
    parser.add_argument("--addnoise-condition", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--inference-steps", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-raw-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-latent-chunks", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolution_to_hw(resolution: str) -> tuple[int, int]:
    if resolution == "540P":
        return 544, 960
    if resolution == "720P":
        return 720, 1280
    raise ValueError(f"Unsupported resolution: {resolution}")


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON list.")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest entry {idx} is not an object.")
        if "id" not in item or "prompt" not in item or "seed" not in item:
            raise ValueError(f"Manifest entry {idx} must include id, prompt, and seed.")
    return data


def ensure_sys_path(skyreels_root: Path) -> None:
    root = str(skyreels_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_pipe(args, model_path: str):
    from skyreels_v2_infer import DiffusionForcingPipeline

    pipe = DiffusionForcingPipeline(
        model_path,
        dit_path=model_path,
        device=torch.device("cuda"),
        weight_dtype=torch.bfloat16,
        use_usp=False,
        offload=args.offload,
    )
    return pipe


def prepare_output_root(args) -> Path:
    if args.output_root:
        root = Path(args.output_root)
    else:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        root = Path("/root/SkyReels-V2/result") / f"skyreels_v2_dynamic10_720p24_async_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_video_artifacts(
    video_dir: Path,
    entry: dict,
    frames: np.ndarray,
    latent_chunks: list[torch.Tensor],
    config: dict,
) -> dict:
    mp4_path = video_dir / f"{entry['id']}.mp4"
    raw_frames_path = video_dir / "raw_frames_uint8.npy"
    latent_path = video_dir / "pre_vae_latent_chunks.pt"
    metadata_path = video_dir / "metadata.json"
    prompt_path = video_dir / "prompt.txt"

    prompt_path.write_text(entry["prompt"] + "\n")
    imageio.mimwrite(
        mp4_path,
        frames,
        fps=config["fps"],
        quality=8,
        output_params=["-loglevel", "error"],
    )

    if config["save_raw_frames"]:
        np.save(raw_frames_path, frames, allow_pickle=False)

    latent_chunk_shapes = [list(chunk.shape) for chunk in latent_chunks]
    latent_chunk_dtypes = [str(chunk.dtype) for chunk in latent_chunks]
    if config["save_latent_chunks"]:
        torch.save(
            {
                "latents_stage": "pre_vae_decode_async_chunks",
                "description": "Final denoised latent chunks captured immediately before each VAE decode call.",
                "model_id": config["model_id"],
                "video_id": entry["id"],
                "seed": entry["seed"],
                "prompt": entry["prompt"],
                "resolution": config["resolution"],
                "width": config["width"],
                "height": config["height"],
                "fps": config["fps"],
                "num_frames": config["num_frames"],
                "base_num_frames": config["base_num_frames"],
                "overlap_history": config["overlap_history"],
                "ar_step": config["ar_step"],
                "causal_block_size": config["causal_block_size"],
                "latent_chunks": latent_chunks,
            },
            latent_path,
        )

    metadata = {
        "video_id": entry["id"],
        "seed": entry["seed"],
        "prompt": entry["prompt"],
        "model_id": config["model_id"],
        "resolution": config["resolution"],
        "width": config["width"],
        "height": config["height"],
        "fps": config["fps"],
        "num_frames": config["num_frames"],
        "duration_seconds": config["num_frames"] / config["fps"],
        "base_num_frames": config["base_num_frames"],
        "overlap_history": config["overlap_history"],
        "ar_step": config["ar_step"],
        "causal_block_size": config["causal_block_size"],
        "addnoise_condition": config["addnoise_condition"],
        "guidance_scale": config["guidance_scale"],
        "shift": config["shift"],
        "inference_steps": config["inference_steps"],
        "paths": {
            "mp4": str(mp4_path),
            "raw_frames_npy": str(raw_frames_path) if config["save_raw_frames"] else None,
            "pre_vae_latent_chunks_pt": str(latent_path) if config["save_latent_chunks"] else None,
            "prompt_txt": str(prompt_path),
        },
        "raw_frames": {
            "shape": list(frames.shape),
            "dtype": str(frames.dtype),
            "bytes": int(frames.nbytes),
        },
        "latent_chunks": {
            "count": len(latent_chunks),
            "shapes": latent_chunk_shapes,
            "dtypes": latent_chunk_dtypes,
        },
        "mp4_bytes": mp4_path.stat().st_size,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def run_one_video(args, pipe, entry: dict, video_dir: Path, config: dict) -> dict:
    latent_chunks: list[torch.Tensor] = []
    original_decode = pipe.vae.decode

    def capture_decode(z):
        latent_chunks.append(z.detach().to("cpu", copy=True))
        return original_decode(z)

    pipe.vae.decode = capture_decode
    try:
        with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
            frames = pipe(
                prompt=entry["prompt"],
                negative_prompt=NEGATIVE_PROMPT,
                image=None,
                end_image=None,
                height=config["height"],
                width=config["width"],
                num_frames=config["num_frames"],
                num_inference_steps=config["inference_steps"],
                shift=config["shift"],
                guidance_scale=config["guidance_scale"],
                generator=torch.Generator(device="cuda").manual_seed(entry["seed"]),
                overlap_history=config["overlap_history"],
                addnoise_condition=config["addnoise_condition"],
                base_num_frames=config["base_num_frames"],
                ar_step=config["ar_step"],
                causal_block_size=config["causal_block_size"],
                fps=config["fps"],
            )[0]
    finally:
        pipe.vae.decode = original_decode

    return save_video_artifacts(video_dir, entry, frames, latent_chunks, config)


def should_skip(video_dir: Path, overwrite: bool) -> bool:
    if overwrite:
        return False
    return (video_dir / "metadata.json").exists() and (video_dir / f"{video_dir.name}.mp4").exists()


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    manifest_path = Path(args.manifest)
    skyreels_root = Path(args.skyreels_root)
    ensure_sys_path(skyreels_root)

    from skyreels_v2_infer.modules import download_model

    output_root = prepare_output_root(args)
    manifest = load_manifest(manifest_path)
    selected = manifest[args.start_index :]
    if args.limit is not None:
        selected = selected[: args.limit]

    height, width = resolution_to_hw(args.resolution)
    model_path = download_model(args.model_id)
    pipe = build_pipe(args, model_path)

    batch_metadata = {
        "manifest_path": str(manifest_path),
        "model_id": args.model_id,
        "resolved_model_path": str(model_path),
        "resolution": args.resolution,
        "width": width,
        "height": height,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "duration_seconds": args.num_frames / args.fps,
        "base_num_frames": args.base_num_frames,
        "overlap_history": args.overlap_history,
        "ar_step": args.ar_step,
        "causal_block_size": args.causal_block_size,
        "addnoise_condition": args.addnoise_condition,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "inference_steps": args.inference_steps,
        "offload": args.offload,
        "videos_requested": len(selected),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_root / "batch_metadata.json").write_text(json.dumps(batch_metadata, indent=2))

    config = {
        "model_id": args.model_id,
        "resolution": args.resolution,
        "width": width,
        "height": height,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "base_num_frames": args.base_num_frames,
        "overlap_history": args.overlap_history,
        "ar_step": args.ar_step,
        "causal_block_size": args.causal_block_size,
        "addnoise_condition": args.addnoise_condition,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "inference_steps": args.inference_steps,
        "save_raw_frames": args.save_raw_frames,
        "save_latent_chunks": args.save_latent_chunks,
    }

    for idx, entry in enumerate(selected, start=1 + args.start_index):
        video_dir = output_root / entry["id"]
        video_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = video_dir / f"{entry['id']}.mp4"
        if mp4_path.exists() and (video_dir / "metadata.json").exists() and not args.overwrite:
            print(f"[{idx}] Skipping existing output for {entry['id']}")
            continue

        print(f"[{idx}] Starting {entry['id']} seed={entry['seed']}")
        started = time.time()
        metadata = run_one_video(args, pipe, entry, video_dir, config)
        elapsed = time.time() - started
        print(
            f"[{idx}] Finished {entry['id']} in {elapsed / 60:.2f} min "
            f"mp4={metadata['paths']['mp4']} raw_frames={metadata['paths']['raw_frames_npy']} "
            f"latents={metadata['paths']['pre_vae_latent_chunks_pt']}"
        )
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
