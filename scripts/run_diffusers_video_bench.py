import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from diffusers import (
    AutoencoderKLHunyuanVideo15,
    CogVideoXPipeline,
    HunyuanVideo15Pipeline,
    HunyuanVideo15Transformer3DModel,
    LTX2Pipeline,
    LTXPipeline,
)
from diffusers.utils import export_to_video


PROMPT = (
    "A fast handheld tracking shot following a cyclist racing downhill through a neon-lit city at night, "
    "sharp turns, splashing through puddles, dramatic motion blur, dynamic camera movement, flying sparks, "
    "wind-blown jacket, cinematic lighting, highly detailed, realistic."
)


def to_frame_list(video):
    if isinstance(video, list):
        if video and isinstance(video[0], list):
            video = video[0]
        return video

    if isinstance(video, np.ndarray):
        if video.ndim == 5:
            video = video[0]
        return [frame for frame in video]

    if torch.is_tensor(video):
        if video.ndim == 5:
            video = video[0]
        video = video.detach().cpu().numpy()
        return [frame for frame in video]

    raise TypeError(f"Unsupported video container: {type(video)!r}")


def trim_cog_latents(pipe, latents, requested_num_frames):
    latent_frames = (requested_num_frames - 1) // pipe.vae_scale_factor_temporal + 1
    patch_size_t = pipe.transformer.config.patch_size_t
    additional_frames = 0
    if patch_size_t is not None and latent_frames % patch_size_t != 0:
        additional_frames = patch_size_t - latent_frames % patch_size_t
    if additional_frames > 0:
        latents = latents[:, additional_frames:]
    return latents, additional_frames


def load_pipeline(args):
    dtype = torch.bfloat16

    if args.family == "cog":
        pipe = CogVideoXPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    elif args.family == "hunyuan":
        if args.transformer_subfolder:
            cfg_path = hf_hub_download(args.model_id, f"{args.transformer_subfolder}/config.json")
            raw_cfg = json.loads(Path(cfg_path).read_text())
            patch_size = raw_cfg.get("patch_size", 1)
            if isinstance(patch_size, (list, tuple)):
                patch_size = int(patch_size[-1])
            patch_size_t = raw_cfg.get("patch_size_t", 1) or 1
            qk_norm = str(raw_cfg.get("qk_norm_type", "rms_norm"))
            if qk_norm == "rms":
                qk_norm = "rms_norm"
            target_size = raw_cfg.get("ideal_resolution", "720p")
            if isinstance(target_size, str) and target_size.endswith("p"):
                target_size = int(target_size[:-1])

            transformer = HunyuanVideo15Transformer3DModel.from_pretrained(
                args.model_id,
                subfolder=args.transformer_subfolder,
                torch_dtype=dtype,
                in_channels=int(raw_cfg.get("in_channels", 65)),
                out_channels=int(raw_cfg.get("out_channels", 32)),
                num_attention_heads=int(raw_cfg.get("heads_num", 16)),
                attention_head_dim=int(raw_cfg.get("hidden_size", 2048) // raw_cfg.get("heads_num", 16)),
                num_layers=int(raw_cfg.get("mm_double_blocks_depth", 54)),
                mlp_ratio=float(raw_cfg.get("mlp_width_ratio", 4.0)),
                patch_size=int(patch_size),
                patch_size_t=int(patch_size_t),
                qk_norm=qk_norm,
                text_embed_dim=int(raw_cfg.get("text_states_dim", 3584)),
                image_embed_dim=int(raw_cfg.get("vision_states_dim", 1152)),
                rope_axes_dim=tuple(raw_cfg.get("rope_dim_list", [16, 56, 56])),
                target_size=int(target_size),
                task_type=str(raw_cfg.get("ideal_task", "t2v")),
                use_meanflow=bool(raw_cfg.get("use_meanflow", False)),
            )
            pipe = HunyuanVideo15Pipeline.from_pretrained(
                args.model_id,
                transformer=transformer,
                torch_dtype=dtype,
            )
        else:
            pipe = HunyuanVideo15Pipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    elif args.family == "ltx":
        if "LTX-2" in args.model_id or "LTX-2." in args.model_id:
            pipe = LTX2Pipeline.from_pretrained(args.model_id, torch_dtype=dtype)
        else:
            pipe = LTXPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    else:
        raise ValueError(f"Unsupported family: {args.family}")

    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["cog", "hunyuan", "ltx"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--transformer-subfolder")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260331)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    mp4_path = out_prefix.with_suffix(".mp4")
    latent_path = out_prefix.with_name(out_prefix.name + "_latents.pt")
    stats_path = out_prefix.with_name(out_prefix.name + "_stats.json")

    pipe = load_pipeline(args)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    if args.family == "hunyuan":
        latent_output = pipe(
            prompt=args.prompt,
            negative_prompt="low quality, blurry, static shot, deformed anatomy, duplicated subject, artifacts",
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            generator=generator,
            output_type="latent",
        )
        latents = latent_output.frames.detach().to("cpu", copy=True)
        torch.save(
            {
                "family": args.family,
                "model_id": args.model_id,
                "seed": args.seed,
                "prompt": args.prompt,
                "width": args.width,
                "height": args.height,
                "num_frames": args.num_frames,
                "fps": args.fps,
                "additional_frames_trimmed": 0,
                "latents": latents,
            },
            latent_path,
        )
        video_processor = pipe.video_processor
        vae_scaling_factor = pipe.vae.config.scaling_factor
        del latent_output
        del pipe
        torch.cuda.empty_cache()

        vae = AutoencoderKLHunyuanVideo15.from_pretrained(
            args.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        decode_latents = latents.to(torch.float32) / vae_scaling_factor
        video = vae.decode(decode_latents, return_dict=False)[0]
        frames = to_frame_list(video_processor.postprocess_video(video, output_type="np"))
    else:
        latent_store = {}

        def capture_latents(_pipe, _step, _timestep, callback_kwargs):
            lat = callback_kwargs["latents"]
            latent_store["latents"] = lat.detach().to("cpu", copy=True)
            return callback_kwargs

        kwargs = dict(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            generator=generator,
            output_type="np",
            callback_on_step_end=capture_latents,
            callback_on_step_end_tensor_inputs=["latents"],
            guidance_scale=args.guidance_scale,
        )

        if args.family == "ltx":
            kwargs["frame_rate"] = args.fps
        else:
            kwargs["negative_prompt"] = (
                "low quality, blurry, static shot, deformed anatomy, duplicated subject, artifacts"
            )

        output = pipe(**kwargs)
        frames = to_frame_list(output.frames)
        latents = latent_store["latents"]

    export_to_video(frames, str(mp4_path), fps=args.fps)

    additional_frames = 0
    if args.family == "cog":
        latents, additional_frames = trim_cog_latents(pipe, latents, args.num_frames)

    if args.family != "hunyuan":
        torch.save(
            {
                "family": args.family,
                "model_id": args.model_id,
                "seed": args.seed,
                "prompt": args.prompt,
                "width": args.width,
                "height": args.height,
                "num_frames": args.num_frames,
                "fps": args.fps,
                "additional_frames_trimmed": additional_frames,
                "latents": latents,
            },
            latent_path,
        )

    first = np.asarray(frames[0])
    frame_shape = tuple(int(x) for x in first.shape)
    raw_video_bytes = len(frames) * first.size * first.itemsize
    stats = {
        "family": args.family,
        "model_id": args.model_id,
        "prompt": args.prompt,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "num_frames": len(frames),
        "fps": args.fps,
        "frame_shape": frame_shape,
        "frame_dtype": str(first.dtype),
        "raw_video_bytes": raw_video_bytes,
        "raw_video_mib": raw_video_bytes / 1024 / 1024,
        "mp4_bytes": mp4_path.stat().st_size,
        "mp4_mib": mp4_path.stat().st_size / 1024 / 1024,
        "latents_shape": list(latents.shape),
        "latents_dtype": str(latents.dtype),
        "latents_raw_bytes": latents.numel() * latents.element_size(),
        "latents_raw_mib": (latents.numel() * latents.element_size()) / 1024 / 1024,
        "latents_pt_bytes": latent_path.stat().st_size,
        "latents_pt_mib": latent_path.stat().st_size / 1024 / 1024,
        "cog_additional_frames_trimmed": additional_frames,
        "mp4_path": str(mp4_path),
        "latents_path": str(latent_path),
    }

    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
