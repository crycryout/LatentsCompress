#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import torch

from ltx_core.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import ModalitySpec
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser, SimpleDenoiser


def parse_args():
    p = argparse.ArgumentParser(description='Run official LTX-2 two-stage pipeline and save intermediate latents.')
    p.add_argument('--checkpoint-path', required=True)
    p.add_argument('--distilled-lora-path', required=True)
    p.add_argument('--distilled-lora-strength', type=float, default=1.0)
    p.add_argument('--spatial-upsampler-path', required=True)
    p.add_argument('--gemma-root', required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--negative-prompt', default=DEFAULT_NEGATIVE_PROMPT)
    p.add_argument('--output-prefix', required=True)
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--height', type=int, default=1024)
    p.add_argument('--width', type=int, default=1536)
    p.add_argument('--num-frames', type=int, default=121)
    p.add_argument('--frame-rate', type=float, default=24.0)
    p.add_argument('--num-inference-steps', type=int, default=40)
    p.add_argument('--streaming-prefetch-count', type=int, default=None)
    p.add_argument('--max-batch-size', type=int, default=1)
    p.add_argument('--enhance-prompt', action='store_true')
    p.add_argument('--quantization', choices=['none', 'fp8-upcast', 'fp8-cast'], default='fp8-upcast')
    p.add_argument('--with-audio', action='store_true', help='Keep the official audio branch enabled.')
    return p.parse_args()


def save_tensor(path: Path, tensor: torch.Tensor, stage: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'stage': stage,
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'tensor': tensor.detach().to('cpu').contiguous(),
    }
    torch.save(payload, path)


def main():
    args = parse_args()
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = str(Path(args.checkpoint_path).resolve())
    distilled_lora = [
        LoraPathStrengthAndSDOps(
            str(Path(args.distilled_lora_path).resolve()),
            args.distilled_lora_strength,
            None,
        )
    ]

    quantization = None
    if args.quantization == 'fp8-cast':
        quantization = QuantizationPolicy.fp8_cast()
    elif args.quantization == 'fp8-upcast':
        from ltx_core.quantization.fp8_cast import UPCAST_DURING_INFERENCE

        quantization = QuantizationPolicy(sd_ops=SDOps(name='identity'), module_ops=(UPCAST_DURING_INFERENCE,))

    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=str(Path(args.spatial_upsampler_path).resolve()),
        gemma_root=str(Path(args.gemma_root).resolve()),
        loras=(),
        quantization=quantization,
        torch_compile=False,
    )

    assert_resolution(height=args.height, width=args.width, is_two_stage=True)

    generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)
    noiser = GaussianNoiser(generator=generator)
    dtype = torch.bfloat16

    t0 = time.time()
    ctx_p, ctx_n = pipeline.prompt_encoder(
        [args.prompt, args.negative_prompt],
        enhance_first_prompt=args.enhance_prompt,
        enhance_prompt_image=None,
        enhance_prompt_seed=args.seed,
        streaming_prefetch_count=args.streaming_prefetch_count,
    )
    v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
    v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding
    if not args.with_audio:
        a_context_p = None
        a_context_n = None

    stage1_output_shape = VideoPixelShape(
        batch=1,
        frames=args.num_frames,
        width=args.width // 2,
        height=args.height // 2,
        fps=args.frame_rate,
    )
    stage1_conditionings = pipeline.image_conditioner(
        lambda enc: combined_image_conditionings(
            images=[],
            height=stage1_output_shape.height,
            width=stage1_output_shape.width,
            video_encoder=enc,
            dtype=dtype,
            device=pipeline.device,
        )
    )

    sigmas = LTX2Scheduler().execute(steps=args.num_inference_steps).to(dtype=torch.float32, device=pipeline.device)
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[29],
    )
    if not args.with_audio:
        audio_guider_params = MultiModalGuiderParams(cfg_scale=1.0, stg_scale=0.0, modality_scale=1.0)

    video_state, audio_state = pipeline.stage_1(
        denoiser=FactoryGuidedDenoiser(
            v_context=v_context_p,
            a_context=a_context_p,
            video_guider_factory=create_multimodal_guider_factory(
                params=MultiModalGuiderParams(
                    cfg_scale=3.0,
                    stg_scale=1.0,
                    rescale_scale=0.7,
                    modality_scale=3.0,
                    skip_step=0,
                    stg_blocks=[29],
                ),
                negative_context=v_context_n,
            ),
            audio_guider_factory=create_multimodal_guider_factory(
                params=audio_guider_params,
                negative_context=a_context_n,
            ),
        ),
        sigmas=sigmas,
        noiser=noiser,
        width=stage1_output_shape.width,
        height=stage1_output_shape.height,
        frames=args.num_frames,
        fps=args.frame_rate,
        video=ModalitySpec(context=v_context_p, conditionings=stage1_conditionings),
        audio=ModalitySpec(context=a_context_p) if args.with_audio else None,
        streaming_prefetch_count=args.streaming_prefetch_count,
        max_batch_size=args.max_batch_size,
    )

    stage1_video_shape = list(video_state.latent.shape)
    save_tensor(out_prefix.with_name(out_prefix.name + '_stage1_video_latents.pt'), video_state.latent, 'stage1_video')
    stage1_audio_shape = None
    if audio_state is not None:
        stage1_audio_shape = list(audio_state.latent.shape)
        save_tensor(out_prefix.with_name(out_prefix.name + '_stage1_audio_latents.pt'), audio_state.latent, 'stage1_audio')

    upscaled_video_latent = pipeline.upsampler(video_state.latent[:1])
    upscaled_video_shape = list(upscaled_video_latent.shape)
    save_tensor(out_prefix.with_name(out_prefix.name + '_upscaled_video_latents.pt'), upscaled_video_latent, 'stage2_upscaled_video')

    distilled_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, device=pipeline.device)
    stage2_conditionings = pipeline.image_conditioner(
        lambda enc: combined_image_conditionings(
            images=[],
            height=args.height,
            width=args.width,
            video_encoder=enc,
            dtype=dtype,
            device=pipeline.device,
        )
    )

    video_state, audio_state = pipeline.stage_2(
        denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
        sigmas=distilled_sigmas,
        noiser=noiser,
        width=args.width,
        height=args.height,
        frames=args.num_frames,
        fps=args.frame_rate,
        video=ModalitySpec(
            context=v_context_p,
            conditionings=stage2_conditionings,
            noise_scale=distilled_sigmas[0].item(),
            initial_latent=upscaled_video_latent,
        ),
        audio=(
            ModalitySpec(
                context=a_context_p,
                noise_scale=distilled_sigmas[0].item(),
                initial_latent=audio_state.latent,
            )
            if args.with_audio and audio_state is not None
            else None
        ),
        streaming_prefetch_count=args.streaming_prefetch_count,
    )

    final_video_shape = list(video_state.latent.shape)
    save_tensor(out_prefix.with_name(out_prefix.name + '_final_video_latents.pt'), video_state.latent, 'final_pre_vae_video')
    final_audio_shape = None
    if audio_state is not None:
        final_audio_shape = list(audio_state.latent.shape)
        save_tensor(out_prefix.with_name(out_prefix.name + '_final_audio_latents.pt'), audio_state.latent, 'final_pre_decoder_audio')

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    decoded_video = pipeline.video_decoder(video_state.latent, tiling_config, generator)
    decoded_audio = pipeline.audio_decoder(audio_state.latent) if audio_state is not None else None
    mp4_path = out_prefix.with_suffix('.mp4')
    encode_video(decoded_video, int(args.frame_rate), decoded_audio, str(mp4_path), video_chunks_number)

    stats = {
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'checkpoint_path': checkpoint_path,
        'distilled_lora_path': str(Path(args.distilled_lora_path).resolve()),
        'spatial_upsampler_path': str(Path(args.spatial_upsampler_path).resolve()),
        'gemma_root': str(Path(args.gemma_root).resolve()),
        'height': args.height,
        'width': args.width,
        'num_frames': args.num_frames,
        'frame_rate': args.frame_rate,
        'num_inference_steps': args.num_inference_steps,
        'seed': args.seed,
        'quantization': args.quantization,
        'with_audio': args.with_audio,
        'device': str(pipeline.device),
        'stage1_video_latent_shape': stage1_video_shape,
        'stage1_audio_latent_shape': stage1_audio_shape,
        'upscaled_video_latent_shape': upscaled_video_shape,
        'final_video_latent_shape': final_video_shape,
        'final_audio_latent_shape': final_audio_shape,
        'mp4_path': str(mp4_path),
        'elapsed_sec': time.time() - t0,
    }
    with open(out_prefix.with_name(out_prefix.name + '_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
