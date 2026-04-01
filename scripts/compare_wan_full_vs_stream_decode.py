#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, '/root/GenLatents/scripts')
from wan_streaming_decode import WanStreamingDecoder, load_latents, maybe_cleanup, synchronize  # type: ignore


DEFAULTS = {
    'wan_t2v_a14b': {
        'latent_paths': [
            '/workspace/outputs/wan22_t2v_a14b_720p16_lighthouse_49f_latents.pt',
        ],
        'ckpt_dir': '/workspace/models/Wan2.2-T2V-A14B',
        'wan_root': '/root/Wan2.2',
    },
    'wan_ti2v_5b': {
        'latent_paths': [
            '/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/000_subject_consistency_r00_p003_a_person_eating_a_burger.pt',
            '/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/015_motion_smoothness_r03_p064_a_bear_climbing_a_tree.pt',
            '/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/032_multiple_objects_r00_p011_a_couch_and_a_potted_plant.pt',
            '/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/048_scene_r00_p037_golf_course.pt',
        ],
        'ckpt_dir': '/workspace/models/Wan2.2-TI2V-5B',
        'wan_root': '/root/Wan2.2',
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare Wan full decode vs streaming decode time and quality.')
    parser.add_argument('--family', choices=list(DEFAULTS.keys()), required=True)
    parser.add_argument('--latent-path', action='append', default=[])
    parser.add_argument('--ckpt-dir', default=None)
    parser.add_argument('--wan-root', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model-dtype', default='bf16')
    parser.add_argument('--use-torch-compile', action='store_true')
    parser.add_argument('--compile-mode', default='default', choices=['default', 'reduce-overhead', 'max-autotune'])
    parser.add_argument('--compile-backend', default='inductor')
    parser.add_argument('--latent-group-size', type=int, default=1)
    parser.add_argument('--stream-conv2-cache', action='store_true')
    parser.add_argument('--warmup-runs', type=int, default=0, help='Untimed warmup runs for both full and stream paths before measurement.')
    parser.add_argument('--output-json', required=True)
    return parser.parse_args()


def to_uint8(video: torch.Tensor) -> torch.Tensor:
    return (((video.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8))


def mse_psnr(ref: torch.Tensor, dist: torch.Tensor, max_val: float) -> tuple[float, float]:
    ref_f = ref.to(torch.float32)
    dist_f = dist.to(torch.float32)
    mse = float(torch.mean((ref_f - dist_f) ** 2).item())
    if mse == 0.0:
        return 0.0, 100.0
    return mse, float(20.0 * math.log10(max_val) - 10.0 * math.log10(mse))


def _run_stream_once(streamer: WanStreamingDecoder, latents: torch.Tensor, latent_group_size: int) -> torch.Tensor:
    streamer.reset_stream()
    parts = []
    for start in range(0, latents.shape[1], latent_group_size):
        chunk = latents[:, start:start + latent_group_size].contiguous()
        frames = streamer.push_latent_chunk(chunk)
        parts.append(frames)
    return torch.cat(parts, dim=1)


def compare_case(streamer: WanStreamingDecoder, latent_path: str, latent_group_size: int, warmup_runs: int) -> dict[str, Any]:
    latents, meta = load_latents(latent_path)

    for _ in range(warmup_runs):
        _ = streamer.decode_full(latents)
        maybe_cleanup()
        _ = _run_stream_once(streamer, latents, latent_group_size)
        maybe_cleanup()

    synchronize()
    t0 = time.perf_counter()
    full_video = streamer.decode_full(latents)
    synchronize()
    full_sec = time.perf_counter() - t0

    synchronize()
    t1 = time.perf_counter()
    stream_video = _run_stream_once(streamer, latents, latent_group_size)
    synchronize()
    stream_sec = time.perf_counter() - t1

    diff = (full_video - stream_video).to(torch.float32)
    float_mse = float(torch.mean(diff * diff).item())
    float_mae = float(torch.mean(torch.abs(diff)).item())
    float_max_abs = float(torch.max(torch.abs(diff)).item())

    full_u8 = to_uint8(full_video)
    stream_u8 = to_uint8(stream_video)
    u8_mse, u8_psnr = mse_psnr(full_u8, stream_u8, 255.0)
    u8_exact_equal = bool(torch.equal(full_u8, stream_u8))

    result = {
        'latent_path': latent_path,
        'meta': meta,
        'full_decode': {
            'shape': list(full_video.shape),
            'time_sec': full_sec,
            'fps_equivalent': float(full_video.shape[1]) / full_sec if full_sec > 0 else None,
        },
        'stream_decode': {
            'shape': list(stream_video.shape),
            'time_sec': stream_sec,
            'fps_equivalent': float(stream_video.shape[1]) / stream_sec if stream_sec > 0 else None,
        },
        'time_compare': {
            'stream_minus_full_sec': stream_sec - full_sec,
            'stream_over_full_ratio': (stream_sec / full_sec) if full_sec > 0 else None,
            'stream_slower_percent': ((stream_sec / full_sec) - 1.0) * 100.0 if full_sec > 0 else None,
        },
        'quality_compare': {
            'same_shape': list(full_video.shape) == list(stream_video.shape),
            'float_exact_equal': bool(torch.equal(full_video, stream_video)),
            'float_mse': float_mse,
            'float_mae': float_mae,
            'float_max_abs': float_max_abs,
            'uint8_exact_equal': u8_exact_equal,
            'uint8_mse': u8_mse,
            'uint8_psnr': u8_psnr,
        },
    }

    del full_video, stream_video, full_u8, stream_u8
    maybe_cleanup()
    return result


def mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def main() -> None:
    args = parse_args()
    defaults = DEFAULTS[args.family]
    latent_paths = args.latent_path or defaults['latent_paths']
    streamer = WanStreamingDecoder(
        family=args.family,
        ckpt_dir=args.ckpt_dir or defaults['ckpt_dir'],
        wan_root=args.wan_root or defaults['wan_root'],
        device=args.device,
        model_dtype=args.model_dtype,
        use_torch_compile=args.use_torch_compile,
        compile_mode=args.compile_mode,
        compile_backend=args.compile_backend,
        stream_conv2_cache=args.stream_conv2_cache,
    )

    cases = []
    for path in latent_paths:
        cases.append(compare_case(streamer, path, args.latent_group_size, args.warmup_runs))

    summary = {
        'case_count': len(cases),
        'mean_full_decode_sec': mean([c['full_decode']['time_sec'] for c in cases]),
        'mean_stream_decode_sec': mean([c['stream_decode']['time_sec'] for c in cases]),
        'mean_stream_over_full_ratio': mean([c['time_compare']['stream_over_full_ratio'] for c in cases]),
        'mean_stream_slower_percent': mean([c['time_compare']['stream_slower_percent'] for c in cases]),
        'mean_uint8_psnr': mean([c['quality_compare']['uint8_psnr'] for c in cases]),
        'mean_uint8_mse': mean([c['quality_compare']['uint8_mse'] for c in cases]),
        'all_same_shape': all(c['quality_compare']['same_shape'] for c in cases),
        'all_float_exact_equal': all(c['quality_compare']['float_exact_equal'] for c in cases),
        'all_uint8_exact_equal': all(c['quality_compare']['uint8_exact_equal'] for c in cases),
        'max_float_max_abs': max(c['quality_compare']['float_max_abs'] for c in cases) if cases else None,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'family': args.family,
        'latent_group_size': args.latent_group_size,
        'use_torch_compile': args.use_torch_compile,
        'compile_mode': args.compile_mode if args.use_torch_compile else None,
        'compile_backend': args.compile_backend if args.use_torch_compile else None,
        'stream_conv2_cache': args.stream_conv2_cache,
        'warmup_runs': args.warmup_runs,
        'cases': cases,
        'summary': summary,
    }, indent=2))
    print(out)


if __name__ == '__main__':
    main()
