#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


def to_uint8(video: torch.Tensor) -> torch.Tensor:
    return ((video.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)


def mse_psnr(ref: torch.Tensor, dist: torch.Tensor, max_val: float = 255.0) -> tuple[float, float]:
    ref_f = ref.to(torch.float32)
    dist_f = dist.to(torch.float32)
    mse = float(torch.mean((ref_f - dist_f) ** 2).item())
    if mse == 0.0:
        return 0.0, 100.0
    return mse, float(20.0 * math.log10(max_val) - 10.0 * math.log10(mse))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare Open-Sora temporal overlap=0 vs 0.25 against full decode.')
    parser.add_argument('--latent-path', default='/workspace/video_bench/opensora_run_t2v/video_256px/sample_0_latents.pt')
    parser.add_argument('--opensora-root', default='/root/Open-Sora')
    parser.add_argument('--opensora-config', default='/root/Open-Sora/configs/diffusion/inference/t2i2v_256px.py')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model-dtype', default='bf16')
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--also-test-extended', action='store_true', help='Also test a longer synthetic latent made by concatenating shifted copies.')
    return parser.parse_args()


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        'fp32': torch.float32,
        'float32': torch.float32,
        'fp16': torch.float16,
        'float16': torch.float16,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f'Unsupported dtype: {name}')
    return mapping[key]


def build_vae(opensora_root: str, opensora_config: str, device: str, model_dtype: str):
    sys.path.insert(0, opensora_root)
    from mmengine import Config  # noqa: WPS433
    from opensora.registry import MODELS, build_module  # noqa: WPS433

    cfg = Config.fromfile(opensora_config)
    root = Path(opensora_root)
    if isinstance(cfg.ae, dict) and 'from_pretrained' in cfg.ae and isinstance(cfg.ae['from_pretrained'], str):
        src = cfg.ae['from_pretrained']
        if src.startswith('./'):
            cfg.ae['from_pretrained'] = str((root / src[2:]).resolve())

    dtype = torch_dtype_from_name(model_dtype)
    cwd = os.getcwd()
    os.chdir(opensora_root)
    try:
        vae = build_module(cfg.ae, MODELS, device_map=device, torch_dtype=dtype).eval()
    finally:
        os.chdir(cwd)
    return vae, cfg


def timed_decode(vae, z: torch.Tensor, device: str, dtype: torch.dtype) -> tuple[torch.Tensor, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = vae.decode(z.to(device, dtype=dtype)).detach().cpu()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t0)


def compare_to_ref(ref: torch.Tensor, out: torch.Tensor) -> dict[str, Any]:
    diff = (ref - out).to(torch.float32)
    float_mse = float(torch.mean(diff * diff).item())
    float_mae = float(torch.mean(torch.abs(diff)).item())
    float_max_abs = float(torch.max(torch.abs(diff)).item())
    ref_u8 = to_uint8(ref)
    out_u8 = to_uint8(out)
    u8_mse, u8_psnr = mse_psnr(ref_u8, out_u8)
    return {
        'same_shape': list(ref.shape) == list(out.shape),
        'float_exact_equal': bool(torch.equal(ref, out)),
        'uint8_exact_equal': bool(torch.equal(ref_u8, out_u8)),
        'float_mse': float_mse,
        'float_mae': float_mae,
        'float_max_abs': float_max_abs,
        'uint8_mse': u8_mse,
        'uint8_psnr': u8_psnr,
    }


def run_case(vae, z: torch.Tensor, device: str, dtype: torch.dtype, label: str) -> dict[str, Any]:
    vae.use_spatial_tiling = False
    vae.use_temporal_tiling = False
    ref, ref_sec = timed_decode(vae, z, device, dtype)

    variants = {}
    for overlap in [0.0, 0.25]:
        vae.use_spatial_tiling = False
        vae.use_temporal_tiling = True
        vae.tile_overlap_factor = overlap
        out, dt = timed_decode(vae, z, device, dtype)
        variants[str(overlap)] = {
            'time_sec': dt,
            'compare_to_full': compare_to_ref(ref, out),
        }
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        'label': label,
        'latent_shape': list(z.shape),
        'full_decode': {
            'time_sec': ref_sec,
            'shape': list(ref.shape),
        },
        'overlap_0.0': variants['0.0'],
        'overlap_0.25': variants['0.25'],
    }


def main() -> None:
    args = parse_args()
    dtype = torch_dtype_from_name(args.model_dtype)
    vae, cfg = build_vae(args.opensora_root, args.opensora_config, args.device, args.model_dtype)

    blob = torch.load(args.latent_path, map_location='cpu')
    z = blob['latents'] if isinstance(blob, dict) else blob

    cases = [run_case(vae, z, args.device, dtype, 'original_latent')]
    if args.also_test_extended:
        z_long = torch.cat([z, z[:, :, 1:], z[:, :, 1:]], dim=2).contiguous()
        cases.append(run_case(vae, z_long, args.device, dtype, 'synthetic_extended_latent'))

    summary = {
        'config_ae_type': cfg.ae['type'] if isinstance(cfg.ae, dict) and 'type' in cfg.ae else None,
        'spatial_tiling_disabled_for_test': True,
        'overlap_0_equals_full_all_cases': all(c['overlap_0.0']['compare_to_full']['float_exact_equal'] for c in cases),
        'overlap_0.25_equals_full_all_cases': all(c['overlap_0.25']['compare_to_full']['float_exact_equal'] for c in cases),
        'mean_psnr_overlap_0.0': sum(c['overlap_0.0']['compare_to_full']['uint8_psnr'] for c in cases) / len(cases),
        'mean_psnr_overlap_0.25': sum(c['overlap_0.25']['compare_to_full']['uint8_psnr'] for c in cases) / len(cases),
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'latent_path': args.latent_path,
        'opensora_config': args.opensora_config,
        'cases': cases,
        'summary': summary,
    }, indent=2))
    print(out)


if __name__ == '__main__':
    main()
