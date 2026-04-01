#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


DEFAULTS = {
    'wan_t2v_a14b': {
        'latent_path': '/workspace/outputs/wan22_t2v_a14b_720p16_lighthouse_49f_latents.pt',
        'ckpt_dir': '/workspace/models/Wan2.2-T2V-A14B',
        'wan_root': '/root/Wan2.2',
        'vae_kind': 'wan2_1',
        'fps': 16,
    },
    'wan_ti2v_5b': {
        'latent_path': '/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/000_subject_consistency_r00_p003_a_person_eating_a_burger.pt',
        'ckpt_dir': '/workspace/models/Wan2.2-TI2V-5B',
        'wan_root': '/root/Wan2.2',
        'vae_kind': 'wan2_2',
        'fps': 24,
    },
}


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


def load_latents(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    blob = torch.load(path, map_location='cpu')
    if isinstance(blob, dict):
        meta = {k: v for k, v in blob.items() if k != 'latents'}
        latents = blob['latents']
    else:
        meta = {}
        latents = blob
    return latents.detach().cpu(), meta


def maybe_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def maybe_mark_compile_step_begin() -> None:
    compiler = getattr(torch, 'compiler', None)
    if compiler is None:
        return
    marker = getattr(compiler, 'cudagraph_mark_step_begin', None)
    if marker is not None:
        marker()


class WanStreamingDecoder:
    def __init__(
        self,
        family: str,
        ckpt_dir: str,
        wan_root: str,
        device: str,
        model_dtype: str,
        use_torch_compile: bool = False,
        compile_mode: str = 'default',
        compile_backend: str = 'inductor',
        stream_conv2_cache: bool = False,
    ):
        self.family = family
        self.device = device
        self.model_dtype = torch_dtype_from_name(model_dtype)
        self.use_torch_compile = use_torch_compile
        self.compile_mode = compile_mode
        self.compile_backend = compile_backend
        self.stream_conv2_cache = stream_conv2_cache
        self.compile_options = {'triton.cudagraphs': False}
        sys.path.insert(0, wan_root)
        if family == 'wan_t2v_a14b':
            from wan.modules.vae2_1 import Wan2_1_VAE  # noqa: WPS433
            self.wrapper = Wan2_1_VAE(
                vae_pth=str(Path(ckpt_dir) / 'Wan2.1_VAE.pth'),
                dtype=self.model_dtype,
                device=device,
            )
            self.model = self.wrapper.model
            self.unpatchify = None
            self.needs_first_chunk_flag = False
        else:
            from wan.modules.vae2_2 import Wan2_2_VAE, unpatchify  # noqa: WPS433
            self.wrapper = Wan2_2_VAE(
                vae_pth=str(Path(ckpt_dir) / 'Wan2.2_VAE.pth'),
                dtype=self.model_dtype,
                device=device,
            )
            self.model = self.wrapper.model
            self.unpatchify = unpatchify
            self.needs_first_chunk_flag = True
        self.scale = self.wrapper.scale
        self.z_dim = self.wrapper.model.z_dim
        self.conv2_cache_t = int(getattr(self.model.conv2, '_padding', (0, 0, 0, 0, 0, 0))[4])
        self._conv2_cache_x = None
        self.started = False
        self.latents_consumed = 0
        self.frames_emitted = 0
        self._maybe_compile_modules()
        self.reset_stream()

    def _maybe_compile_modules(self) -> None:
        if not self.use_torch_compile:
            return
        if not hasattr(torch, 'compile'):
            raise RuntimeError('torch.compile is not available in this environment')
        compile_kwargs = {
            'backend': self.compile_backend,
            'fullgraph': False,
        }
        if self.compile_options:
            compile_kwargs['options'] = self.compile_options
        else:
            compile_kwargs['mode'] = self.compile_mode
        # Compile the heavy decode path while keeping the cache orchestration in Python.
        self.model.conv2 = torch.compile(
            self.model.conv2,
            **compile_kwargs,
        )
        self.model.decoder = torch.compile(
            self.model.decoder,
            **compile_kwargs,
        )

    def reset_stream(self) -> None:
        self.model.clear_cache()
        self._conv2_cache_x = None
        self.started = False
        self.latents_consumed = 0
        self.frames_emitted = 0

    def _descale(self, z: torch.Tensor) -> torch.Tensor:
        if isinstance(self.scale[0], torch.Tensor):
            return z / self.scale[1].view(1, self.z_dim, 1, 1, 1) + self.scale[0].view(1, self.z_dim, 1, 1, 1)
        return z / self.scale[1] + self.scale[0]

    def decode_full(self, latents_cthw: torch.Tensor) -> torch.Tensor:
        if self.use_torch_compile:
            maybe_mark_compile_step_begin()
        return self.wrapper.decode([latents_cthw.to(self.device, dtype=torch.float32)])[0].detach().cpu()

    def push_latent_chunk(self, latents_cthw: torch.Tensor) -> torch.Tensor:
        if self.use_torch_compile:
            maybe_mark_compile_step_begin()
        z = latents_cthw.unsqueeze(0).to(self.device, dtype=torch.float32)
        z = self._descale(z)
        outs = []
        total_steps = int(z.shape[2])
        for i in range(total_steps):
            z_i = z[:, :, i:i + 1, :, :]
            if self.use_torch_compile:
                maybe_mark_compile_step_begin()
            if self.stream_conv2_cache:
                cache_x = self._conv2_cache_x
                x_i = self.model.conv2(z_i, cache_x=cache_x)
                if self.conv2_cache_t > 0:
                    merged = z_i if cache_x is None else torch.cat([cache_x.to(z_i.device), z_i], dim=2)
                    self._conv2_cache_x = merged[:, :, -self.conv2_cache_t:, :, :].detach()
                else:
                    self._conv2_cache_x = None
            else:
                if i == 0:
                    if self.use_torch_compile:
                        maybe_mark_compile_step_begin()
                    x = self.model.conv2(z)
                x_i = x[:, :, i:i + 1, :, :]
            self.model._conv_idx = [0]
            kwargs = {}
            if self.needs_first_chunk_flag and not self.started:
                kwargs['first_chunk'] = True
            if self.use_torch_compile:
                maybe_mark_compile_step_begin()
            out_i = self.model.decoder(
                x_i,
                feat_cache=self.model._feat_map,
                feat_idx=self.model._conv_idx,
                **kwargs,
            )
            outs.append(out_i)
            self.started = True
            self.latents_consumed += 1
        out = torch.cat(outs, dim=2)
        if self.unpatchify is not None:
            out = self.unpatchify(out, patch_size=2)
        out = out.squeeze(0).detach().cpu()
        self.frames_emitted += int(out.shape[1])
        return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Stateful streaming decode for Wan latents with fixed-rate group emission.')
    parser.add_argument('--family', choices=list(DEFAULTS.keys()), required=True)
    parser.add_argument('--latent-path')
    parser.add_argument('--ckpt-dir')
    parser.add_argument('--wan-root')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model-dtype', default='bf16')
    parser.add_argument('--use-torch-compile', action='store_true')
    parser.add_argument('--compile-mode', default='default', choices=['default', 'reduce-overhead', 'max-autotune'])
    parser.add_argument('--compile-backend', default='inductor')
    parser.add_argument('--stream-conv2-cache', action='store_true', help='Use a true streaming cache for conv2 instead of applying conv2 to the whole latent chunk.')
    parser.add_argument('--latent-group-size', type=int, default=1, help='How many latent time steps to decode per emitted group.')
    parser.add_argument('--emit-mode', choices=['as_fast_as_possible', 'fixed_fps'], default='fixed_fps')
    parser.add_argument('--fps', type=float, default=None, help='Target output FPS for fixed-rate emission. Defaults to latent metadata fps.')
    parser.add_argument('--save-groups-dir', default=None, help='Optional directory to save each emitted raw frame group as .pt tensors.')
    parser.add_argument('--output-json', default=None)
    parser.add_argument('--max-groups', type=int, default=None, help='Optional cap on emitted groups for short pacing tests.')
    return parser.parse_args()


def apply_defaults(args: argparse.Namespace) -> None:
    defaults = DEFAULTS[args.family]
    for key, value in defaults.items():
        if getattr(args, key, None) in (None, ''):
            setattr(args, key, value)


def main() -> None:
    args = parse_args()
    apply_defaults(args)
    latents, meta = load_latents(args.latent_path)
    fps = float(args.fps or meta.get('fps') or DEFAULTS[args.family]['fps'])
    streamer = WanStreamingDecoder(
        family=args.family,
        ckpt_dir=args.ckpt_dir,
        wan_root=args.wan_root,
        device=args.device,
        model_dtype=args.model_dtype,
        use_torch_compile=args.use_torch_compile,
        compile_mode=args.compile_mode,
        compile_backend=args.compile_backend,
        stream_conv2_cache=args.stream_conv2_cache,
    )

    synchronize()
    t0 = time.perf_counter()
    full_video = streamer.decode_full(latents)
    synchronize()
    full_dt = time.perf_counter() - t0
    full_frames = int(full_video.shape[1])
    del full_video
    maybe_cleanup()

    streamer.reset_stream()
    save_dir = Path(args.save_groups_dir) if args.save_groups_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    chunk_reports = []
    total_groups = 0
    total_frames = 0
    next_deadline = time.perf_counter()
    for start in range(0, latents.shape[1], args.latent_group_size):
        if args.max_groups is not None and total_groups >= args.max_groups:
            break
        chunk = latents[:, start:start + args.latent_group_size].contiguous()
        synchronize()
        t_chunk_start = time.perf_counter()
        frames = streamer.push_latent_chunk(chunk)
        synchronize()
        decode_sec = time.perf_counter() - t_chunk_start
        group_frames = int(frames.shape[1])
        total_groups += 1
        total_frames += group_frames

        sleep_sec = 0.0
        emit_ts = time.perf_counter()
        if args.emit_mode == 'fixed_fps':
            group_duration = group_frames / fps
            next_deadline = max(next_deadline, emit_ts)
            next_deadline += group_duration
            sleep_sec = max(0.0, next_deadline - time.perf_counter())
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            emit_ts = time.perf_counter()

        group_path = None
        if save_dir is not None:
            group_path = save_dir / f'group_{total_groups - 1:03d}.pt'
            torch.save({
                'family': args.family,
                'group_index': total_groups - 1,
                'latent_start': start,
                'latent_len': int(chunk.shape[1]),
                'frames': frames,
                'fps': fps,
            }, group_path)

        chunk_reports.append({
            'group_index': total_groups - 1,
            'latent_start': start,
            'latent_len': int(chunk.shape[1]),
            'decoded_frames': group_frames,
            'decode_sec': decode_sec,
            'decode_fps': (group_frames / decode_sec) if decode_sec > 0 else None,
            'sleep_sec': sleep_sec,
            'emit_timestamp_sec': emit_ts,
            'output_shape': list(frames.shape),
            'saved_group_path': str(group_path) if group_path else None,
        })
        del frames
        maybe_cleanup()

    report = {
        'family': args.family,
        'latent_path': args.latent_path,
        'latent_shape': list(latents.shape),
        'use_torch_compile': args.use_torch_compile,
        'compile_mode': args.compile_mode if args.use_torch_compile else None,
        'compile_backend': args.compile_backend if args.use_torch_compile else None,
        'stream_conv2_cache': args.stream_conv2_cache,
        'latent_group_size': args.latent_group_size,
        'emit_mode': args.emit_mode,
        'target_fps': fps,
        'full_decode': {
            'frames': full_frames,
            'time_sec': full_dt,
            'output_fps_equivalent': full_frames / full_dt if full_dt > 0 else None,
        },
        'stream_decode': {
            'success': True,
            'group_count': total_groups,
            'total_frames_emitted': total_frames,
            'matches_full_frame_count': total_frames == full_frames,
            'groups': chunk_reports,
        },
        'meta': meta,
    }

    out_path = Path(args.output_json or f'/root/GenLatents/examples/{args.family}_wan_streaming_decode.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
