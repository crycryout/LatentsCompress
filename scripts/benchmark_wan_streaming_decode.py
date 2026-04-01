#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from pathlib import Path
from typing import Any

# Reuse the validated streamer implementation.
sys.path.insert(0, '/root/GenLatents/scripts')
from wan_streaming_decode import (  # type: ignore
    DEFAULTS,
    WanStreamingDecoder,
    load_latents,
    maybe_cleanup,
    synchronize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Benchmark Wan streaming VAE decode throughput and playback feasibility.')
    parser.add_argument('--family', choices=list(DEFAULTS.keys()), required=True)
    parser.add_argument('--latent-path', action='append', default=[], help='Repeatable. If omitted, uses family default latent.')
    parser.add_argument('--ckpt-dir', default=None)
    parser.add_argument('--wan-root', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model-dtype', default='bf16')
    parser.add_argument('--use-torch-compile', action='store_true')
    parser.add_argument('--compile-mode', default='default', choices=['default', 'reduce-overhead', 'max-autotune'])
    parser.add_argument('--compile-backend', default='inductor')
    parser.add_argument('--stream-conv2-cache', action='store_true')
    parser.add_argument('--latent-group-size', type=int, default=1)
    parser.add_argument('--target-fps', type=float, default=None, help='Playback FPS to compare against; defaults to family fps or latent metadata fps.')
    parser.add_argument('--warmup-runs', type=int, default=1)
    parser.add_argument('--measure-runs', type=int, default=2)
    parser.add_argument('--output-json', required=True)
    return parser.parse_args()


def mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def median(xs: list[float]) -> float | None:
    return stats.median(xs) if xs else None


def pstdev(xs: list[float]) -> float | None:
    return stats.pstdev(xs) if len(xs) > 1 else 0.0 if xs else None


def cv(xs: list[float]) -> float | None:
    if not xs:
        return None
    m = mean(xs)
    if not m:
        return None
    s = pstdev(xs)
    return s / m if s is not None else None


def bench_once(streamer: WanStreamingDecoder, latents, target_fps: float, latent_group_size: int) -> dict[str, Any]:
    synchronize()
    import time
    t0 = time.perf_counter()
    full_video = streamer.decode_full(latents)
    synchronize()
    full_dt = time.perf_counter() - t0
    full_frames = int(full_video.shape[1])
    del full_video
    maybe_cleanup()

    streamer.reset_stream()
    groups = []
    total_frames = 0
    stream_t0 = time.perf_counter()
    for start in range(0, latents.shape[1], latent_group_size):
        chunk = latents[:, start:start + latent_group_size].contiguous()
        synchronize()
        t1 = time.perf_counter()
        frames = streamer.push_latent_chunk(chunk)
        synchronize()
        dt = time.perf_counter() - t1
        group_frames = int(frames.shape[1])
        groups.append({
            'latent_start': start,
            'latent_len': int(chunk.shape[1]),
            'decoded_frames': group_frames,
            'decode_sec': dt,
            'decode_fps': group_frames / dt if dt > 0 else None,
        })
        total_frames += group_frames
        del frames
        maybe_cleanup()
    synchronize()
    stream_dt = time.perf_counter() - stream_t0

    first = groups[0]
    steady = groups[1:] if len(groups) > 1 else []
    steady_secs = [g['decode_sec'] for g in steady]
    steady_frames = [g['decoded_frames'] for g in steady]
    steady_total_frames = sum(steady_frames)
    steady_total_time = sum(steady_secs)
    steady_fps = steady_total_frames / steady_total_time if steady_total_time > 0 else None
    first_group_latency = first['decode_sec']
    first_group_frames = first['decoded_frames']
    first_frame_latency = first_group_latency / first_group_frames if first_group_frames > 0 else None

    realtime_ok_after_first = (steady_fps is not None and steady_fps >= target_fps)
    full_realtime_ok = (full_frames / full_dt) >= target_fps if full_dt > 0 else False

    return {
        'full_decode': {
            'frames': full_frames,
            'time_sec': full_dt,
            'fps_equivalent': full_frames / full_dt if full_dt > 0 else None,
        },
        'stream_decode': {
            'group_count': len(groups),
            'total_frames': total_frames,
            'total_time_sec': stream_dt,
            'fps_equivalent': total_frames / stream_dt if stream_dt > 0 else None,
            'first_group_latency_sec': first_group_latency,
            'first_group_frames': first_group_frames,
            'first_frame_latency_lower_bound_sec': first_frame_latency,
            'steady_group_decode_sec_mean': mean(steady_secs),
            'steady_group_decode_sec_median': median(steady_secs),
            'steady_group_decode_sec_cv': cv(steady_secs),
            'steady_group_frames_mean': mean([float(x) for x in steady_frames]),
            'steady_fps': steady_fps,
            'groups': groups,
        },
        'playback_feasibility': {
            'target_fps': target_fps,
            'realtime_possible_after_first_group': realtime_ok_after_first,
            'realtime_possible_full_decode': full_realtime_ok,
            'steady_rate_over_target': (steady_fps / target_fps) if (steady_fps is not None and target_fps > 0) else None,
            'first_group_playback_coverage_sec': first_group_frames / target_fps if target_fps > 0 else None,
            'producer_consumer_gap_sec': ((steady_total_time / steady_total_frames) - (1.0 / target_fps)) if steady_total_frames > 0 else None,
        },
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(path: list[str]) -> list[float]:
        vals = []
        for c in cases:
            cur: Any = c
            try:
                for k in path:
                    cur = cur[k]
                if cur is not None:
                    vals.append(float(cur))
            except Exception:
                pass
        return vals

    return {
        'case_count': len(cases),
        'full_decode_fps_equivalent_mean': mean(collect(['benchmark', 'full_decode', 'fps_equivalent'])),
        'stream_decode_fps_equivalent_mean': mean(collect(['benchmark', 'stream_decode', 'fps_equivalent'])),
        'first_group_latency_sec_mean': mean(collect(['benchmark', 'stream_decode', 'first_group_latency_sec'])),
        'steady_fps_mean': mean(collect(['benchmark', 'stream_decode', 'steady_fps'])),
        'steady_group_decode_sec_mean': mean(collect(['benchmark', 'stream_decode', 'steady_group_decode_sec_mean'])),
        'steady_group_decode_sec_cv_mean': mean(collect(['benchmark', 'stream_decode', 'steady_group_decode_sec_cv'])),
        'realtime_possible_after_first_group_all': all(c['benchmark']['playback_feasibility']['realtime_possible_after_first_group'] for c in cases),
        'realtime_possible_after_first_group_any': any(c['benchmark']['playback_feasibility']['realtime_possible_after_first_group'] for c in cases),
    }


def main() -> None:
    args = parse_args()
    defaults = DEFAULTS[args.family]
    latent_paths = args.latent_path or [defaults['latent_path']]
    ckpt_dir = args.ckpt_dir or defaults['ckpt_dir']
    wan_root = args.wan_root or defaults['wan_root']

    streamer = WanStreamingDecoder(
        family=args.family,
        ckpt_dir=ckpt_dir,
        wan_root=wan_root,
        device=args.device,
        model_dtype=args.model_dtype,
        use_torch_compile=args.use_torch_compile,
        compile_mode=args.compile_mode,
        compile_backend=args.compile_backend,
        stream_conv2_cache=args.stream_conv2_cache,
    )

    cases: list[dict[str, Any]] = []
    for latent_path in latent_paths:
        latents, meta = load_latents(latent_path)
        target_fps = float(args.target_fps or meta.get('fps') or defaults['fps'])

        for _ in range(args.warmup_runs):
            _ = bench_once(streamer, latents, target_fps, args.latent_group_size)
            maybe_cleanup()

        runs = []
        for _ in range(args.measure_runs):
            runs.append(bench_once(streamer, latents, target_fps, args.latent_group_size))
            maybe_cleanup()

        # choose the median run by stream total time to reduce variance
        ordered = sorted(runs, key=lambda r: r['stream_decode']['total_time_sec'])
        chosen = ordered[len(ordered) // 2]
        cases.append({
            'latent_path': latent_path,
            'meta': meta,
            'benchmark': chosen,
        })

    report = {
        'family': args.family,
        'latent_group_size': args.latent_group_size,
        'use_torch_compile': args.use_torch_compile,
        'compile_mode': args.compile_mode if args.use_torch_compile else None,
        'compile_backend': args.compile_backend if args.use_torch_compile else None,
        'stream_conv2_cache': args.stream_conv2_cache,
        'warmup_runs': args.warmup_runs,
        'measure_runs': args.measure_runs,
        'cases': cases,
        'summary': summarize_cases(cases),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(out)


if __name__ == '__main__':
    main()
