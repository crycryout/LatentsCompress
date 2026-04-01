---
name: video-vae-decode
description: Benchmark, debug, or implement video VAE decode paths, including streaming decode, tiled decode, official versus unofficial decode paths, and GPU/CPU fallback behavior. Use when working on Wan, HunyuanVideo, Open-Sora, or related latent-to-video decode pipelines.
---

# Video VAE Decode

Use this skill when the task is about decoding saved latents back to video, especially when memory limits, streaming behavior, or quality drift matter.

## Start Here

- Read [references/methods.md](references/methods.md) for available scripts and decode patterns.
- Read [references/findings.md](references/findings.md) for the current conclusions on Wan, Open-Sora, and Hunyuan decode behavior.

## Main repo scripts

- `scripts/wan_streaming_decode.py`
- `scripts/benchmark_wan_streaming_decode.py`
- `scripts/check_streaming_vae_decode.py`
- `scripts/compare_wan_full_vs_stream_decode.py`
- `scripts/compare_opensora_temporal_overlap.py`
- `scripts/nonofficial_temporal_chunk_decode.py`
- `scripts/lighttaew2_2_streaming_decode.py`
- `scripts/benchmark_lighttae_streaming_decode.py`

## Default workflow

1. Decide whether the goal is exact reconstruction, streaming throughput, or memory-safe decode.
2. If official decode OOMs, confirm whether the model supports spatial tiling only or true temporal tiling.
3. Separate official findings from unofficial prototypes.
4. When quality matters, compare against full decode using PSNR/SSIM or exact frame diffs.
5. When performance matters, report both first-chunk latency and steady-state throughput.

## Key caution

A model saying it supports tiled decode does not imply it supports temporally equivalent streaming decode. For long video VAEs, time-chunking often changes numerical results unless the decoder was explicitly designed for it.
