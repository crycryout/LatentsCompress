---
name: video-model-inference
description: Run and benchmark open video generation models with optional pre-VAE latent export. Use when working on Wan2.2, HunyuanVideo 1.5, LTX-Video, or LTX-2 inference, comparing official versus diffusers pipelines, choosing feasible single-GPU configs, or saving final denoised latents before VAE decode.
---

# Video Model Inference

Use this skill when the task is to run or benchmark video generation models and keep the workflow reproducible.

## Scope

This repo currently has working or partially working paths for:
- `Wan2.2` via local/native and helper scripts
- `HunyuanVideo 1.5` via diffusers and official native repo notes
- `LTX-Video 0.9.8` via diffusers
- `LTX-2` via official two-stage pipeline wrapper with latent export

## Start Here

- For diffusers-based runs, open [references/methods.md](references/methods.md) and use `scripts/run_diffusers_video_bench.py` first.
- For current practical conclusions and known limits, open [references/findings.md](references/findings.md).
- For official `LTX-2` latent-export runs, use [scripts/run_ltx2_native_save_latents.py](/root/LatentsCompress/scripts/run_ltx2_native_save_latents.py).

## Default workflow

1. Decide whether the model should be run through `diffusers` or its official/native repo.
2. Prefer saving `pre-VAE decode final latents` whenever the user wants downstream decode, compression, or codec experiments.
3. For long or high-resolution runs, validate feasibility with a single smoke test before starting a batch.
4. Record exact config: width, height, frame count, fps, steps, seed, quantization/offload choices.
5. Keep outputs and stats together so later codec/decode work can reuse them.

## Repo scripts

- `scripts/run_diffusers_video_bench.py`: one-shot diffusers generation with optional latent export.
- `scripts/run_diffusers_vbench_batch.py`: batch runner over a manifest.
- `scripts/run_post_wan_vbench_jobs.py`: sequential post-Wan jobs.
- `scripts/run_ltx2_native_save_latents.py`: official `LTX-2` two-stage wrapper with intermediate and final latent export.
- `scripts/build_vbench_wan22_manifest.py`: build fixed prompt manifests.

## Notes

- Single-GPU feasibility is model- and pipeline-dependent; do not assume smaller parameter count means lower peak VRAM.
- For `LTX-2`, the official two-stage path is the correct reference path, but exact single-GPU configs can still OOM.
- For `HunyuanVideo 1.5`, official native and official diffusers behave differently enough that conclusions should always mention which path was used.
