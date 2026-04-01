---
name: latent-codec-bench
description: Compress saved video latents, reconstruct them, and evaluate storage savings and quality. Use when working on intra/inter latent compression, zstd-based latent containers, Wan latent codec experiments, or full-report generation over fixed prompt sets.
---

# Latent Codec Bench

Use this skill when the task is to compress saved latents, decode them back, and compare size-versus-quality tradeoffs.

## Start Here

- Read [references/methods.md](references/methods.md) for the implemented codec workflows.
- Read [references/findings.md](references/findings.md) for the main quantitative conclusions.

## Main scripts

- `scripts/wan22_zstd_codec.py`
- `scripts/run_wan22_latent_codec_eval.py`
- `scripts/wan_latent_codec_bench.py`
- `scripts/eval_wan22_lighttae_samples.py`

## Supported families in this repo

Primary implemented paths are based on Wan2.2 TI2V saved latents.

## Default workflow

1. Start from saved final latents before VAE decode.
2. Choose an intra or inter codec family.
3. Reconstruct latents.
4. Decode reconstructed latents to video.
5. Compare raw-frame and MP4-level quality metrics.
6. Keep both lightweight reports and a small curated playable sample pack.

## Main codec families

- `intra_q8_zstd`
- `inter_delta_q8_zstd`
- additional experimental value codecs in `wan_latent_codec_bench.py`

## Key caution

Inter codecs may save more space but can accumulate temporal error and are less friendly to random access.
