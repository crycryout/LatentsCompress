# Methods

## Diffusers model runs

Use `scripts/run_diffusers_video_bench.py` for:
- `CogVideoX`
- `HunyuanVideo 1.5` community diffusers model
- `LTX-Video 0.9.8`
- other diffusers-native video pipelines already wired in the repo

The script can:
- generate video
- save final denoised latents before VAE decode
- save a stats JSON next to the output

Typical output set:
- `<prefix>.mp4`
- `<prefix>_latents.pt`
- `<prefix>_stats.json`

## Official LTX-2 runs with latent export

Use `scripts/run_ltx2_native_save_latents.py`.

This wrapper uses the official `LTX-2` two-stage pipeline blocks and saves:
- stage 1 video latents
- optional stage 1 audio latents
- upscaled video latents
- final pre-VAE video latents
- optional final audio latents

Important runtime knobs:
- `--checkpoint-path`
- `--distilled-lora-path`
- `--spatial-upsampler-path`
- `--gemma-root`
- `--height`, `--width`
- `--num-frames`
- `--frame-rate`
- `--num-inference-steps`
- `--streaming-prefetch-count`
- `--quantization`

## Practical config rules

### LTX family
- width and height must be divisible by `32`
- frame count should follow `8n + 1`
- start from lower resolution and/or fewer frames on a single GPU

### HunyuanVideo 1.5
- be explicit whether the run is `official native` or `official diffusers`
- for long clips, log whether offload, tiling, FlashAttention, and latent export were enabled

### Wan2.2
- if later codec work is expected, always save final latents before VAE decode
- keep prompt manifests fixed for cross-model comparisons
