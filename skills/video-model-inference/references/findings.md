# Findings

## Wan2.2

- `Wan2.2-TI2V-5B` 64-sample VBench generation and latent export were completed successfully.
- Fixed-prompt manifests were used to make later codec/decode comparisons reproducible.

## HunyuanVideo 1.5

### Official diffusers
- `720p / 121 frames / 50 steps` with `pipeline offloading + VAE tiling` still OOMed on a single 96GB GPU.
- Reducing duration to about `2s` (`49` frames) allowed official diffusers sampling to proceed.
- `81 frames / 16 fps` was more feasible than `121 frames`, but wall time remained very high.

### Official native repo
- Using official native code with FlashAttention materially improved step time.
- A sampled `720p / 121f / 50 steps` run used roughly `79GB` VRAM during denoising.
- Native latent export was added via a minimal patch outside this repo during experimentation.

## LTX-Video 0.9.8

- Diffusers generation with latent export worked on this machine.
- Example successful smoke test:
  - `704x512`
  - `81 frames`
  - `16 fps`
  - `30 steps`
- See `examples/vbench_generation/ltxvideo_smoketest/` for sample artifacts.

## LTX-2 official two-stage pipeline

- The official single-GPU target config
  - `768x512`
  - `121 frames`
  - `24 fps`
  - `40 steps`
  - latent export enabled
  still OOMed on a single 96GB GPU.
- Reducing to `704x512`, `640x448`, and `640x384` still OOMed at the first denoising step when using the `fp8` checkpoint.
- Reducing frames from `121` to `97` helped, but `fp8` still OOMed at the first step due to an `fp8 upcast` peak.
- The local `fp4` checkpoint did not load cleanly into the current official builder path; it failed with weight shape mismatches before inference.

## Recommendation

For single-GPU work, treat `LTX-2` as a model that still needs more aggressive memory reduction or a different official loading path than the current wrapper uses. Do not promise exact official `768x512 / 121f / 40 steps` success on 96GB based on current local evidence.
