# Session Summary

This file is the best resume artifact for continuing this project on a new machine. It complements the raw Codex chat logs in the same directory.

## Scope

This project covered four main tracks:

1. Generating and saving pre-VAE latents for multiple T2V models.
2. Large-scale Wan2.2 VBench prompt generation and latent collection.
3. Latent compression experiments on Wan2.2 latents.
4. VAE streaming decode implementation and benchmarking for Wan/Open-Sora.

## Main Repositories

- `GenLatents`: `git@github.com:crycryout/GenLatents.git`
- `LatentsCompress`: `git@github.com:crycryout/LatentsCompress.git`

## Key Local Repos Used During the Session

- `/root/GenLatents`
- `/root/LatentsCompress`
- `/root/Wan2.2`
- `/root/Open-Sora`
- `/workspace/video_bench`

## Important Results

### 1. Wan2.2 VBench Generation

The full Wan2.2-TI2V-5B VBench run completed successfully.

- Output root: `/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42`
- Videos: `/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/native_16fps`
- Latents: `/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents`

Run properties:
- Model: `Wan2.2-TI2V-5B`
- Samples: `64`
- Sampling: `16 dimensions x 4 prompts`
- Seed: `42`
- Latents saved: final denoised latents before VAE decode

### 2. Full 64-sample Wan Latent Codec Benchmark

Complete report files:

- `/workspace/video_bench/latent_codec/runs/wan22_ti2v5b_vbench_16x4_seed42_q8_64/wan64_codec_report.json`
- `/workspace/video_bench/latent_codec/runs/wan22_ti2v5b_vbench_16x4_seed42_q8_64/wan64_codec_report.csv`
- `/workspace/video_bench/latent_codec/runs/wan22_ti2v5b_vbench_16x4_seed42_q8_64/wan64_codec_report.md`
- `/workspace/video_bench/latent_codec/runs/wan22_ti2v5b_vbench_16x4_seed42_q8_64/summary.json`

Lightweight copies are also stored in this repo under:
- `examples/vbench_codec/`

Mean results across 64 samples:
- Raw frame size: `327.1066 MB`
- Latent `.pt` size: `20.953484 MB`
- MP4 size: `13.521741 MB`
- `intra_q8_zstd` latent size: `4.551173 MB`
- `inter_delta_q8_zstd` latent size: `3.930084 MB`
- `intra_q8_zstd` raw-frame PSNR: `48.830249 dB`
- `inter_delta_q8_zstd` raw-frame PSNR: `41.251003 dB`
- `intra_q8_zstd` MP4 SSIM: `0.993296`
- `inter_delta_q8_zstd` MP4 SSIM: `0.989603`

Interpretation:
- `intra_q8_zstd` is the safer quality-preserving default.
- `inter_delta_q8_zstd` is smaller, but quality drops more.

### 3. Wan/Open-Sora Streaming Decode

Implemented and benchmarked scripts:
- `scripts/wan_streaming_decode.py`
- `scripts/check_streaming_vae_decode.py`
- `scripts/benchmark_wan_streaming_decode.py`
- `scripts/compare_wan_full_vs_stream_decode.py`
- `scripts/compare_opensora_temporal_overlap.py`

Stored reports:
- `examples/streaming/`

Important conclusions:

#### Wan
- Stateful streaming decode was implemented successfully.
- For Wan, `latent_group_size=1` yields:
  - first group: `1` output frame
  - later groups: `4` output frames each
- Total output frame count matches full decode.

Quality/time comparison vs full decode:
- Streaming decode is slightly slower than full decode.
- Streaming decode is not pixel-identical to full decode.
- Typical quality difference is small but measurable.

Compile results:
- `torch.compile` improves full decode more than streaming decode.
- Main reason: full decode is a large stable graph, while streaming decode is a small-step stateful loop with dynamic cache transitions.

#### Open-Sora 2.0
- Native temporal tiled decode was compared against full decode.
- In the tested `Open-Sora v2 / hunyuan_vae` path, temporal tiling matched full decode exactly in the tested cases.
- `25% overlap` did not improve quality in those tests, because both overlap and no-overlap already matched full decode.

## VBench Prompt Assets

The fixed manifests used during the session are included under:
- `examples/vbench_generation/wan22_vbench_16x20_seed42.json`
- `examples/vbench_generation/wan22_vbench_16x4_seed42.json`
- `examples/vbench_generation/wan22_vbench_16x2_seed42.json`

Generation helper scripts included in this repo:
- `scripts/build_vbench_wan22_manifest.py`
- `scripts/run_wan22_vbench_batch.py`
- `scripts/run_diffusers_video_bench.py`
- `scripts/run_diffusers_vbench_batch.py`
- `scripts/run_post_wan_vbench_jobs.py`
- `scripts/check_video_model_env.py`

## Other Model Plans and Status

The planned follow-up models using the same VBench prompts were:
- `HunyuanVideo 1.5` (`8.3B`)
- `Mochi 1` (`10B`)
- `LTX-Video` (`13B`)

Target plan after reduction:
- `32` videos per model
- `16 dimensions x 2 prompts`

Important blocker encountered:
- First Hunyuan run failed before generation because Hugging Face cache writes under `/workspace` hit `Disk quota exceeded`.
- This was not overall filesystem exhaustion. `/workspace` still had very large free space, but the active write path or workspace quota blocked cache writes.

## Storage Findings

Approximate `/workspace` usage observed during the session:
- `/workspace/hf-cache`: `245 GB`
- `/workspace/models`: `192 GB`
- `/workspace/.cache`: `24 GB`
- `/workspace/video_bench`: `9.3 GB`

Interpretation:
- Most storage usage came from model weights and Hugging Face caches.
- Generated videos and latent results were a relatively small fraction.

## Important Implementation Notes

### Final Latent Saving

The generation scripts were adjusted so saved latents are clearly marked as:
- stage: `pre_vae_decode_final`
- description: `Final denoised latents captured before VAE decode.`

This was added to the diffusers-side generation path so the saved `.pt` files are explicitly documented as final latents before VAE decode.

### GitHub and SSH

GitHub SSH access was restored by writing `/root/.ssh/config` to explicitly use:
- `/root/.ssh/github_latentscompress_ed25519`

This enabled pushes to both:
- `crycryout/GenLatents`
- `crycryout/LatentsCompress`

## Best Resume Strategy On a New Machine

Do not assume Codex will directly resume the raw session files as a native live session.
Treat these artifacts as project memory.

Recommended resume order:

1. Clone `GenLatents` and `LatentsCompress`.
2. Read this file first: `examples/codex_chat/SESSION_SUMMARY.md`
3. Then inspect raw logs in:
   - `examples/codex_chat/history.jsonl`
   - `examples/codex_chat/session_index.jsonl`
   - `examples/codex_chat/sessions/2026/03/31/*.jsonl`
4. For code/results, use:
   - `examples/streaming/`
   - `examples/vbench_generation/`
   - `examples/vbench_codec/`

## Suggested Next Steps

If continuing the project, the most natural next tasks are:

1. Fix the `/workspace` Hugging Face cache quota/path issue and rerun the planned `HunyuanVideo 1.5 / Mochi 1 / LTX-Video` jobs.
2. Improve Wan streaming decode further by exploring overlap/stitching or a more native tiled decode path.
3. If needed, publish large binary outputs separately using Git LFS or external storage, instead of regular Git.
