# Latent Compression, Decode, MP4, and Quality Summary

This note consolidates the main latent-compression, decode, MP4-size, and quality-loss results produced in this repo.

Primary source files:

- `examples/vbench_codec/wan64_codec_report.md`
- `examples/vbench_codec/wan64_summary.json`
- `examples/streaming/wan_stream_decode_unified_summary.md`
- `examples/streaming/wan_t2v_a14b_full_vs_stream_compare.json`
- `examples/streaming/wan_ti2v_5b_full_vs_stream_compare_4cases.json`
- `examples/streaming/opensora_temporal_overlap_compare.json`
- `examples/streaming/hunyuan/coastline_720p121f50_accel_flash_temporal_chunked.mp4`
- `examples/streaming/hunyuan/coastline_720p121f50_accel_flash_temporal_chunked_remux.mp4`
- `examples/vbench_generation/ltxvideo_smoketest/ltx_car_704x512_81f16fps_30s_stats.json`

## 1. Wan2.2 TI2V-5B: 64-video latent codec benchmark

Dataset:

- 64 videos
- prompt set: VBench 16 dimensions x 4 prompts
- model family: `Wan2.2-TI2V-5B`

Mean baseline sizes:

| Metric | Mean |
|---|---:|
| Raw decoded frames | `327.1066 MB` |
| Saved latent `.pt` | `20.953484 MB` |
| Baseline MP4 | `13.521741 MB` |

Codec results:

| Codec | Mean latent container | Mean raw-frame PSNR | Mean MP4 PSNR | Mean MP4 SSIM |
|---|---:|---:|---:|---:|
| `intra_q8_zstd` | `4.551173 MB` | `48.830249 dB` | `46.480539 dB` | `0.993296` |
| `inter_delta_q8_zstd` | `3.930084 MB` | `41.251003 dB` | `42.876847 dB` | `0.989603` |

Takeaways:

- `intra_q8_zstd` is the safer default.
- `inter_delta_q8_zstd` is smaller, but quality drops more clearly.
- The latent containers are substantially smaller than both the raw latent `.pt` and the reconstructed raw-frame tensor size.

Compression readout:

- `intra_q8_zstd`: about `20.95 / 4.55 ~= 4.6x` latent-size reduction
- `inter_delta_q8_zstd`: about `20.95 / 3.93 ~= 5.3x` latent-size reduction

Interpretation:

- `intra_q8_zstd` is the better "production-like" latent archival choice when quality stability matters.
- `inter_delta_q8_zstd` is the better "push size harder" option when sequential decode is acceptable.

## 2. Wan VAE decode: full decode vs streaming decode

The repo contains two kinds of decode comparison:

1. wall-clock timing: full decode vs streaming decode
2. quality difference: full decode vs streaming decode

### 2.1 Timing summary

From `examples/streaming/wan_stream_decode_unified_summary.md`.

#### Wan A14B

| Mode | Full decode | Stream decode | Stream slower than full |
|---|---:|---:|---:|
| eager, no cache | `6.0999 s` | `6.5216 s` | `6.91%` |
| eager, conv2 cache | `6.0140 s` | `6.5143 s` | `8.32%` |
| compile + warmup, no cache | `4.9511 s` | `6.4453 s` | `30.18%` |
| compile + warmup, conv2 cache | `4.9488 s` | `6.4743 s` | `30.83%` |

#### Wan TI2V-5B

| Mode | Full decode | Stream decode | Stream slower than full |
|---|---:|---:|---:|
| eager, no cache (4-case mean) | `17.3541 s` | `18.3333 s` | `5.65%` |
| compile + warmup, no cache (1 case) | `15.7413 s` | `18.4519 s` | `17.22%` |
| compile + warmup, conv2 cache (1 case) | `15.7700 s` | `18.5473 s` | `17.61%` |

Timing conclusion:

- In eager mode, streaming decode is only modestly slower than full decode.
- In compile mode, full decode speeds up more than stream decode, so the percentage gap grows.
- The `conv2` cache itself adds very little extra runtime; the main overhead comes from the streaming orchestration pattern.

### 2.2 Quality summary

#### Wan A14B

From `examples/streaming/wan_t2v_a14b_full_vs_stream_compare.json`.

| Metric | Value |
|---|---:|
| Full decode time | `6.0999 s` |
| Stream decode time | `6.5216 s` |
| Stream slower than full | `6.91%` |
| Float max abs diff | `0.2093745` |
| Uint8 MSE | `0.2058908` |
| Uint8 PSNR | `54.9944 dB` |
| Exact equality | `false` |

#### Wan TI2V-5B

From `examples/streaming/wan_ti2v_5b_full_vs_stream_compare_4cases.json`.

4-case mean:

| Metric | Value |
|---|---:|
| Mean full decode time | `17.3541 s` |
| Mean stream decode time | `18.3333 s` |
| Mean stream slower than full | `5.65%` |
| Mean uint8 MSE | `0.2906456` |
| Mean uint8 PSNR | `53.6922 dB` |
| Max float max abs | `0.6378441` |
| Exact equality | `false` |

Quality conclusion:

- Streaming decode is not numerically identical to full decode.
- The quality gap exists, but it is still small enough to stay in the "close reconstruction" regime for the tested cases.
- For Wan, the main trade-off is therefore:
  - full decode = best numeric fidelity
  - streaming decode = similar output with modest timing overhead and small quality loss

## 3. Open-Sora temporal overlap decode

From `examples/streaming/opensora_temporal_overlap_compare.json`.

Tested:

- native temporal tiled decode
- overlap `0.0`
- overlap `0.25`
- compared against full decode

Result:

| Case | overlap 0.0 vs full | overlap 0.25 vs full |
|---|---|---|
| original latent | exact match | exact match |
| synthetic extended latent | exact match | exact match |

Measured summary:

- `mean_psnr_overlap_0.0 = 100.0`
- `mean_psnr_overlap_0.25 = 100.0`
- both `overlap_0_equals_full_all_cases` and `overlap_0.25_equals_full_all_cases` are `true`

Conclusion:

- In the tested Open-Sora native path, the built-in temporal tiled decode matched full decode exactly.
- The tested `25%` overlap did not improve quality, because the no-overlap path was already exact on these cases.

## 4. Hunyuan official decode vs unofficial temporal chunk decode

Relevant artifacts:

- output video: `examples/streaming/hunyuan/coastline_720p121f50_accel_flash_temporal_chunked.mp4`
- remuxed output: `examples/streaming/hunyuan/coastline_720p121f50_accel_flash_temporal_chunked_remux.mp4`

What happened:

- Official Hunyuan GPU VAE decode for a saved `121`-frame latent OOMed even after using the official low-memory decode path.
- The core limitation is that the official VAE supports spatial tiling, but not temporal tiling.
- A nonofficial temporal-chunk decode prototype reused the official VAE logic while unlocking temporal chunking at instance level.

Outcome:

- Official whole-sequence GPU decode: OOM
- Unofficial temporal chunk GPU decode: success
- Produced a valid MP4 and a remuxed compatibility version

Interpretation:

- This is a decode-architecture limitation, not evidence that the latent itself is invalid.
- For long Hunyuan clips, temporal chunking is the decisive missing piece for GPU decode viability.

## 5. LTX-Video 0.9.8 saved-latent generation footprint

From `examples/vbench_generation/ltxvideo_smoketest/ltx_car_704x512_81f16fps_30s_stats.json`.

Configuration:

- model: `LTX-Video-0.9.8-13B-distilled`
- resolution: `704x512`
- frames: `81`
- fps: `16`
- steps: `30`

Recorded sizes:

| Metric | Value |
|---|---:|
| Raw decoded video tensor | `334.125 MiB` |
| MP4 | `0.7409 MiB` |
| Saved latent raw bytes | `1.890625 MiB` |
| Saved latent `.pt` | `1.8929 MiB` |
| Latent shape | `[1, 3872, 128]` |
| Latent stage | `pre_vae_decode_final` |

Interpretation:

- LTX's heavily compressed latent space produces very small pre-VAE latent payloads.
- The gap between raw decoded tensor size and saved latent size is much larger than in the Wan cases.
- This is consistent with LTX's more aggressive latent compression design.

## 6. Practical conclusions

### Latent compression

- `Wan intra_q8_zstd` is the strongest default choice when quality matters.
- `Wan inter_delta_q8_zstd` is best when container size matters more than reconstruction fidelity.

### Decode

- Wan streaming decode is viable and close to full decode, but not numerically exact.
- Open-Sora native temporal tiled decode can be exact.
- Hunyuan official GPU decode for long clips is blocked mainly by lack of temporal tiling.

### MP4 output

- The repo contains both raw outputs and a remuxed Hunyuan MP4 for compatibility.
- When a file looks "damaged" in a picky player, remuxing the container can fix playback without re-encoding.

### Precision loss

- Wan latent compression shows the clearest quantitative quality story:
  - `intra_q8_zstd`: strong size reduction with small quality loss
  - `inter_delta_q8_zstd`: smaller files with noticeably larger quality loss
- Wan streaming decode quality loss is real but still limited.
- Open-Sora native overlap tests showed no measurable loss relative to full decode for the tested cases.
