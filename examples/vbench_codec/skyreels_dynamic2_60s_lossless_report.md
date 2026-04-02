# SkyReels Dynamic 2x60s Lossless Compression Report

This note records the lossless latent compression results for the two `60s`, `720p`, `24fps` videos generated with `SkyReels-V2-DF-14B-720P` under:

- `examples/vbench_codec/skyreels_dynamic2_60s_lossless/neon_hoverbike_chain_reaction_report.md`
- `examples/vbench_codec/skyreels_dynamic2_60s_lossless/wingsuit_rescue_extreme_run_report.md`

Common setup:

- latent shape: `16 x 365 x 90 x 160`
- latent dtype: `bfloat16`
- codecs tested: `zstd`, `openzl`, `pcodec`
- XOR transform: reversible blockwise `base + XOR residuals` over time-major `TCHW`
- XOR windows tested: `2`, `4`, `8`, `16`, `32`, `64`

## 1. Neon Hoverbike Chain Reaction

Source file:

- `examples/vbench_codec/skyreels_dynamic2_60s_lossless/neon_hoverbike_chain_reaction_report.md`

Sizes:

| item | bytes |
|---|---:|
| latent `.pt` | `168,193,859` |
| latent payload | `168,192,000` |
| MP4 | `246,934,632` |
| raw RGB24 frames | `4,028,313,600` |
| raw YUV420 frames | `2,014,156,800` |

Baseline lossless latent codecs:

| codec | bytes |
|---|---:|
| `zstd` | `117,693,468` |
| `openzl` | `117,020,719` |
| `pcodec` | `98,329,215` |

Best XOR result by window:

| window | best XOR codec | bytes | delta vs best baseline |
|---:|---|---:|---:|
| `2` | `pcodec_xor` | `107,429,454` | `+9,100,239` |
| `4` | `pcodec_xor` | `109,300,591` | `+10,971,376` |
| `8` | `pcodec_xor` | `111,556,210` | `+13,226,995` |
| `16` | `pcodec_xor` | `113,121,849` | `+14,792,634` |
| `32` | `pcodec_xor` | `113,816,671` | `+15,487,456` |
| `64` | `pcodec_xor` | `113,960,762` | `+15,631,547` |

MP4 compression ratio versus raw frames:

- RGB24: `16.313279x`
- YUV420: `8.156640x`

## 2. Wingsuit Rescue Extreme Run

Source file:

- `examples/vbench_codec/skyreels_dynamic2_60s_lossless/wingsuit_rescue_extreme_run_report.md`

Sizes:

| item | bytes |
|---|---:|
| latent `.pt` | `168,193,845` |
| latent payload | `168,192,000` |
| MP4 | `235,025,186` |
| raw RGB24 frames | `4,028,313,600` |
| raw YUV420 frames | `2,014,156,800` |

Baseline lossless latent codecs:

| codec | bytes |
|---|---:|
| `zstd` | `122,247,295` |
| `openzl` | `122,849,331` |
| `pcodec` | `103,542,248` |

Best XOR result by window:

| window | best XOR codec | bytes | delta vs best baseline |
|---:|---|---:|---:|
| `2` | `pcodec_xor` | `110,576,505` | `+7,034,257` |
| `4` | `pcodec_xor` | `113,718,146` | `+10,175,898` |
| `8` | `pcodec_xor` | `115,934,470` | `+12,392,222` |
| `16` | `pcodec_xor` | `117,269,985` | `+13,727,737` |
| `32` | `pcodec_xor` | `117,207,192` | `+13,664,944` |
| `64` | `pcodec_xor` | `117,493,129` | `+13,950,881` |

MP4 compression ratio versus raw frames:

- RGB24: `17.139923x`
- YUV420: `8.569962x`

## Bottom Line

Across both `60s` `SkyReels-V2-DF-14B-720P` samples:

- `pcodec` is the best plain lossless latent codec of the three tested.
- `openzl` and `zstd` are close to each other, but both are clearly larger than `pcodec`.
- blockwise `base + XOR residuals` does not help on these `bfloat16` latents.
- the best XOR setting on both samples is `window=2` with `pcodec_xor`, but it is still worse than plain `pcodec`.
- the final MP4 is larger than the best lossless latent codec, but much smaller than raw RGB24/YUV420 frame storage.
