# Wingsuit Rescue Extreme Run Lossless Compression Report

- latent: `/workspace/video_bench/skyreels_v2_dynamic2_60s_720p/latents/wingsuit_rescue_extreme_run.pt`
- mp4: `/workspace/video_bench/skyreels_v2_dynamic2_60s_720p/videos/wingsuit_rescue_extreme_run.mp4`
- latent shape: `[16, 365, 90, 160]`
- latent dtype: `bfloat16`

## Size Summary

| item | bytes |
|---|---:|
| latent `.pt` | `168193845` |
| latent payload | `168192000` |
| mp4 | `235025186` |
| raw RGB24 frames | `4028313600` |
| raw YUV420 frames | `2014156800` |

## Baseline Lossless Codecs

| codec | bytes | verified |
|---|---:|---:|
| `zstd` | `122247295` | `True` |
| `openzl` | `122849331` | `True` |
| `pcodec` | `103542248` | `True` |

## XOR Window Codecs

| window | zstd_xor | openzl_xor | pcodec_xor | best | delta_vs_best_baseline_bytes | verified |
|---:|---:|---:|---:|---|---:|---:|
| `2` | `125710317` | `129220370` | `110576505` | `pcodec_xor` | `7034257` | `True` |
| `4` | `128824658` | `131984191` | `113718146` | `pcodec_xor` | `10175898` | `True` |
| `8` | `130502768` | `133733257` | `115934470` | `pcodec_xor` | `12392222` | `True` |
| `16` | `131102925` | `134906801` | `117269985` | `pcodec_xor` | `13727737` | `True` |
| `32` | `132526503` | `134639397` | `117207192` | `pcodec_xor` | `13664944` | `True` |
| `64` | `132293186` | `134860268` | `117493129` | `pcodec_xor` | `13950881` | `True` |

## MP4 vs Raw Frames

- RGB24 compression ratio: `17.139923037865398`
- YUV420 compression ratio: `8.569961518932699`
