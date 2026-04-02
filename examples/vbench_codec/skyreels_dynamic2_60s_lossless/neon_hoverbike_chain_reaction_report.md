# Neon Hoverbike Chain Reaction Lossless Compression Report

- latent: `/workspace/video_bench/skyreels_v2_dynamic2_60s_720p/latents/neon_hoverbike_chain_reaction.pt`
- mp4: `/workspace/video_bench/skyreels_v2_dynamic2_60s_720p/videos/neon_hoverbike_chain_reaction.mp4`
- latent shape: `[16, 365, 90, 160]`
- latent dtype: `bfloat16`

## Size Summary

| item | bytes |
|---|---:|
| latent `.pt` | `168193859` |
| latent payload | `168192000` |
| mp4 | `246934632` |
| raw RGB24 frames | `4028313600` |
| raw YUV420 frames | `2014156800` |

## Baseline Lossless Codecs

| codec | bytes | verified |
|---|---:|---:|
| `zstd` | `117693468` | `True` |
| `openzl` | `117020719` | `True` |
| `pcodec` | `98329215` | `True` |

## XOR Window Codecs

| window | zstd_xor | openzl_xor | pcodec_xor | best | delta_vs_best_baseline_bytes | verified |
|---:|---:|---:|---:|---|---:|---:|
| `2` | `120314715` | `123873464` | `107429454` | `pcodec_xor` | `9100239` | `True` |
| `4` | `125235865` | `125483918` | `109300591` | `pcodec_xor` | `10971376` | `True` |
| `8` | `127205297` | `127343157` | `111556210` | `pcodec_xor` | `13226995` | `True` |
| `16` | `128025981` | `128806024` | `113121849` | `pcodec_xor` | `14792634` | `True` |
| `32` | `128404336` | `129747573` | `113816671` | `pcodec_xor` | `15487456` | `True` |
| `64` | `128528632` | `130291832` | `113960762` | `pcodec_xor` | `15631547` | `True` |

## MP4 vs Raw Frames

- RGB24 compression ratio: `16.31327921633933`
- YUV420 compression ratio: `8.156639608169664`
