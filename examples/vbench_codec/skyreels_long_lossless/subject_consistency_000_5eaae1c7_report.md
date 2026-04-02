# SkyReels Long Latent Lossless Codec Comparison

- latent: `/workspace/video_bench/skyreels_v2_vbench2_60s_720p/latents/subject_consistency/subject_consistency_000_5eaae1c7.pt`
- mp4: `/workspace/video_bench/skyreels_v2_vbench2_60s_720p/videos/subject_consistency/subject_consistency_000_5eaae1c7.mp4`
- latent shape: `[16, 365, 90, 160]`
- latent dtype: `bfloat16`

## Size Summary

| item | bytes | MB | MiB |
|---|---:|---:|---:|
| original latent `.pt` | `168193880` | `168.19388` | `160.402184` |
| original latent payload | `168192000` | `168.192` | `160.400391` |
| raw video RGB24 | `4028313600` | `4028.3136` | `3841.699219` |
| raw video YUV420 | `2014156800` | `2014.1568` | `1920.849609` |
| generated MP4 | `142925494` | `142.925494` | `136.304373` |

## Lossless Codec Results

| codec | target | compressed bytes | MB | MiB | ratio vs payload | ratio vs `.pt` | verified lossless |
|---|---|---:|---:|---:|---:|---:|---:|
| `zstd` | `latent_payload` | `102685993` | `102.685993` | `97.928994` | `1.637925` | `1.637944` | `True` |
| `openzl` | `latent_payload` | `105590684` | `105.590684` | `100.699123` | `1.592868` | `1.592886` | `True` |
| `pcodec` | `latent_payload` | `85940706` | `85.940706` | `81.959444` | `1.95707` | `1.957092` | `True` |

## Video Context

- resolution: `1280x720` at `24.0` fps
- frames: `1457`
- duration: `60.708333` s

## Notes

- ZSTD and OpenZL are run on the exact latent payload bytes extracted from the saved tensor.
- Pcodec is run on a `uint16` view of the same `bfloat16` payload, so the bit-pattern is preserved exactly.
- The saved `.pt` container overhead is tiny, so payload-vs-file comparisons differ only slightly.
