# Findings

## Single-sample pattern

The dominant pattern was:
- `inter_delta_q8_zstd` -> smaller compressed latent
- `intra_q8_zstd` -> better reconstruction quality and more stable behavior

## 64-sample Wan2.2 TI2V report

Artifacts:
- `examples/vbench_codec/wan64_codec_report.json`
- `examples/vbench_codec/wan64_codec_report.csv`
- `examples/vbench_codec/wan64_codec_report.md`
- `examples/vbench_codec/wan64_summary.json`

Mean results across 64 samples:
- raw frame size: about `327.11 MB`
- original latent `.pt`: about `20.95 MB`
- original MP4: about `13.52 MB`

### intra_q8_zstd
- compressed latent: about `4.55 MB`
- raw-frame PSNR: about `48.83 dB`
- MP4 PSNR: about `46.48 dB`
- MP4 SSIM: about `0.9933`

### inter_delta_q8_zstd
- compressed latent: about `3.93 MB`
- raw-frame PSNR: about `41.25 dB`
- MP4 PSNR: about `42.88 dB`
- MP4 SSIM: about `0.9896`

## Practical recommendation

- Use `intra_q8_zstd` when quality and simplicity matter most.
- Use `inter_delta_q8_zstd` when storage pressure is high and a small additional quality drop is acceptable.
