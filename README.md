# LatentsCompress

Latent compression experiments for Wan2.2 TI2V outputs.

This repo contains:
- `scripts/wan_latent_codec_bench.py`: main benchmark script for intra/inter latent compression and reconstruction quality evaluation.
- `scripts/run_wan22_latent_codec_eval.py`: evaluation script for zstd container experiments using baseline MP4 re-encoding.
- `scripts/wan22_zstd_codec.py`: latent container encode/decode helpers.
- `examples/`: one-sample benchmark summary and report.

## Implemented schemes

Two core families are implemented:
- `intra`: compress each latent independently.
- `inter`: add temporal prediction across adjacent latent frames, with keyframes.

Current value codecs in `wan_latent_codec_bench.py`:
- `qint8`
- `qint6`
- `qint4`
- `fp16`

Current value codecs in `run_wan22_latent_codec_eval.py` / `wan22_zstd_codec.py`:
- `intra_fp16_zstd`
- `inter_delta_fp16_zstd`
- `intra_q8_zstd`
- `inter_delta_q8_zstd`

## External prerequisites

This repo assumes you already have:
- a local Wan2.2 checkout at `/root/Wan2.2`
- Wan2.2 TI2V latent `.pt` files and generated MP4 files
- `ffmpeg`
- Python packages from `requirements.txt`

The scripts are written against the local Wan environment used during the benchmark work and import Wan modules directly.

## Example artifact

Included example files come from one benchmarked sample:
- original latent `.pt`: about `20.95 MB`
- original MP4: about `8.19 MB`
- `intra` compressed latent: about `4.84 MB`
- `inter` compressed latent: about `4.59 MB`

From the included example summary:
- `inter` achieved a smaller latent container than `intra`
- MP4 reconstruction quality stayed high, with PSNR around `49 dB` and SSIM around `0.992`

## Notes

This repository intentionally excludes heavy generated videos, model weights, and temporary debug outputs.
