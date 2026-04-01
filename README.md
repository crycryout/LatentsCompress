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

## Streaming Decode Benchmarks

This repository now also includes Wan/Open-Sora VAE streaming decode benchmarks and comparison reports:

- Scripts are in `scripts/benchmark_wan_streaming_decode.py`, `scripts/check_streaming_vae_decode.py`, `scripts/compare_wan_full_vs_stream_decode.py`, `scripts/compare_opensora_temporal_overlap.py`, and `scripts/wan_streaming_decode.py`.
- Benchmark outputs are under `examples/streaming/`.

## VBench Prompt Generation Assets

This repository also includes the fixed-prompt VBench manifests and generation helpers used for Wan and follow-up model runs:

- Manifests: `examples/vbench_generation/wan22_vbench_16x20_seed42.json`, `wan22_vbench_16x4_seed42.json`, `wan22_vbench_16x2_seed42.json`
- Generation scripts: `scripts/build_vbench_wan22_manifest.py`, `scripts/run_wan22_vbench_batch.py`, `scripts/run_diffusers_video_bench.py`, `scripts/run_diffusers_vbench_batch.py`, `scripts/run_post_wan_vbench_jobs.py`
- Sample model stats and summaries are in `examples/vbench_generation/`.

## Full 64-Sample Wan Codec Report

The complete 64-sample Wan2.2-TI2V-5B VBench latent codec benchmark is included in lightweight report form:

- `examples/vbench_codec/wan64_codec_report.json`
- `examples/vbench_codec/wan64_codec_report.csv`
- `examples/vbench_codec/wan64_codec_report.md`
- `examples/vbench_codec/wan64_summary.json`

To make the results easier to inspect, the repository now includes 10 curated playable MP4 triplets under `examples/vbench_codec/video_samples/`:

- `baseline_mp4/`: uncompressed baseline videos
- `intra_q8_zstd/`: videos reconstructed from intra latent compression
- `inter_delta_q8_zstd/`: videos reconstructed from inter latent compression

The full 64-sample binary artifact set is still intentionally excluded to keep the repository manageable; the repo contains the scripts, lightweight reports, and a small playable sample pack for inspection.

## Codex Chat Logs

Local Codex conversation artifacts for this project have also been archived under `examples/codex_chat/`:

- `history.jsonl`
- `session_index.jsonl`
- `codex-tui.log`
- full session rollouts under `examples/codex_chat/sessions/2026/03/31/`

Sensitive local authentication files such as Codex auth tokens or SSH credentials are intentionally excluded.
