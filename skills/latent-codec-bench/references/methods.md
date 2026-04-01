# Methods

## zstd latent containers

Implemented in:
- `scripts/wan22_zstd_codec.py`
- `scripts/run_wan22_latent_codec_eval.py`

### Intra
- Quantize each latent independently
- Compress with zstd
- Decode by dequantizing back to approximate float latent

### Inter
- Apply temporal delta across latent frames
- Quantize the deltas
- Compress with zstd
- Decode by cumulative reconstruction over time

## Full codec benchmark

Use `scripts/wan_latent_codec_bench.py` for broader codec sweeps and reconstruction benchmarking.

## Sample evaluation pack

The repo keeps lightweight reports for all 64 Wan samples plus a curated playable sample pack under:
- `examples/vbench_codec/`
- `examples/vbench_codec/video_samples/`
