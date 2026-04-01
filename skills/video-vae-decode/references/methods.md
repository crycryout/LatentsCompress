# Methods

## Wan streaming decode

Use:
- `scripts/wan_streaming_decode.py`
- `scripts/benchmark_wan_streaming_decode.py`
- `scripts/compare_wan_full_vs_stream_decode.py`

Capabilities:
- stateful time-step streaming decode
- first-chunk latency measurement
- steady-state fps measurement
- full-vs-stream comparison
- optional `torch.compile` benchmarking

## Open-Sora temporal overlap

Use:
- `scripts/compare_opensora_temporal_overlap.py`

This compares:
- full decode
- native temporal tiled decode with different overlap factors

## Hunyuan unofficial temporal chunk decode

Use:
- `scripts/nonofficial_temporal_chunk_decode.py`

This prototype was built because the official Hunyuan 1.5 VAE path supports spatial tiling but locks out temporal tiling. The prototype reuses the official VAE implementation while enabling temporal chunk decode at the instance level.

## LightTAE comparison

Use:
- `scripts/lighttaew2_2_streaming_decode.py`
- `scripts/benchmark_lighttae_streaming_decode.py`
- `scripts/profile_wan22_decode.py`

These scripts are for decode-side benchmarking and profiling against Wan VAE baselines.
