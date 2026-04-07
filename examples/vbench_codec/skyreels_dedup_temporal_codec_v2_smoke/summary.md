# SkyReels Dedup Temporal Codec V2

## Per-Sample Results

| Sample | Global Pcodec MB | Per-Channel Pcodec MB | Codec V2 MB | Saved vs Global Pcodec MB | Saved vs Per-Channel Pcodec MB |
|---|---:|---:|---:|---:|---:|
| wingsuit_rescue_glacier_pullup | 49.111 | 48.793 | 47.799 | 1.312 | 0.994 |

## Aggregate Totals

| Item | Bytes | MB | Ratio vs payload |
|---|---:|---:|---:|
| payload | 85248000 | 85.248 | 1.000x |
| global pcodec | 49110723 | 49.111 | 1.736x |
| per-channel pcodec sum | 48792913 | 48.793 | 1.747x |
| codec v2 | 47798779 | 47.799 | 1.783x |

## Codec Idea

- Split the latent by channel and choose a mode independently for each channel.
- Candidate modes are `raw_pcodec` and several `temporal_split_*` variants.
- `temporal_split_*` stores the high byte directly, then uses high-byte continuity to split low-byte data into a stable stream and a changed stream.
- Stable low-byte deltas go through `pcodec`; changed positions go through `zstd` or `pcodec`, depending on which is smaller for that channel.
