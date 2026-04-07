# SkyReels Dedup Temporal Codec V2

## Per-Sample Results

| Sample | Global Pcodec MB | Per-Channel Pcodec MB | Codec V2 MB | Saved vs Global Pcodec MB | Saved vs Per-Channel Pcodec MB |
|---|---:|---:|---:|---:|---:|
| wingsuit_rescue_glacier_pullup | 49.111 | 48.793 | 47.799 | 1.312 | 0.994 |
| neon_hoverbike_chain_reaction | 44.842 | 44.808 | 44.811 | 0.032 | -0.003 |
| avalanche_snowmobile_bridge_escape | 43.150 | 42.598 | 42.601 | 0.549 | -0.003 |

## Aggregate Totals

| Item | Bytes | MB | Ratio vs payload |
|---|---:|---:|---:|
| payload | 255744000 | 255.744 | 1.000x |
| global pcodec | 137103355 | 137.103 | 1.865x |
| per-channel pcodec sum | 136199250 | 136.199 | 1.878x |
| codec v2 | 135210750 | 135.211 | 1.891x |

## Codec Idea

- Split the latent by channel and choose a mode independently for each channel.
- Candidate modes are `raw_pcodec` and several `temporal_split_*` variants.
- `temporal_split_*` stores the high byte directly, then uses high-byte continuity to split low-byte data into a stable stream and a changed stream.
- Stable low-byte deltas go through `pcodec`; changed positions go through `zstd` or `pcodec`, depending on which is smaller for that channel.

## Interpretation

- Codec V2 beats the plain global `pcodec` baseline on all three deduplicated long-video latents.
- The biggest gain comes from `wingsuit_rescue_glacier_pullup`, where most channels prefer a temporal split mode.
- On `neon_hoverbike_chain_reaction` and `avalanche_snowmobile_bridge_escape`, the adaptive search falls back to `raw_pcodec` on every channel, which means the remaining gap to the per-channel baseline is almost entirely container metadata overhead.
- So the second-generation result is already positive, and the next obvious optimization target is a thinner container format when many channels choose the same mode.
