# SkyReels Dedup Temporal Lossless Compression

## Per-Sample Best Results

| Sample | Payload MB | Best ZSTD MB | Best Pcodec MB | Best New Codec | New Codec MB | Saved vs Pcodec MB | same-hi fraction |
|---|---:|---:|---:|---|---:|---:|---:|
| wingsuit_rescue_glacier_pullup | 85.248 | 49.582 | 49.111 | temporal_split_raw_zstd | 48.220 | 0.891 | 0.6383 |
| neon_hoverbike_chain_reaction | 85.248 | 47.898 | 44.842 | temporal_split_raw_pcodec | 47.421 | -2.579 | 0.4640 |
| avalanche_snowmobile_bridge_escape | 85.248 | 46.611 | 43.150 | temporal_split_raw_zstd | 45.607 | -2.457 | 0.5909 |

## Aggregate Totals

| Item | Bytes | MB | Ratio vs payload |
|---|---:|---:|---:|
| payload | 255744000 | 255.744 | 1.000x |
| best zstd (`bshuffle_u16`) | 144090515 | 144.091 | 1.775x |
| best pcodec (`raw_u16`) | 137103355 | 137.103 | 1.865x |
| best new temporal codec | 141247980 | 141.248 | 1.811x |
| adaptive best of (`pcodec_raw_u16`, `best temporal`) | 136212633 | 136.213 | 1.878x |

## Codec Idea

- Store the `bf16` high byte stream losslessly with `zstd`.
- Reconstruct the high-byte timeline first, then derive where the exponent/sign byte stayed unchanged across time.
- For positions whose high byte stayed unchanged, encode the low byte as modulo-256 temporal deltas and compress that stream with `pcodec`.
- For positions whose high byte changed, store the current low byte directly and compress that smaller changed-only stream with `zstd`.
- The mask is not stored separately; it is recovered from the restored high-byte stream.

## Interpretation

- The pure temporal split codec is not universally better than plain `pcodec`.
- It clearly helps on `wingsuit_rescue_glacier_pullup`, where the high-byte continuity is highest.
- On the more dynamic `neon_hoverbike_chain_reaction` and `avalanche_snowmobile_bridge_escape`, plain `pcodec_raw_u16` remains stronger.
- An adaptive codec that chooses the smaller of `pcodec_raw_u16` and the time-aware codec per latent does beat both the all-`zstd` and all-`pcodec` baselines on the three-latent total.
