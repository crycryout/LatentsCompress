# SkyReels Dedup Temporal Lossless Compression

## Per-Sample Best Results

| Sample | Payload MB | Best ZSTD MB | Best Pcodec MB | Best New Codec | New Codec MB | Saved vs Pcodec MB | same-hi fraction |
|---|---:|---:|---:|---|---:|---:|---:|
| wingsuit_rescue_glacier_pullup | 85.248 | 49.582 | 49.111 | temporal_split_raw_zstd | 48.220 | 0.891 | 0.6383 |

## Aggregate Totals

| Item | Bytes | MB | Ratio vs payload |
|---|---:|---:|---:|
| payload | 85248000 | 85.248 | 1.000x |
| best zstd (`bshuffle_u16`) | 49581555 | 49.582 | 1.719x |
| best pcodec (`raw_u16`) | 49110723 | 49.111 | 1.736x |
| best new temporal codec | 48220001 | 48.220 | 1.768x |

## Codec Idea

- Store the `bf16` high byte stream losslessly with `zstd`.
- Reconstruct the high-byte timeline first, then derive where the exponent/sign byte stayed unchanged across time.
- For positions whose high byte stayed unchanged, encode the low byte as modulo-256 temporal deltas and compress that stream with `pcodec`.
- For positions whose high byte changed, store the current low byte directly and compress that smaller changed-only stream with `zstd`.
- The mask is not stored separately; it is recovered from the restored high-byte stream.
