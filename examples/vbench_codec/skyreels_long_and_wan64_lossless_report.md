# SkyReels Long Latent and Wan64 Lossless Compression Report

This note summarizes the latest lossless latent-compression checks added for:

- one `SkyReels-V2-DF-14B-720P` long-video latent (`60s`, `720p`, `24fps`)
- the existing `64` saved `Wan2.2` short-video latents (`5s`)

The underlying scripts are:

- `scripts/compare_single_lossless_long_latent_codecs.py`
- `scripts/compare_single_long_latent_xor_codecs.py`
- `scripts/eval_wan64_xor_codecs.py`

## 1. SkyReels 60s Long Latent: Plain Lossless Codecs

Source files:

- `examples/vbench_codec/skyreels_long_lossless/subject_consistency_000_5eaae1c7_report.md`
- `examples/vbench_codec/skyreels_long_lossless/subject_consistency_000_5eaae1c7_report.json`

Test target:

- latent shape: `16 x 365 x 90 x 160`
- latent dtype: `bfloat16`
- original latent `.pt`: `168,193,880` bytes
- latent payload only: `168,192,000` bytes
- generated MP4: `142,925,494` bytes
- raw RGB24 video equivalent: `4,028,313,600` bytes

Lossless payload compression results:

| codec | compressed bytes | compressed MiB | ratio vs payload |
|---|---:|---:|---:|
| `pcodec` | `85,940,706` | `81.959` | `1.957x` |
| `zstd` | `102,685,993` | `97.929` | `1.638x` |
| `openzl` | `105,590,684` | `100.699` | `1.593x` |

Key points:

- `pcodec` was the smallest of the three on this long latent.
- all three paths were verified lossless.
- the uncompressed latent `.pt` was larger than the final MP4, but all three lossless latent codecs produced files smaller than the MP4.

## 2. SkyReels 60s Long Latent: Blockwise `base + XOR residuals`

Source file:

- `examples/vbench_codec/skyreels_long_lossless/skyreels_long_xor_fast.json`

This experiment first reordered the latent into time-major `TCHW`, then applied a reversible block transform:

- frame `0` in a block is stored as the block base
- later frames in the same block are stored as `frame XOR base`

The transformed payload was then compressed with `zstd` and `pcodec`.

Time-major baselines:

- `zstd_raw`: `103,807,925`
- `zstd_bshuffle`: `92,201,568`
- `pcodec_raw`: `86,467,823`

Windowed XOR results:

| window | `zstd_xor` | `zstd_xor_bshuffle` | `pcodec_xor` |
|---:|---:|---:|---:|
| `2` | `105,347,290` | `92,241,841` | `94,376,810` |
| `4` | `107,011,320` | `92,446,017` | `93,270,595` |
| `8` | `109,014,278` | `92,894,186` | `94,405,276` |
| `16` | `110,326,290` | `93,269,237` | `95,774,955` |

Key points:

- `base + XOR residuals` did not beat the best baseline on the long latent.
- for `zstd`, the best result remained plain `byteshuffle` without XOR.
- for `pcodec`, XOR consistently made the file larger.

## 3. Wan64 Short Latents: `Pcodec` DeltaSpec Sweep

Source file:

- `examples/vbench_codec/wan64_xor/pcodec_deltas_wan64.json`

The earlier `pcodec` runs used:

- `ChunkConfig(compression_level=8)`

This was compared against explicit `DeltaSpec` choices on the `64` saved `Wan2.2` short latents.

Totals over all `64` files:

| config | total bytes | ratio |
|---|---:|---:|
| `default` | `1,083,454,515` | `1.23758x` |
| `auto` | `1,083,454,515` | `1.23758x` |
| `no_op` | `1,101,555,757` | `1.21725x` |
| `try_consecutive(1)` | `1,084,356,126` | `1.23656x` |
| `try_consecutive(2)` | `1,128,667,467` | `1.18801x` |
| `try_conv1(1)` | `1,111,845,459` | `1.20598x` |
| `try_lookback()` | `1,118,379,507` | `1.19894x` |

Key points:

- `default` and `auto` were identical on all `64` files.
- explicit `DeltaSpec` tuning did not improve the aggregate result.
- `try_consecutive(1)` helped on some files but was still worse in total than the default path.

## 4. Wan64 Short Latents: Blockwise `base + XOR residuals`

Source file:

- `examples/vbench_codec/wan64_xor/wan64_xor_codecs.json`

This repeated the same reversible `base + XOR residuals` idea on the `64` short latents.

Aggregate baselines:

| baseline | total bytes |
|---|---:|
| `zstd_raw` | `1,234,074,569` |
| `zstd_bshuffle` | `1,079,741,500` |
| `pcodec_raw` | `1,079,997,821` |

Aggregate XOR totals:

| window | `zstd_xor` | `zstd_xor_bshuffle` | `pcodec_xor` |
|---:|---:|---:|---:|
| `2` | `1,223,829,915` | `1,086,937,965` | `1,102,748,856` |
| `4` | `1,222,181,883` | `1,090,507,253` | `1,099,563,940` |
| `8` | `1,226,301,611` | `1,093,110,348` | `1,101,477,818` |
| `16` | `1,233,617,903` | `1,094,130,478` | `1,110,429,226` |

Interpretation:

- `zstd_xor` did improve over plain `zstd_raw`.
- `zstd_xor_bshuffle` did not beat `zstd_bshuffle`.
- `pcodec_xor` did not beat `pcodec_raw`.

The best aggregate deltas versus baseline were:

- `window=4`: `zstd_xor` saved `11,892,686` bytes vs `zstd_raw`
- `window=2`: `zstd_xor_bshuffle` was still worse by `7,196,465` bytes vs `zstd_bshuffle`
- `window=4`: `pcodec_xor` was still worse by `19,566,119` bytes vs `pcodec_raw`

Per-sample counts reinforce the same result:

- `zstd_xor` vs `zstd_raw`: often better
- `zstd_xor_bshuffle` vs `zstd_bshuffle`: usually worse
- `pcodec_xor` vs `pcodec_raw`: usually worse, and never better for `window=2`

## Bottom Line

For the current latent layouts and exact bit-preserving transforms tested here:

- `pcodec` is the strongest plain lossless codec for the tested `SkyReels` long latent.
- `pcodec` default behavior already matched `DeltaSpec.auto()` on the `Wan64` set.
- blockwise `base + XOR residuals` is not a good next step if the baseline already uses `byteshuffle` or `pcodec`.
- if there is still room to improve, the stronger directions are likely:
  - reversible subtraction-style residual transforms
  - better time-major / bitshuffle-style layouts
  - transforms that preserve useful numeric structure for `pcodec` instead of destroying it with XOR
