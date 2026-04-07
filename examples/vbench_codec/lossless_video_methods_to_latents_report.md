# Lossless Video Compression Methods and Latent Migration

## Scope

- Targets:
  - [wingsuit latent](/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/wingsuit_rescue_glacier_pullup/full_video_latents_dedup.pt)
  - [neon latent](/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/neon_hoverbike_chain_reaction/full_video_latents_dedup.pt)
  - [avalanche latent](/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/avalanche_snowmobile_bridge_escape/full_video_latents_dedup.pt)
- Existing latent baselines:
  - [temporal codec v1 summary](/root/LatentsCompress/examples/vbench_codec/skyreels_dedup_temporal_lossless/summary.md)
  - [temporal codec v2 summary](/root/LatentsCompress/examples/vbench_codec/skyreels_dedup_temporal_codec_v2/summary.md)
  - [deep temporal stats summary](/root/LatentsCompress/examples/vbench_codec/skyreels_dedup_temporal_stats/summary.md)
  - [motion compensation probe](/root/LatentsCompress/examples/vbench_codec/latent_motion_comp_probe.json)

## 1. Survey: Main Families of Lossless Video Compression

### A. Intra-only predictive codecs

- Representative codecs:
  - FFV1
  - HuffYUV
  - UTVideo
  - Lagarith
- Core idea:
  - Predict each sample from nearby spatial neighbors inside the same frame.
  - Encode only the residual plus context/entropy coding.
- Why it works:
  - Strong local smoothness and edge continuity.
- Why it matters for latents:
  - This maps naturally to spatial prediction inside each latent frame, but it does not directly exploit temporal continuity.

### B. Hybrid inter-frame lossless codecs

- Representative codecs:
  - Lossless H.264/AVC
  - Lossless HEVC / RExt
  - Lossless AV1
- Core idea:
  - Use inter prediction from reference frames.
  - Use motion estimation / motion compensation to find similar regions across time.
  - Skip transform or use reversible transforms, then entropy-code exact residuals.
- Why it works:
  - Adjacent frames often contain the same objects, moved or slightly deformed.
- Why it matters for latents:
  - This is the most relevant family for long-video latents, because latents are also time-correlated.

### C. Conditional replenishment / skip-style methods

- Core idea:
  - Detect regions that changed little or not at all.
  - Reuse previous-frame content and only code changed regions.
- Why it works:
  - Large regions are often temporally stable, especially in low-motion content.
- Why it matters for latents:
  - Our `high-byte stable -> low-byte delta` design is effectively a latent-domain conditional replenishment scheme.

### D. Motion-compensated predictive methods

- Core idea:
  - Instead of comparing the same pixel location across frames, align the previous frame to the current one using motion vectors.
  - Encode motion plus exact residuals.
- Why it works:
  - It converts object motion into smaller residuals.
- Why it matters for latents:
  - It should help if latent channels preserve spatial object correspondence strongly enough.

### E. Wavelet / hierarchical temporal decomposition

- Representative ideas:
  - Temporal lifting
  - Motion-compensated temporal filtering
- Core idea:
  - Separate slow temporal trends from high-frequency temporal innovations.
- Why it matters for latents:
  - Promising in theory, but harder to keep simple and fully reversible than predictive residual coding.

## 2. What the Standards and Papers Say

- FFV1 is a predictive lossless codec centered on reversible prediction and context modeling.
  - Source: RFC 9043, FFV1 specification
  - Link: https://www.rfc-editor.org/rfc/rfc9043.html
- H.264/AVC’s basic principle is to exploit temporal correlation using motion estimation and reference frames.
  - Source: Fujitsu H.264/AVC overview
  - Link: https://www.fujitsu.com/global/documents/about/resources/publications/fstj/archives/vol49-1/paper09.pdf
- AV1 contains full inter-prediction machinery including motion-vector stack construction and temporal scanning, and also supports lossless operation.
  - Source: AV1 Bitstream & Decoding Process Specification
  - Link: https://aomediacodec.github.io/av1-spec/av1-spec.pdf

## 3. Latent-Specific Findings

### 3.1 Literal duplication is rare

- Exact adjacent `uint16` equality is low:
  - `wingsuit`: `0.97%`
  - `neon`: `0.56%`
  - `avalanche`: `0.86%`
- So naive “frame differencing because many values repeat exactly” is not the main opportunity.

### 3.2 Higher-level bf16 state is much more stable

- From [deep temporal stats summary](/root/LatentsCompress/examples/vbench_codec/skyreels_dedup_temporal_stats/summary.md):

| Sample | sign equal | exponent equal | mantissa equal |
|---|---:|---:|---:|
| `wingsuit` | 0.8812 | 0.5170 | 0.0131 |
| `neon` | 0.7833 | 0.3699 | 0.0106 |
| `avalanche` | 0.8689 | 0.4795 | 0.0123 |

- Interpretation:
  - `sign` and `exponent` stay stable far more often than the whole `bf16` word.
  - `mantissa` keeps jittering.
  - So the right latent-domain analogy to video inter coding is not “reuse whole value,” but “reuse higher-level state and only code the fine detail.”

### 3.3 Motion compensation shows some promise, but not obviously enough yet

- From [latent_motion_comp_probe.json](/root/LatentsCompress/examples/vbench_codec/latent_motion_comp_probe.json):

| Sample | mean MAE gain from small shifts | mean non-zero chosen shift fraction |
|---|---:|---:|
| `wingsuit` | 1.77% | 27.24% |
| `neon` | 0.93% | 48.37% |
| `avalanche` | 2.44% | 65.25% |

- Interpretation:
  - Small translational alignment does reduce latent prediction error a bit.
  - The gain is real but modest.
  - Before building a full lossless motion-compensated latent codec, we would need to confirm that the reduced residual is worth the cost of coding motion vectors and a more complex container.

## 4. Migration Attempts and Results

### Attempt A: Plain back-end baselines

- `zstd_raw`
- `zstd_bshuffle_u16`
- `pcodec_raw_global`

These are strong numeric baselines but do not explicitly model temporal structure.

### Attempt B: Temporal split codec V1

- File: [benchmark_skyreels_dedup_temporal_lossless.py](/root/LatentsCompress/scripts/benchmark_skyreels_dedup_temporal_lossless.py)
- Idea:
  - Store high byte directly.
  - Use high-byte continuity to split low byte into “stable” and “changed” streams.
- Result:
  - Helped strongly on `wingsuit`
  - Did not beat `pcodec` uniformly on all three videos

### Attempt C: Adaptive per-sample selection

- Best-of selection between `pcodec_raw_u16` and the V1 temporal codec beat both all-`zstd` and all-`pcodec` on total size.
- This proved the temporal idea was useful, but not yet robust enough as a single fixed codec.

### Attempt D: Channel-adaptive temporal codec V2

- File: [benchmark_skyreels_dedup_temporal_codec_v2.py](/root/LatentsCompress/scripts/benchmark_skyreels_dedup_temporal_codec_v2.py)
- Summary: [codec v2 summary](/root/LatentsCompress/examples/vbench_codec/skyreels_dedup_temporal_codec_v2/summary.md)
- Idea:
  - Treat each channel as its own sub-stream.
  - Per channel, choose the smallest among:
    - `raw_pcodec`
    - `temporal_split_raw_zstd`
    - `temporal_split_raw_pcodec`
    - `temporal_split_delta_zstd`
    - `temporal_split_delta_pcodec`

### Result of V2

| Sample | Global Pcodec MB | Per-Channel Pcodec MB | Codec V2 MB |
|---|---:|---:|---:|
| `wingsuit` | 49.111 | 48.793 | 47.799 |
| `neon` | 44.842 | 44.808 | 44.811 |
| `avalanche` | 43.150 | 42.598 | 42.601 |

Aggregate totals:

| Method | Total MB |
|---|---:|
| global `pcodec` | 137.103 |
| per-channel `pcodec` | 136.199 |
| codec V2 | 135.211 |

- So V2 beats the plain global `pcodec` baseline on all three latents and wins on aggregate.
- Most of the gain comes from `wingsuit`, where many channels prefer temporal split modes.
- On `neon` and `avalanche`, V2 mostly falls back to raw per-channel `pcodec`, which means the remaining gap is mostly container overhead rather than a wrong modeling choice.

## 5. Which Video Methods Migrate Best to Latents?

### Best fit so far

- Conditional replenishment / skip-style reuse
  - Migrated successfully as high-byte continuity gating
- Hybrid predictive coding with adaptive mode selection
  - Migrated successfully as per-channel mode search
- Plane / component split
  - Migrated successfully via `bf16` high/low byte decomposition

### Partial fit

- Motion-compensated prediction
  - There is measurable gain in latent-domain predictive error
  - But the gain is small enough that a full codec needs careful rate accounting before it is likely to win

### Weak fit so far

- Exact-duplicate / block-copy assumptions
  - Too little literal duplication after dedup
  - Latents are continuous-valued and mantissas jitter

## 6. Practical Takeaways

- Treating long-video latents like generic numeric tensors leaves compression on the table.
- Treating them like ordinary video pixels is also too literal.
- The best current abstraction is:
  - latents have video-like temporal structure,
  - but that structure lives more in stable higher-order state than in exact repeated values.

- In practice, the most effective ingredients so far are:
  - split by channel,
  - split `bf16` into more stable and less stable parts,
  - only apply temporal prediction where higher-level state remains stable,
  - keep a safe fallback path like `raw_pcodec`.

## 7. Next Steps

- Reduce V2 container overhead for cases where most channels select the same mode.
- Upgrade from per-channel to grouped `channel-block` adaptation.
- Revisit motion compensation, but only with explicit rate accounting for motion vectors and skip maps.
- Test whether exponent-coded masks plus block-group mode reuse can outperform plain per-channel `pcodec` on high-motion samples without regressing low-motion ones.
