# Wan2.2 VAE/TAE Decode Experiments

This document indexes the Wan2.2 decode experiments added to this repository and summarizes the main conclusions.

Common setup across these studies:

- inputs are saved, uncompressed `Wan2.2 TI2V-5B` latents
- the reference target is the official full-sequence `Wan2.2_VAE` decode of the same latent
- quality is reported both on raw decoded frames and on MP4-encoded videos

## 1. Fast-path decode benchmark

Primary report:

- [`wan22_decode/wan22_decode_benchmark_report.md`](./wan22_decode/wan22_decode_benchmark_report.md)

Key takeaways:

- `lighttaew2_2` is fast enough for a streaming preview path on the tested clip.
- first-frame latency was about `5.81 ms`
- steady 4-frame decode time was about `7.24 ms`
- steady throughput was about `552.63 fps`
- end-to-end throughput on the full 121-frame clip was about `96.25 fps`
- the official `Wan2.2_VAE` remained much slower, and batch decode latency scaled almost linearly with batch size in the tested setup

Important implication:

- `TAE` is already viable as the low-latency decode path
- `VAE` decode is the main systems bottleneck

## 2. Sparse VAE-anchor hybrid decode

Primary reports:

- [`wan22_hybrid_decode_first10_k8w3/README.md`](./wan22_hybrid_decode_first10_k8w3/README.md)
- [`wan22_hybrid_decode_tradeoff_search/README.md`](./wan22_hybrid_decode_tradeoff_search/README.md)
- early one-sample pilots under [`../hybrid_decode/`](../hybrid_decode/) and [`../hybrid_decode_smoke/`](../hybrid_decode_smoke/)

What was tested:

- decode all latent steps with `lighttaew2_2`
- inject sparse official `VAE` anchor groups at a fixed interval
- use heuristic repair methods such as direct key replacement, low-pass propagation, and tile-affine correction

Main findings:

- the simple `k8_w3_keyreplace` pilot improved quality on all 10 tested clips
- mean gain was only about `+0.2297 dB` on raw frames and `+0.2274 dB` on MP4 PSNR
- mean amortized cost was under the `24 fps` budget, but strict serial playback still needed a small prebuffer
- searching over longer key windows and intervals did not materially change the conclusion

Important implication:

- sparse `VAE` anchors help, but `keyreplace` alone is a weak lever

## 3. Oracle upper bound for sparse repair

Primary report:

- [`wan22_oracle_repair_run1/README.md`](./wan22_oracle_repair_run1/README.md)

What was tested:

- replace short-window anchors with oracle anchors taken from the full `VAE` decode itself
- keep the sparse repair strategy otherwise unchanged

Main findings:

- even with perfect anchors, `oracle_k8_keyreplace` only gained about `+0.42 dB` raw and `+0.39 dB` MP4 on average
- low-pass and tile-affine variants were in the same range

Important implication:

- the low ceiling is structural to sparse heuristic key propagation, not just an anchor-quality problem

## 4. Stronger causal repair and GPU-native implementation

Primary reports:

- [`wan22_hybrid_decode_v2_run1/README.md`](./wan22_hybrid_decode_v2_run1/README.md)
- [`wan22_hybrid_decode_v2_run3/README.md`](./wan22_hybrid_decode_v2_run3/README.md)

What changed:

- moved from simple key replacement to a stronger causal low-frequency and tile-wise repair path
- then moved that repair path closer to GPU-native execution so streamed `TAE` groups, sparse `VAE` keys, and correction state stayed on GPU

Main findings:

- quality gains increased to roughly the `+1 dB` class on the tested clips
- moving correction to GPU dramatically reduced correction cost
- but the front-path repair math was not the real systems bottleneck
- the remaining bottleneck was sparse `VAE` key decode itself, which still could not be hidden on the same GPU

Important implication:

- better repair logic is useful
- but a single-GPU pipeline still cannot depend on expensive sparse `VAE` refreshes if the goal is stable low-buffer streaming

## 5. Server-side relation side-info

Primary report:

- [`wan22_relation_sideinfo_v1_run1/README.md`](./wan22_relation_sideinfo_v1_run1/README.md)

What was tested:

- the server stores the original latents
- the server also computes compact side information that captures the relation between `TAE` decode and `VAE` decode
- the client receives latents plus side information
- the client runs `TAE` only, then applies the side information locally

Tested operating points:

- `tile_affine_f8_4x4_fp16`
- `lowres_residual_f8_q8`
- `lowres_residual_f4_q8`

Main findings:

- very small metadata around `18 KB` only gave about `+0.20 dB`
- medium side info around `3.67 MB`, or about `18.37%` of latent bytes, gave about `+2.00 dB` MP4 gain
- larger side info around `12.77 MB`, or about `63.92%` of latent bytes, gave about `+4.00 dB` MP4 gain

Important implication:

- the earlier weak gains were partly caused by weak repair information
- richer server-side relation side-info can recover several dB without requiring client-side `VAE`
- as side-info becomes stronger, the method starts to resemble a compact residual stream and must be compared against simpler residual-coding baselines

## Current Conclusions

- `TAE` is already fast enough for streaming preview and low-latency playback.
- `Wan2.2_VAE` decode is the dominant decode bottleneck and remains close to linear in batch latency in the tested path.
- sparse `VAE` keyframe replacement has a low ceiling, even with oracle anchors.
- stronger repair logic helps, but it does not remove the fundamental cost of sparse `VAE` key decode on the same GPU.
- the most promising quality lever so far is server-side `TAE -> VAE` relation side-info, especially the `lowres_residual_f8_q8` operating point.

## Recommended Next Steps

- benchmark the relation side-info path under a strict `24 fps` streaming deadline model
- compare side-info against explicit low-resolution residual-video baselines at matched byte budgets
- explore learned causal repair models that can use compact side-info more efficiently than heuristic filters
- evaluate whether asynchronous or heterogeneous `VAE` anchor generation can be hidden off the client critical path
