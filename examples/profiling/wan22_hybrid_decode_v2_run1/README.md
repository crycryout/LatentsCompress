# Wan2.2 Hybrid Decode V2 Prototype

This prototype adds a stronger causal repair stage on top of sparse short-window VAE anchors:

- global low-frequency color/contrast correction
- tile-wise affine correction
- optional adaptive key scheduling

Result on the current reported sample (`burger`):

- `tae_only`: raw `32.02 dB`, MP4 `36.33 dB`
- `fixed_k8_w3_lfcolor_tile`: raw `32.48 dB`, MP4 `36.75 dB`
- `adaptive_w3_lfcolor_tile`: raw `32.96 dB`, MP4 `37.27 dB`

## Main Takeaway

This is the first prototype here that pushes the gain close to `+1 dB`, which is materially larger than the earlier `+0.2 dB` class of results.

But it is **not stream-ready yet**:

- the current implementation reports a huge required initial buffer
- the per-step correction path is still too expensive

The most likely reason is that the correction stage is paying too much host-side / CPU-side tensor work after TAE decode, so this run should be treated as:

- good evidence that stronger causal repair can improve quality
- bad evidence for current system efficiency

It is a quality-oriented prototype, not an accepted streaming solution.
