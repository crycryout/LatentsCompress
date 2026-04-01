# Wan2.2 Hybrid Decode V2 Run3

This run reuses the stronger `v2` repair algorithm, but moves the streaming repair path closer to a GPU-native implementation:

- `TAE` streamed groups stay on GPU
- sparse VAE key groups stay on GPU
- correction state and correction application stay on GPU
- per-key decode no longer forces the correction math itself through CPU tensors

## What Changed

Compared with `run2`, quality is essentially unchanged, but the correction cost collapsed:

- on `burger`, `adaptive_w3_lfcolor_tile` kept roughly the same quality (`32.96 dB raw`, `37.27 dB MP4`)
- correction total time dropped from about `4.60 s` to about `0.019 s`
- required serial initial buffer dropped from about `242` frames to about `125` frames

The same pattern appears on `bear climbing a tree`:

- quality stayed near `24.41 dB raw`, `27.98 dB MP4`
- correction total dropped from about `4.32 s` to about `0.022 s`
- required serial initial buffer dropped from about `224` frames to about `125` frames

## Main Takeaway

This run isolates the bottleneck:

- the stronger causal repair itself is no longer the main systems problem
- the dominant problem is still sparse `VAE` key decode

The new split metrics show that:

- the optimistic front path buffer (`async_front_*`) is near zero or one frame
- but `async_key_decode_utilization_max` is around `17x`

So even after fixing the front path, the key decode workload is still far too heavy to hide behind the current schedule on the same GPU.

## Implication

The next step should not be “optimize correction math again”.

It should be one of:

- much cheaper key anchors
- a genuinely asynchronous/offloaded key decode path
- or a smaller learned repair model that depends less on expensive VAE key refreshes
