# Wan2.2 Hybrid Decode Trade-off Search

This run compares sparse-key `keyreplace` variants on two uncompressed Wan2.2 latent samples:

- `000_subject_consistency_r00_p003_a_person_eating_a_burger`
- `008_temporal_flickering_r00_p003_a_tranquil_tableau_of_alley`

Variants:

- `k8_w3_keyreplace`
- `k12_w4_keyreplace`
- `k12_w6_keyreplace`
- `k16_w8_keyreplace`
- `k16_w12_keyreplace`

## Main Takeaway

Longer VAE windows do improve the anchor groups themselves, but global video quality still only moves by a few tenths of a dB.

Best practical point in this search:

- `k12_w4_keyreplace`
- mean raw gain vs `tae_only`: about `+0.19 dB`
- mean MP4 gain vs `tae_only`: about `+0.17 dB`
- mean estimated amortized cost: about `33.85 ms/frame`
- serial initial buffer in this stricter model: about `6` frames on average

More aggressive windows:

- `k12_w6`, `k16_w8`, `k16_w12` raise anchor quality much more
- but they do **not** translate that into large global PSNR gains
- and they explode the serial buffer requirement (`~41-99` frames in this run)

## Why This Matters

This search shows the current bottleneck is not just “window too short”.

Even when the sparse VAE key groups become much closer to full VAE, pure `keyreplace` still cannot move the whole clip very much, because most frames remain untouched TAE output.

So the next stage should not focus on more `window/key_interval` tuning alone. The stronger lever is a causal repair mechanism that lets sparse VAE information affect the whole GOP.
