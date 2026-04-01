# Wan2.2 Oracle Repair Upper Bound

This run is an **upper-bound analysis** for the repair algorithm.

Instead of using short-window VAE anchors, it uses the exact full-VAE key groups as oracle anchors, then asks:

- if key anchors were perfect,
- how much could sparse causal repair improve `TAE-only`?

Samples:

- `000_subject_consistency_r00_p003_a_person_eating_a_burger`
- `008_temporal_flickering_r00_p003_a_tranquil_tableau_of_alley`

Variants:

- `oracle_k8_keyreplace`
- `oracle_k8_lpf`
- `oracle_k8_tile_affine`
- `oracle_k8_combo`
- `oracle_k12_combo`

## Main Takeaway

Even with perfect key anchors, the gains remain modest:

- `oracle_k8_keyreplace`: mean raw gain about `+0.42 dB`, mean MP4 gain about `+0.39 dB`
- `oracle_k8_lpf`: mean raw gain about `+0.39 dB`, mean MP4 gain about `+0.38 dB`
- `oracle_k8_tile_affine`: mean raw gain about `+0.32 dB`, mean MP4 gain about `+0.33 dB`

This is the most important conclusion from the run:

`sparse keyframe replacement / propagation` itself has a low ceiling in the current setting.

In other words, the current path is not failing only because the short-window VAE anchors are weak. Even with perfect anchors, a sparse-key heuristic repair does **not** jump to multi-dB gains, let alone anything near `50 dB`.

## Interpretation

This upper bound strongly suggests:

- simple sparse `keyreplace` cannot be the whole answer
- low-frequency/tile-affine heuristics help a bit, but not by an order of magnitude
- if the target is truly large quality jumps under streaming constraints, the next candidates are:
  - a stronger causal repair model
  - a decoder closer to full VAE fidelity
  - or a different system assumption than “single lightweight TAE + sparse heuristic anchors”
