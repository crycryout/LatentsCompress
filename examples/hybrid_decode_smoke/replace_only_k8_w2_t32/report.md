# Wan2.2 Hybrid Decode Report: replace_only_k8_w2_t32

This experiment uses uncompressed saved Wan2.2 latents, the official `Wan2.2_VAE` full decode as the quality baseline, and `lighttaew2_2` as the fast path.

- keyframe_interval (latent steps): `8`
- anchor_window (latent steps): `2`
- repair_mode: `replace_only`
- tile_size: `32`

## Summary

- sample_count: `1`
- mean_raw_frame_psnr_db: `30.8002`
- mean_mp4_psnr_db: `34.7909`
- mean_mp4_ssim: `0.957708`
- mean_hybrid_seq_decode_sec_estimate: `2.5494`
- realtime_seq_possible_all: `True`
- mean_anchor_vs_tae_psnr_db: `27.7873`

## Per-sample

| stem | raw_psnr_db | mp4_psnr_db | mp4_ssim | hybrid_seq_sec | realtime_seq_possible |
| --- | ---: | ---: | ---: | ---: | ---: |
| `000_subject_consistency_r00_p003_a_person_eating_a_burger` | `30.8002` | `34.7909` | `0.957708` | `2.5494` | `True` |
