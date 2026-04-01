# Wan2.2 Hybrid Decode Report

- latent: `/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/000_subject_consistency_r00_p003_a_person_eating_a_burger.pt`
- latent_shape: `[48, 31, 44, 80]`
- anchor_interval_steps: `12`
- anchor_context_steps: `3`
- anchor_steps: `[0, 1, 13, 25, 30]`
- baseline_vae_decode_sec: `49.4820`
- tae_decode_sec: `0.2879`
- total_anchor_decode_sec: `6.9845`
- effective_total_decode_sec: `7.2724`
- effective_total_fps: `16.6382`
- amortized_group_sec_estimate: `0.126006`
- amortized_group_budget_sec: `0.166667`
- amortized_group_realtime_feasible: `True`

## Anchor Fidelity

| latent_step | frames | window_steps | decode_sec | anchor_psnr_db_vs_full_vae | tae_psnr_db_vs_full_vae |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0` | `1` | `1` | `0.5255` | `inf` | `34.1373` |
| `1` | `4` | `2` | `1.6795` | `inf` | `30.4000` |
| `13` | `4` | `3` | `1.3900` | `38.3194` | `32.9668` |
| `25` | `4` | `3` | `1.5997` | `39.0142` | `35.9684` |
| `30` | `4` | `3` | `1.7899` | `37.2617` | `32.8857` |

## Method Metrics

| method | raw_psnr_db | mp4_psnr_db | mp4_ssim | mp4_bytes |
| --- | ---: | ---: | ---: | ---: |
| `tae_only` | `32.0155` | `36.3302` | `0.956670` | `11093573` |
| `hybrid_key_replace` | `32.4587` | `36.7665` | `0.961285` | `10779008` |
| `hybrid_prev_lpf` | `31.3327` | `35.5213` | `0.958037` | `11746703` |
