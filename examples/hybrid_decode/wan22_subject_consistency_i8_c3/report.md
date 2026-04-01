# Wan2.2 Hybrid Decode Report

- latent: `/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/000_subject_consistency_r00_p003_a_person_eating_a_burger.pt`
- latent_shape: `[48, 31, 44, 80]`
- anchor_interval_steps: `8`
- anchor_context_steps: `3`
- anchor_steps: `[0, 1, 9, 17, 25, 30]`
- baseline_vae_decode_sec: `51.9547`
- tae_decode_sec: `0.5240`
- total_anchor_decode_sec: `6.9882`
- effective_total_decode_sec: `7.5122`
- effective_total_fps: `16.1072`
- amortized_group_sec_estimate: `0.163053`
- amortized_group_budget_sec: `0.166667`
- amortized_group_realtime_feasible: `True`

## Anchor Fidelity

| latent_step | frames | window_steps | decode_sec | anchor_psnr_db_vs_full_vae | tae_psnr_db_vs_full_vae |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0` | `1` | `1` | `0.4080` | `inf` | `34.1373` |
| `1` | `4` | `2` | `1.1987` | `inf` | `30.4000` |
| `9` | `4` | `3` | `1.3928` | `37.3568` | `31.0643` |
| `17` | `4` | `3` | `1.2998` | `39.2325` | `33.2203` |
| `25` | `4` | `3` | `1.3898` | `39.0142` | `35.9684` |
| `30` | `4` | `3` | `1.2991` | `37.2617` | `32.8857` |

## Method Metrics

| method | raw_psnr_db | mp4_psnr_db | mp4_ssim | mp4_bytes |
| --- | ---: | ---: | ---: | ---: |
| `tae_only` | `32.0155` | `36.3302` | `0.956670` | `11093573` |
| `hybrid_key_replace` | `32.6152` | `36.8988` | `0.962363` | `10708906` |
| `hybrid_prev_lpf` | `31.3474` | `35.5167` | `0.958425` | `11774803` |
| `hybrid_interp_lpf` | `31.7995` | `35.9625` | `0.960212` | `11547946` |
