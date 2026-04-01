# VBench Codec Video Samples

This folder contains 10 curated Wan2.2 TI2V VBench sample triplets from the `wan22_ti2v5b_vbench_16x4_seed42_q8_64` run.

Each sample is included in three playable MP4 variants:
- `baseline_mp4/`: original MP4 exported from the uncompressed latent run
- `intra_q8_zstd/`: reconstructed MP4 from `intra_q8_zstd` latent compression
- `inter_delta_q8_zstd/`: reconstructed MP4 from `inter_delta_q8_zstd` latent compression

An additional reference decode set is also included:
- `lighttaew2_2/`: reconstructed MP4 from the same saved uncompressed Wan2.2 latents decoded with `lighttaew2_2`

These 10 `lighttaew2_2` videos correspond to the same 10 prompt groups behind the 20 compressed sample videos (`10` intra + `10` inter), so they can be compared directly against both compressed variants.

The selected set keeps the repository size manageable while still covering multiple VBench task families.

| Sample | Task family | Baseline | Intra q8 + zstd | Inter delta q8 + zstd |
| --- | --- | --- | --- | --- |
| `000` | `subject_consistency` | `baseline_mp4/000_subject_consistency_r00_p003_a_person_eating_a_burger.mp4` | `intra_q8_zstd/000_subject_consistency_r00_p003_a_person_eating_a_burger.mp4` | `inter_delta_q8_zstd/000_subject_consistency_r00_p003_a_person_eating_a_burger.mp4` |
| `020` | `aesthetic_quality` | `baseline_mp4/020_aesthetic_quality_r00_p028_origami_dancers_in_white_paper_3d_render_on_white_background_studio_shot_dancing.mp4` | `intra_q8_zstd/020_aesthetic_quality_r00_p028_origami_dancers_in_white_paper_3d_render_on_white_background_studio_shot_dancing.mp4` | `inter_delta_q8_zstd/020_aesthetic_quality_r00_p028_origami_dancers_in_white_paper_3d_render_on_white_background_studio_shot_dancing.mp4` |
| `026` | `imaging_quality` | `baseline_mp4/026_imaging_quality_r02_p035_busy_freeway_at_night.mp4` | `intra_q8_zstd/026_imaging_quality_r02_p035_busy_freeway_at_night.mp4` | `inter_delta_q8_zstd/026_imaging_quality_r02_p035_busy_freeway_at_night.mp4` |
| `032` | `multiple_objects` | `baseline_mp4/032_multiple_objects_r00_p011_a_couch_and_a_potted_plant.mp4` | `intra_q8_zstd/032_multiple_objects_r00_p011_a_couch_and_a_potted_plant.mp4` | `inter_delta_q8_zstd/032_multiple_objects_r00_p011_a_couch_and_a_potted_plant.mp4` |
| `038` | `human_action` | `baseline_mp4/038_human_action_r02_p045_a_person_is_sharpening_knives.mp4` | `intra_q8_zstd/038_human_action_r02_p045_a_person_is_sharpening_knives.mp4` | `inter_delta_q8_zstd/038_human_action_r02_p045_a_person_is_sharpening_knives.mp4` |
| `043` | `color` | `baseline_mp4/043_color_r03_p077_a_green_vase.mp4` | `intra_q8_zstd/043_color_r03_p077_a_green_vase.mp4` | `inter_delta_q8_zstd/043_color_r03_p077_a_green_vase.mp4` |
| `047` | `spatial_relationship` | `baseline_mp4/047_spatial_relationship_r03_p068_a_pizza_on_the_top_of_a_donut_front_view.mp4` | `intra_q8_zstd/047_spatial_relationship_r03_p068_a_pizza_on_the_top_of_a_donut_front_view.mp4` | `inter_delta_q8_zstd/047_spatial_relationship_r03_p068_a_pizza_on_the_top_of_a_donut_front_view.mp4` |
| `052` | `temporal_style` | `baseline_mp4/052_temporal_style_r00_p024_a_shark_is_swimming_in_the_ocean_pan_right.mp4` | `intra_q8_zstd/052_temporal_style_r00_p024_a_shark_is_swimming_in_the_ocean_pan_right.mp4` | `inter_delta_q8_zstd/052_temporal_style_r00_p024_a_shark_is_swimming_in_the_ocean_pan_right.mp4` |
| `058` | `appearance_style` | `baseline_mp4/058_appearance_style_r02_p029_a_panda_drinking_coffee_in_a_cafe_in_paris_by_hokusai_in_the_style_of_ukiyo.mp4` | `intra_q8_zstd/058_appearance_style_r02_p029_a_panda_drinking_coffee_in_a_cafe_in_paris_by_hokusai_in_the_style_of_ukiyo.mp4` | `inter_delta_q8_zstd/058_appearance_style_r02_p029_a_panda_drinking_coffee_in_a_cafe_in_paris_by_hokusai_in_the_style_of_ukiyo.mp4` |
| `062` | `overall_consistency` | `baseline_mp4/062_overall_consistency_r02_p029_campfire_at_night_in_a_snowy_forest_with_starry_sky_in_the_background.mp4` | `intra_q8_zstd/062_overall_consistency_r02_p029_campfire_at_night_in_a_snowy_forest_with_starry_sky_in_the_background.mp4` | `inter_delta_q8_zstd/062_overall_consistency_r02_p029_campfire_at_night_in_a_snowy_forest_with_starry_sky_in_the_background.mp4` |

Approximate committed payload:
- `baseline_mp4`: `65.64 MB`
- `intra_q8_zstd`: `65.86 MB`
- `inter_delta_q8_zstd`: `66.20 MB`
- `lighttaew2_2`: `86.73 MB`

TAE quality report files are stored one level above this folder:
- `../lighttaew2_2_report.json`
- `../lighttaew2_2_report.csv`
- `../lighttaew2_2_report.md`
- `../lighttaew2_2_summary.json`

The `lighttaew2_2` summary over the 10 samples is:
- mean official `Wan2.2_VAE` decode time: `16.8667 s`
- mean `lighttaew2_2` decode time: `0.1542 s`
- mean speedup: `110.5952x`
- mean raw frame PSNR vs official decode: `30.7227 dB`
- mean MP4 PSNR vs official decode baseline MP4: `34.7777 dB`
- mean MP4 SSIM vs official decode baseline MP4: `0.959747`

Total sample video payload is about `284.42 MB`.
