# Latent Compression, Decode, MP4, and Quality Summary

This note consolidates the main latent-compression, decode, MP4-size, and quality-loss results produced in this repo.

Primary source files:

- `examples/vbench_codec/wan64_codec_report.md`
- `examples/vbench_codec/wan64_summary.json`
- `examples/streaming/wan_stream_decode_unified_summary.md`
- `examples/streaming/wan_t2v_a14b_full_vs_stream_compare.json`
- `examples/streaming/wan_ti2v_5b_full_vs_stream_compare_4cases.json`
- `examples/streaming/opensora_temporal_overlap_compare.json`
- `examples/streaming/hunyuan/coastline_720p121f50_accel_flash_temporal_chunked.mp4`
- `examples/streaming/hunyuan/coastline_720p121f50_accel_flash_temporal_chunked_remux.mp4`
- `examples/vbench_generation/ltxvideo_smoketest/ltx_car_704x512_81f16fps_30s_stats.json`

## 0. Wan64 quick table

This table puts the 64 Wan2.2 TI2V-5B videos into one compact view. Precision-loss columns use raw-frame PSNR against the baseline latent decode, so **higher is better**.

| sample | latent_pt_mb | intra_latent_mb | inter_latent_mb | mp4_mb | intra_raw_psnr_db | inter_raw_psnr_db |
|---|---:|---:|---:|---:|---:|---:|
| `000_subject_consistency_r00_p003_a_person_eating_a_burger` | `20.9534` | `4.586` | `4.2159` | `8.1929` | `54.745446` | `46.498933` |
| `001_subject_consistency_r01_p014_a_car_accelerating_to_gain_speed` | `20.9535` | `4.4768` | `4.0381` | `11.2583` | `51.792594` | `40.952894` |
| `002_subject_consistency_r02_p031_a_truck_anchored_in_a_tranquil_bay` | `20.9536` | `4.1801` | `3.5553` | `14.5095` | `44.622505` | `41.031064` |
| `003_subject_consistency_r03_p035_a_boat_sailing_smoothly_on_a_calm_lake` | `20.9537` | `4.3693` | `4.065` | `12.6946` | `49.253695` | `40.557102` |
| `004_background_consistency_r00_p013_bridge` | `20.9531` | `4.3502` | `3.8204` | `21.1767` | `45.231172` | `38.817723` |
| `005_background_consistency_r01_p017_campus` | `20.9531` | `4.4095` | `3.9097` | `18.3521` | `46.392182` | `37.859855` |
| `006_background_consistency_r02_p028_downtown` | `20.9531` | `4.6186` | `4.0505` | `21.6564` | `46.39134` | `38.493377` |
| `007_background_consistency_r03_p069_ski_slope` | `20.9531` | `4.4378` | `4.1571` | `20.2891` | `45.770144` | `35.087575` |
| `008_temporal_flickering_r00_p003_a_tranquil_tableau_of_alley` | `20.9534` | `4.609` | `3.6296` | `15.6651` | `43.575338` | `39.314796` |
| `009_temporal_flickering_r01_p004_a_tranquil_tableau_of_bar` | `20.9534` | `4.5067` | `3.7717` | `18.4166` | `45.367137` | `38.21461` |
| `010_temporal_flickering_r02_p011_a_tranquil_tableau_of_house` | `20.9534` | `4.4731` | `3.4468` | `19.817` | `41.502363` | `37.82547` |
| `011_temporal_flickering_r03_p054_a_tranquil_tableau_of_in_the_heart_of_plaka_the_neoclassical_architecture_of_the` | `20.9542` | `4.6539` | `3.6958` | `16.2356` | `39.691545` | `40.592788` |
| `012_motion_smoothness_r00_p011_a_car_stuck_in_traffic_during_rush_hour` | `20.9536` | `4.5158` | `4.2009` | `14.3742` | `50.146063` | `40.116792` |
| `013_motion_smoothness_r01_p027_a_train_speeding_down_the_tracks` | `20.9535` | `4.5625` | `4.205` | `14.4808` | `48.8182` | `38.585702` |
| `014_motion_smoothness_r02_p029_a_train_accelerating_to_gain_speed` | `20.9535` | `4.4158` | `4.2045` | `13.264` | `48.189849` | `37.413926` |
| `015_motion_smoothness_r03_p064_a_bear_climbing_a_tree` | `20.9532` | `4.6932` | `4.353` | `20.3969` | `48.530903` | `37.686396` |
| `016_dynamic_degree_r00_p003_a_person_eating_a_burger` | `20.9532` | `4.586` | `4.2159` | `8.1929` | `54.745446` | `46.498933` |
| `017_dynamic_degree_r01_p025_a_bus_stuck_in_traffic_during_rush_hour` | `20.9536` | `4.5076` | `4.1714` | `15.7961` | `48.708808` | `38.507059` |
| `018_dynamic_degree_r02_p069_a_giraffe_bending_down_to_drink_water_from_a_river` | `20.9538` | `4.6678` | `4.2712` | `20.5115` | `46.917985` | `39.177318` |
| `019_dynamic_degree_r03_p071_a_giraffe_running_to_join_a_herd_of_its_kind` | `20.9537` | `4.4475` | `3.9263` | `15.2065` | `50.327071` | `41.117357` |
| `020_aesthetic_quality_r00_p028_origami_dancers_in_white_paper_3d_render_on_white_background_studio_shot_dancing` | `20.9542` | `4.3461` | `3.9128` | `7.4505` | `52.317718` | `43.626884` |
| `021_aesthetic_quality_r01_p053_a_jellyfish_floating_through_the_ocean_with_bioluminescent_tentacles` | `20.954` | `4.5808` | `4.1563` | `9.6369` | `57.882447` | `50.151216` |
| `022_aesthetic_quality_r02_p057_a_steam_train_moving_on_a_mountainside` | `20.9536` | `4.4389` | `4.0723` | `19.3017` | `44.590023` | `36.087879` |
| `023_aesthetic_quality_r03_p075_a_happy_fuzzy_panda_playing_guitar_nearby_a_campfire_snow_mountain_in_the_backgr` | `20.9541` | `4.566` | `3.6958` | `10.1852` | `47.018615` | `40.655689` |
| `024_imaging_quality_r00_p000_close_up_of_grapes_on_a_rotating_table` | `20.9536` | `4.6221` | `4.1324` | `8.2745` | `54.159306` | `44.83245` |
| `025_imaging_quality_r01_p020_a_shark_is_swimming_in_the_ocean` | `20.9534` | `4.2955` | `3.8452` | `8.3106` | `57.034181` | `48.939675` |
| `026_imaging_quality_r02_p035_busy_freeway_at_night` | `20.9532` | `4.3804` | `3.95` | `5.2977` | `55.294019` | `48.619834` |
| `027_imaging_quality_r03_p089_hyper_realistic_spaceship_landing_on_mars` | `20.9536` | `4.637` | `4.084` | `14.6645` | `51.009765` | `44.040252` |
| `028_object_class_r00_p019_a_cow` | `20.953` | `4.5025` | `3.9404` | `17.1499` | `46.043908` | `37.67202` |
| `029_object_class_r01_p035_a_baseball_glove` | `20.9531` | `4.6189` | `3.8916` | `8.7811` | `50.36334` | `41.521656` |
| `030_object_class_r02_p043_a_knife` | `20.9531` | `4.6768` | `3.9163` | `11.5613` | `50.981421` | `42.538291` |
| `031_object_class_r03_p054_a_donut` | `20.9531` | `4.666` | `4.0864` | `15.9801` | `49.282057` | `40.247947` |
| `032_multiple_objects_r00_p011_a_couch_and_a_potted_plant` | `20.9533` | `4.7462` | `3.6431` | `7.4389` | `49.634319` | `46.25754` |
| `033_multiple_objects_r01_p013_a_tv_and_a_laptop` | `20.9532` | `4.5795` | `3.4893` | `9.4589` | `44.756019` | `40.851593` |
| `034_multiple_objects_r02_p027_a_teddy_bear_and_a_frisbee` | `20.9533` | `4.4948` | `3.7014` | `13.285` | `45.543203` | `37.855794` |
| `035_multiple_objects_r03_p043_a_car_and_a_motorcycle` | `20.9532` | `4.5436` | `4.0147` | `15.0377` | `48.884868` | `41.414493` |
| `036_human_action_r00_p012_a_person_is_skateboarding` | `20.9532` | `4.5393` | `4.1571` | `11.0403` | `53.437785` | `43.578868` |
| `037_human_action_r01_p044_a_person_is_planting_trees` | `20.9532` | `4.7` | `4.1076` | `20.5667` | `46.169286` | `37.336303` |
| `038_human_action_r02_p045_a_person_is_sharpening_knives` | `20.9533` | `4.5895` | `4.0676` | `7.4881` | `54.797969` | `45.20836` |
| `039_human_action_r03_p048_a_person_is_hula_hooping` | `20.9532` | `4.616` | `4.1683` | `15.9243` | `51.035007` | `40.742666` |
| `040_color_r00_p005_a_purple_bicycle` | `20.9531` | `4.661` | `4.0514` | `15.8151` | `48.237636` | `39.309906` |
| `041_color_r01_p033_a_blue_umbrella` | `20.9531` | `4.6206` | `3.8592` | `14.2675` | `45.29574` | `37.307713` |
| `042_color_r02_p058_a_red_chair` | `20.953` | `4.7602` | `3.5351` | `6.7318` | `50.675337` | `44.468963` |
| `043_color_r03_p077_a_green_vase` | `20.953` | `4.8245` | `3.6808` | `4.7503` | `54.390409` | `50.707787` |
| `044_spatial_relationship_r00_p010_a_bird_on_the_left_of_a_cat_front_view` | `20.9537` | `4.6709` | `3.4765` | `8.5266` | `48.271618` | `39.745333` |
| `045_spatial_relationship_r01_p015_a_cow_on_the_right_of_an_elephant_front_view` | `20.9538` | `4.6415` | `3.898` | `12.0164` | `48.870708` | `40.661296` |
| `046_spatial_relationship_r02_p048_a_train_on_the_right_of_a_boat_front_view` | `20.9537` | `4.4824` | `4.1409` | `12.7993` | `47.701582` | `40.737295` |
| `047_spatial_relationship_r03_p068_a_pizza_on_the_top_of_a_donut_front_view` | `20.9537` | `4.7353` | `3.5087` | `5.0242` | `53.845232` | `47.529948` |
| `048_scene_r00_p037_golf_course` | `20.953` | `4.2147` | `3.6682` | `13.3739` | `47.182873` | `42.00202` |
| `049_scene_r01_p070_sky` | `20.953` | `4.1681` | `3.8494` | `25.0092` | `44.777076` | `37.148761` |
| `050_scene_r02_p079_train_railway` | `20.953` | `4.4974` | `4.0592` | `20.0937` | `46.895766` | `36.962137` |
| `051_scene_r03_p080_train_station_platform` | `20.9531` | `4.4523` | `3.8262` | `15.6896` | `44.697709` | `39.153095` |
| `052_temporal_style_r00_p024_a_shark_is_swimming_in_the_ocean_pan_right` | `20.9536` | `4.352` | `4.1018` | `7.8704` | `56.708861` | `48.291774` |
| `053_temporal_style_r01_p046_a_cute_happy_corgi_playing_in_park_sunset_tilt_down` | `20.9538` | `4.6281` | `4.0571` | `12.8135` | `50.911159` | `40.946233` |
| `054_temporal_style_r02_p073_a_couple_in_formal_evening_wear_going_home_get_caught_in_a_heavy_downpour_with_u` | `20.9541` | `4.6294` | `4.198` | `9.2594` | `54.13599` | `45.453791` |
| `055_temporal_style_r03_p090_snow_rocky_mountains_peaks_canyon_snow_blanketed_rocky_mountains_surround_and_sh` | `20.9542` | `4.6696` | `3.4444` | `14.214` | `39.35475` | `40.786059` |
| `056_appearance_style_r00_p005_a_beautiful_coastal_beach_in_spring_waves_lapping_on_sand_in_cyberpunk_style` | `20.9541` | `4.5858` | `4.0618` | `14.5865` | `50.448831` | `42.576134` |
| `057_appearance_style_r01_p008_a_beautiful_coastal_beach_in_spring_waves_lapping_on_sand_surrealism_style` | `20.9541` | `4.5342` | `4.1277` | `8.8957` | `53.952535` | `47.783368` |
| `058_appearance_style_r02_p029_a_panda_drinking_coffee_in_a_cafe_in_paris_by_hokusai_in_the_style_of_ukiyo` | `20.9541` | `4.6482` | `3.1942` | `8.5062` | `45.564344` | `38.263486` |
| `059_appearance_style_r03_p084_snow_rocky_mountains_peaks_canyon_snow_blanketed_rocky_mountains_surround_and_sh` | `20.9542` | `4.7004` | `3.547` | `15.676` | `38.871831` | `39.826946` |
| `060_overall_consistency_r00_p010_fireworks` | `20.9531` | `4.7515` | `4.4287` | `21.2748` | `45.17589` | `34.383125` |
| `061_overall_consistency_r01_p012_flying_through_fantasy_landscapes` | `20.9535` | `4.4817` | `4.174` | `23.8543` | `46.908143` | `36.408605` |
| `062_overall_consistency_r02_p029_campfire_at_night_in_a_snowy_forest_with_starry_sky_in_the_background` | `20.954` | `4.6155` | `3.4197` | `6.8089` | `52.877689` | `43.406198` |
| `063_overall_consistency_r03_p037_an_astronaut_is_riding_a_horse_in_the_space_in_a_photorealistic_style` | `20.954` | `4.7427` | `4.2807` | `10.2134` | `48.429202` | `39.67515` |

## 1. Wan2.2 TI2V-5B: 64-video latent codec benchmark

Dataset:

- 64 videos
- prompt set: VBench 16 dimensions x 4 prompts
- model family: `Wan2.2-TI2V-5B`

Mean baseline sizes:

| Metric | Mean |
|---|---:|
| Raw decoded frames | `327.1066 MB` |
| Saved latent `.pt` | `20.953484 MB` |
| Baseline MP4 | `13.521741 MB` |

Codec results:

| Codec | Mean latent container | Mean raw-frame PSNR | Mean MP4 PSNR | Mean MP4 SSIM |
|---|---:|---:|---:|---:|
| `intra_q8_zstd` | `4.551173 MB` | `48.830249 dB` | `46.480539 dB` | `0.993296` |
| `inter_delta_q8_zstd` | `3.930084 MB` | `41.251003 dB` | `42.876847 dB` | `0.989603` |

Takeaways:

- `intra_q8_zstd` is the safer default.
- `inter_delta_q8_zstd` is smaller, but quality drops more clearly.
- The latent containers are substantially smaller than both the raw latent `.pt` and the reconstructed raw-frame tensor size.

Compression readout:

- `intra_q8_zstd`: about `20.95 / 4.55 ~= 4.6x` latent-size reduction
- `inter_delta_q8_zstd`: about `20.95 / 3.93 ~= 5.3x` latent-size reduction

Interpretation:

- `intra_q8_zstd` is the better "production-like" latent archival choice when quality stability matters.
- `inter_delta_q8_zstd` is the better "push size harder" option when sequential decode is acceptable.

## 2. Wan VAE decode: full decode vs streaming decode

The repo contains two kinds of decode comparison:

1. wall-clock timing: full decode vs streaming decode
2. quality difference: full decode vs streaming decode

### 2.1 Timing summary

From `examples/streaming/wan_stream_decode_unified_summary.md`.

#### Wan A14B

| Mode | Full decode | Stream decode | Stream slower than full |
|---|---:|---:|---:|
| eager, no cache | `6.0999 s` | `6.5216 s` | `6.91%` |
| eager, conv2 cache | `6.0140 s` | `6.5143 s` | `8.32%` |
| compile + warmup, no cache | `4.9511 s` | `6.4453 s` | `30.18%` |
| compile + warmup, conv2 cache | `4.9488 s` | `6.4743 s` | `30.83%` |

#### Wan TI2V-5B

| Mode | Full decode | Stream decode | Stream slower than full |
|---|---:|---:|---:|
| eager, no cache (4-case mean) | `17.3541 s` | `18.3333 s` | `5.65%` |
| compile + warmup, no cache (1 case) | `15.7413 s` | `18.4519 s` | `17.22%` |
| compile + warmup, conv2 cache (1 case) | `15.7700 s` | `18.5473 s` | `17.61%` |

Timing conclusion:

- In eager mode, streaming decode is only modestly slower than full decode.
- In compile mode, full decode speeds up more than stream decode, so the percentage gap grows.
- The `conv2` cache itself adds very little extra runtime; the main overhead comes from the streaming orchestration pattern.

### 2.2 Quality summary

#### Wan A14B

From `examples/streaming/wan_t2v_a14b_full_vs_stream_compare.json`.

| Metric | Value |
|---|---:|
| Full decode time | `6.0999 s` |
| Stream decode time | `6.5216 s` |
| Stream slower than full | `6.91%` |
| Float max abs diff | `0.2093745` |
| Uint8 MSE | `0.2058908` |
| Uint8 PSNR | `54.9944 dB` |
| Exact equality | `false` |

#### Wan TI2V-5B

From `examples/streaming/wan_ti2v_5b_full_vs_stream_compare_4cases.json`.

4-case mean:

| Metric | Value |
|---|---:|
| Mean full decode time | `17.3541 s` |
| Mean stream decode time | `18.3333 s` |
| Mean stream slower than full | `5.65%` |
| Mean uint8 MSE | `0.2906456` |
| Mean uint8 PSNR | `53.6922 dB` |
| Max float max abs | `0.6378441` |
| Exact equality | `false` |

Quality conclusion:

- Streaming decode is not numerically identical to full decode.
- The quality gap exists, but it is still small enough to stay in the "close reconstruction" regime for the tested cases.
- For Wan, the main trade-off is therefore:
  - full decode = best numeric fidelity
  - streaming decode = similar output with modest timing overhead and small quality loss

## 3. Open-Sora temporal overlap decode

From `examples/streaming/opensora_temporal_overlap_compare.json`.

Tested:

- native temporal tiled decode
- overlap `0.0`
- overlap `0.25`
- compared against full decode

Result:

| Case | overlap 0.0 vs full | overlap 0.25 vs full |
|---|---|---|
| original latent | exact match | exact match |
| synthetic extended latent | exact match | exact match |

Measured summary:

- `mean_psnr_overlap_0.0 = 100.0`
- `mean_psnr_overlap_0.25 = 100.0`
- both `overlap_0_equals_full_all_cases` and `overlap_0.25_equals_full_all_cases` are `true`

Conclusion:

- In the tested Open-Sora native path, the built-in temporal tiled decode matched full decode exactly.
- The tested `25%` overlap did not improve quality, because the no-overlap path was already exact on these cases.

## 4. Hunyuan official decode vs unofficial temporal chunk decode

Relevant artifacts:

- output video: `examples/streaming/hunyuan/coastline_720p121f50_accel_flash_temporal_chunked.mp4`
- remuxed output: `examples/streaming/hunyuan/coastline_720p121f50_accel_flash_temporal_chunked_remux.mp4`

What happened:

- Official Hunyuan GPU VAE decode for a saved `121`-frame latent OOMed even after using the official low-memory decode path.
- The core limitation is that the official VAE supports spatial tiling, but not temporal tiling.
- A nonofficial temporal-chunk decode prototype reused the official VAE logic while unlocking temporal chunking at instance level.

Outcome:

- Official whole-sequence GPU decode: OOM
- Unofficial temporal chunk GPU decode: success
- Produced a valid MP4 and a remuxed compatibility version

Interpretation:

- This is a decode-architecture limitation, not evidence that the latent itself is invalid.
- For long Hunyuan clips, temporal chunking is the decisive missing piece for GPU decode viability.

## 5. LTX-Video 0.9.8 saved-latent generation footprint

From `examples/vbench_generation/ltxvideo_smoketest/ltx_car_704x512_81f16fps_30s_stats.json`.

Configuration:

- model: `LTX-Video-0.9.8-13B-distilled`
- resolution: `704x512`
- frames: `81`
- fps: `16`
- steps: `30`

Recorded sizes:

| Metric | Value |
|---|---:|
| Raw decoded video tensor | `334.125 MiB` |
| MP4 | `0.7409 MiB` |
| Saved latent raw bytes | `1.890625 MiB` |
| Saved latent `.pt` | `1.8929 MiB` |
| Latent shape | `[1, 3872, 128]` |
| Latent stage | `pre_vae_decode_final` |

Interpretation:

- LTX's heavily compressed latent space produces very small pre-VAE latent payloads.
- The gap between raw decoded tensor size and saved latent size is much larger than in the Wan cases.
- This is consistent with LTX's more aggressive latent compression design.

## 6. Practical conclusions

### Latent compression

- `Wan intra_q8_zstd` is the strongest default choice when quality matters.
- `Wan inter_delta_q8_zstd` is best when container size matters more than reconstruction fidelity.

### Decode

- Wan streaming decode is viable and close to full decode, but not numerically exact.
- Open-Sora native temporal tiled decode can be exact.
- Hunyuan official GPU decode for long clips is blocked mainly by lack of temporal tiling.

### MP4 output

- The repo contains both raw outputs and a remuxed Hunyuan MP4 for compatibility.
- When a file looks "damaged" in a picky player, remuxing the container can fix playback without re-encoding.

### Precision loss

- Wan latent compression shows the clearest quantitative quality story:
  - `intra_q8_zstd`: strong size reduction with small quality loss
  - `inter_delta_q8_zstd`: smaller files with noticeably larger quality loss
- Wan streaming decode quality loss is real but still limited.
- Open-Sora native overlap tests showed no measurable loss relative to full decode for the tested cases.

## Appendix A. Wan2.2 64-video full per-sample metrics

This appendix rewrites the 64-video Wan2.2 TI2V-5B metric sheet into a human-readable Markdown layout. All original CSV fields are preserved, but grouped by sample into smaller tables.


### 000_subject_consistency_r00_p003_a_person_eating_a_burger

Prompt: `a person eating a burger`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953351` |
| latent_pt_mb | `20.9534` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `8192914` |
| mp4_mb | `8.1929` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4585964` | `4.586` | `4.569` | `4.5685` | `9.546e-05` | `0.00821351` | `0.02800232` | `0.2180398` | `54.745446` | `49.252154` | `0.992508` |
| inter_delta_q8_zstd | `4215926` | `4.2159` | `4.97` | `4.9695` | `0.00183027` | `0.03173181` | `0.34295741` | `1.45608604` | `46.498933` | `47.194402` | `0.990952` |

### 001_subject_consistency_r01_p014_a_car_accelerating_to_gain_speed

Prompt: `a car accelerating to gain speed`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953471` |
| latent_pt_mb | `20.9535` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `11258327` |
| mp4_mb | `11.2583` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4476776` | `4.4768` | `4.6805` | `4.6799` | `0.00010135` | `0.00853383` | `0.02401172` | `0.43034914` | `51.792594` | `47.59188` | `0.992278` |
| inter_delta_q8_zstd | `4038062` | `4.0381` | `5.189` | `5.1884` | `0.00195619` | `0.03258847` | `0.40670514` | `5.22144747` | `40.952894` | `43.041363` | `0.988603` |

### 002_subject_consistency_r02_p031_a_truck_anchored_in_a_tranquil_bay

Prompt: `a truck anchored in a tranquil bay`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953613` |
| latent_pt_mb | `20.9536` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `14509476` |
| mp4_mb | `14.5095` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4180129` | `4.1801` | `5.0127` | `5.0121` | `8.269e-05` | `0.00772484` | `0.0243163` | `2.24300313` | `44.622505` | `44.932702` | `0.993528` |
| inter_delta_q8_zstd | `3555260` | `3.5553` | `5.8937` | `5.893` | `0.00058832` | `0.01764218` | `0.23694962` | `5.12830639` | `41.031064` | `42.579594` | `0.990401` |

### 003_subject_consistency_r03_p035_a_boat_sailing_smoothly_on_a_calm_lake

Prompt: `a boat sailing smoothly on a calm lake`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953705` |
| latent_pt_mb | `20.9537` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `12694611` |
| mp4_mb | `12.6946` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4369252` | `4.3693` | `4.7957` | `4.7951` | `7.74e-05` | `0.00747193` | `0.022708` | `0.77216637` | `49.253695` | `47.208497` | `0.993499` |
| inter_delta_q8_zstd | `4064997` | `4.065` | `5.1547` | `5.154` | `0.00109634` | `0.02407717` | `0.31945759` | `5.71965933` | `40.557102` | `42.49111` | `0.98879` |

### 004_background_consistency_r00_p013_bridge

Prompt: `bridge`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953118` |
| latent_pt_mb | `20.9531` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `21176675` |
| mp4_mb | `21.1767` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4350170` | `4.3502` | `4.8166` | `4.8161` | `0.00010157` | `0.00852683` | `0.0259881` | `1.94967914` | `45.231172` | `44.143648` | `0.99202` |
| inter_delta_q8_zstd | `3820387` | `3.8204` | `5.4846` | `5.484` | `0.00120613` | `0.02503079` | `0.39512539` | `8.53705502` | `38.817723` | `40.237529` | `0.986149` |

### 005_background_consistency_r01_p017_campus

Prompt: `campus`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953118` |
| latent_pt_mb | `20.9531` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `18352086` |
| mp4_mb | `18.3521` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4409543` | `4.4095` | `4.7518` | `4.7513` | `0.00012082` | `0.00937058` | `0.02653265` | `1.49232078` | `46.392182` | `44.824303` | `0.9935` |
| inter_delta_q8_zstd | `3909746` | `3.9097` | `5.3592` | `5.3587` | `0.00165904` | `0.03012374` | `0.35589552` | `10.64375591` | `37.859855` | `39.848414` | `0.987504` |

### 006_background_consistency_r02_p028_downtown

Prompt: `downtown`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953132` |
| latent_pt_mb | `20.9531` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `21656445` |
| mp4_mb | `21.6564` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4618610` | `4.6186` | `4.5367` | `4.5362` | `8.068e-05` | `0.00767919` | `0.02368331` | `1.4926101` | `46.39134` | `43.802063` | `0.992497` |
| inter_delta_q8_zstd | `4050459` | `4.0505` | `5.173` | `5.1725` | `0.00108659` | `0.02410062` | `0.26632369` | `9.19904423` | `38.493377` | `39.996878` | `0.988454` |

### 007_background_consistency_r03_p069_ski_slope

Prompt: `ski slope`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953139` |
| latent_pt_mb | `20.9531` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `20289116` |
| mp4_mb | `20.2891` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4437752` | `4.4378` | `4.7216` | `4.7211` | `0.00011675` | `0.00916466` | `0.02580845` | `1.72213018` | `45.770144` | `44.533079` | `0.991492` |
| inter_delta_q8_zstd | `4157130` | `4.1571` | `5.0403` | `5.0398` | `0.00180275` | `0.0312928` | `0.43119013` | `20.15221786` | `35.087575` | `37.516201` | `0.98268` |

### 008_temporal_flickering_r00_p003_a_tranquil_tableau_of_alley

Prompt: `A tranquil tableau of alley`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953436` |
| latent_pt_mb | `20.9534` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `15665115` |
| mp4_mb | `15.6651` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4609016` | `4.609` | `4.5462` | `4.5457` | `7.041e-05` | `0.00716328` | `0.02054757` | `2.85460877` | `43.575338` | `44.217448` | `0.99225` |
| inter_delta_q8_zstd | `3629565` | `3.6296` | `5.773` | `5.7723` | `0.00111293` | `0.02437396` | `0.30334204` | `7.61378813` | `39.314796` | `41.37285` | `0.988473` |

### 009_temporal_flickering_r01_p004_a_tranquil_tableau_of_bar

Prompt: `A tranquil tableau of bar`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953358` |
| latent_pt_mb | `20.9534` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `18416615` |
| mp4_mb | `18.4166` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4506712` | `4.5067` | `4.6494` | `4.6489` | `9.684e-05` | `0.00834126` | `0.02493543` | `1.88958585` | `45.367137` | `44.676865` | `0.992925` |
| inter_delta_q8_zstd | `3771650` | `3.7717` | `5.5555` | `5.5549` | `0.00164354` | `0.02969682` | `0.36149395` | `9.8088789` | `38.21461` | `40.255855` | `0.986618` |

### 010_temporal_flickering_r02_p011_a_tranquil_tableau_of_house

Prompt: `A tranquil tableau of house`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953436` |
| latent_pt_mb | `20.9534` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `19817004` |
| mp4_mb | `19.817` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4473136` | `4.4731` | `4.6843` | `4.6837` | `8.223e-05` | `0.00772952` | `0.02070308` | `4.60091305` | `41.502363` | `42.769758` | `0.991019` |
| inter_delta_q8_zstd | `3446807` | `3.4468` | `6.0791` | `6.0784` | `0.00118851` | `0.02502793` | `0.28428453` | `10.72836018` | `37.82547` | `39.970926` | `0.98652` |

### 011_temporal_flickering_r03_p054_a_tranquil_tableau_of_in_the_heart_of_plaka_the_neoclassical_architecture_of_the

Prompt: `A tranquil tableau of in the heart of Plaka, the neoclassical architecture of the old city harmonizes with the ancient ruins`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20954191` |
| latent_pt_mb | `20.9542` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `16235593` |
| mp4_mb | `16.2356` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4653921` | `4.6539` | `4.5025` | `4.5018` | `8.152e-05` | `0.00764673` | `0.02280593` | `6.98113203` | `39.691545` | `41.600567` | `0.993108` |
| inter_delta_q8_zstd | `3695815` | `3.6958` | `5.6697` | `5.6689` | `0.00045435` | `0.01573036` | `0.16477394` | `5.67285299` | `40.592788` | `42.348096` | `0.993447` |

### 012_motion_smoothness_r00_p011_a_car_stuck_in_traffic_during_rush_hour

Prompt: `a car stuck in traffic during rush hour`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953634` |
| latent_pt_mb | `20.9536` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `14374153` |
| mp4_mb | `14.3742` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4515764` | `4.5158` | `4.6401` | `4.6395` | `8.496e-05` | `0.00783127` | `0.02090907` | `0.62874436` | `50.146063` | `47.218206` | `0.992917` |
| inter_delta_q8_zstd | `4200919` | `4.2009` | `4.9879` | `4.9873` | `0.00150188` | `0.02860964` | `0.40416929` | `6.32996321` | `40.116792` | `42.245706` | `0.988098` |

### 013_motion_smoothness_r01_p027_a_train_speeding_down_the_tracks

Prompt: `a train speeding down the tracks`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953457` |
| latent_pt_mb | `20.9535` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `14480769` |
| mp4_mb | `14.4808` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4562492` | `4.5625` | `4.5925` | `4.592` | `0.00012439` | `0.00943477` | `0.02740493` | `0.85361177` | `48.8182` | `46.272656` | `0.99256` |
| inter_delta_q8_zstd | `4204960` | `4.205` | `4.983` | `4.9825` | `0.00226594` | `0.03494143` | `0.45261145` | `9.00554752` | `38.585702` | `40.752395` | `0.985785` |

### 014_motion_smoothness_r02_p029_a_train_accelerating_to_gain_speed

Prompt: `a train accelerating to gain speed`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953471` |
| latent_pt_mb | `20.9535` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `13264039` |
| mp4_mb | `13.264` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4415773` | `4.4158` | `4.7451` | `4.7446` | `0.00012673` | `0.00953108` | `0.02899742` | `0.98649639` | `48.189849` | `45.755096` | `0.99327` |
| inter_delta_q8_zstd | `4204503` | `4.2045` | `4.9836` | `4.983` | `0.00252834` | `0.03647766` | `0.50375831` | `11.79472446` | `37.413926` | `39.683717` | `0.987021` |

### 015_motion_smoothness_r03_p064_a_bear_climbing_a_tree

Prompt: `a bear climbing a tree`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953195` |
| latent_pt_mb | `20.9532` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `20396897` |
| mp4_mb | `20.3969` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4693226` | `4.6932` | `4.4646` | `4.4641` | `0.00010409` | `0.00861783` | `0.02578002` | `0.91199005` | `48.530903` | `44.994487` | `0.994206` |
| inter_delta_q8_zstd | `4353008` | `4.353` | `4.8135` | `4.813` | `0.00186437` | `0.03174552` | `0.36952376` | `11.07747459` | `37.686396` | `39.928404` | `0.990403` |

### 016_dynamic_degree_r00_p003_a_person_eating_a_burger

Prompt: `a person eating a burger`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953188` |
| latent_pt_mb | `20.9532` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `8192914` |
| mp4_mb | `8.1929` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4585964` | `4.586` | `4.569` | `4.5685` | `9.546e-05` | `0.00821351` | `0.02800232` | `0.2180398` | `54.745446` | `49.252154` | `0.992508` |
| inter_delta_q8_zstd | `4215926` | `4.2159` | `4.97` | `4.9695` | `0.00183027` | `0.03173181` | `0.34295741` | `1.45608604` | `46.498933` | `47.194402` | `0.990952` |

### 017_dynamic_degree_r01_p025_a_bus_stuck_in_traffic_during_rush_hour

Prompt: `a bus stuck in traffic during rush hour`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953613` |
| latent_pt_mb | `20.9536` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `15796116` |
| mp4_mb | `15.7961` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4507641` | `4.5076` | `4.6485` | `4.6479` | `9.497e-05` | `0.00828309` | `0.02284145` | `0.875386` | `48.708808` | `46.410341` | `0.992794` |
| inter_delta_q8_zstd | `4171360` | `4.1714` | `5.0232` | `5.0226` | `0.00146646` | `0.02833233` | `0.33882678` | `9.1701088` | `38.507059` | `40.861671` | `0.988068` |

### 018_dynamic_degree_r02_p069_a_giraffe_bending_down_to_drink_water_from_a_river

Prompt: `a giraffe bending down to drink water from a river`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953754` |
| latent_pt_mb | `20.9538` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `20511469` |
| mp4_mb | `20.5115` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4667778` | `4.6678` | `4.489` | `4.4884` | `7.928e-05` | `0.00751688` | `0.02231914` | `1.32215345` | `46.917985` | `44.765369` | `0.992479` |
| inter_delta_q8_zstd | `4271168` | `4.2712` | `4.9059` | `4.9052` | `0.00114274` | `0.02405726` | `0.35928935` | `7.8586607` | `39.177318` | `40.963073` | `0.987558` |

### 019_dynamic_degree_r03_p071_a_giraffe_running_to_join_a_herd_of_its_kind

Prompt: `a giraffe running to join a herd of its kind`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953712` |
| latent_pt_mb | `20.9537` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `15206538` |
| mp4_mb | `15.2065` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4447499` | `4.4475` | `4.7113` | `4.7107` | `0.000104` | `0.00865349` | `0.02533007` | `0.60307765` | `50.327071` | `46.669001` | `0.991362` |
| inter_delta_q8_zstd | `3926309` | `3.9263` | `5.3367` | `5.3361` | `0.00163568` | `0.02968769` | `0.38226557` | `5.02741432` | `41.117357` | `43.068389` | `0.985962` |

### 020_aesthetic_quality_r00_p028_origami_dancers_in_white_paper_3d_render_on_white_background_studio_shot_dancing

Prompt: `Origami dancers in white paper, 3D render, on white background, studio shot, dancing modern dance.`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20954177` |
| latent_pt_mb | `20.9542` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `7450477` |
| mp4_mb | `7.4505` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4346106` | `4.3461` | `4.8214` | `4.8206` | `0.00010913` | `0.008791` | `0.0287028` | `0.38133666` | `52.317718` | `49.331254` | `0.995434` |
| inter_delta_q8_zstd | `3912798` | `3.9128` | `5.3553` | `5.3545` | `0.00172617` | `0.02998482` | `0.44137415` | `2.82092786` | `43.626884` | `45.347437` | `0.994359` |

### 021_aesthetic_quality_r01_p053_a_jellyfish_floating_through_the_ocean_with_bioluminescent_tentacles

Prompt: `A jellyfish floating through the ocean, with bioluminescent tentacles`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20954029` |
| latent_pt_mb | `20.954` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `9636924` |
| mp4_mb | `9.6369` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4580762` | `4.5808` | `4.5744` | `4.5737` | `7.525e-05` | `0.00739111` | `0.01970309` | `0.10588529` | `57.882447` | `52.164032` | `0.99589` |
| inter_delta_q8_zstd | `4156286` | `4.1563` | `5.0415` | `5.0408` | `0.00108554` | `0.02406751` | `0.30855513` | `0.62799871` | `50.151216` | `50.166786` | `0.994948` |

### 022_aesthetic_quality_r02_p057_a_steam_train_moving_on_a_mountainside

Prompt: `A steam train moving on a mountainside`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953627` |
| latent_pt_mb | `20.9536` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `19301682` |
| mp4_mb | `19.3017` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4438881` | `4.4389` | `4.7205` | `4.7199` | `0.00010336` | `0.00867656` | `0.02333081` | `2.25984216` | `44.590023` | `43.93375` | `0.991734` |
| inter_delta_q8_zstd | `4072272` | `4.0723` | `5.1454` | `5.1448` | `0.00149557` | `0.02834064` | `0.33988661` | `16.00635719` | `36.087879` | `38.455146` | `0.985438` |

### 023_aesthetic_quality_r03_p075_a_happy_fuzzy_panda_playing_guitar_nearby_a_campfire_snow_mountain_in_the_backgr

Prompt: `A happy fuzzy panda playing guitar nearby a campfire, snow mountain in the background`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20954113` |
| latent_pt_mb | `20.9541` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `10185202` |
| mp4_mb | `10.1852` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4566006` | `4.566` | `4.5892` | `4.5885` | `0.00011919` | `0.00923408` | `0.02703369` | `1.29187` | `47.018615` | `46.400532` | `0.994224` |
| inter_delta_q8_zstd | `3695759` | `3.6958` | `5.6698` | `5.6689` | `0.00174912` | `0.03047876` | `0.40397799` | `5.59128332` | `40.655689` | `42.638746` | `0.99161` |

### 024_imaging_quality_r00_p000_close_up_of_grapes_on_a_rotating_table

Prompt: `Close up of grapes on a rotating table.`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953613` |
| latent_pt_mb | `20.9536` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `8274470` |
| mp4_mb | `8.2745` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4622076` | `4.6221` | `4.5334` | `4.5328` | `0.00010802` | `0.00875692` | `0.02727389` | `0.2495455` | `54.159306` | `49.291016` | `0.993456` |
| inter_delta_q8_zstd | `4132400` | `4.1324` | `5.0706` | `5.0699` | `0.00205873` | `0.03325092` | `0.41930255` | `2.13715196` | `44.83245` | `46.289878` | `0.991552` |

### 025_imaging_quality_r01_p020_a_shark_is_swimming_in_the_ocean

Prompt: `a shark is swimming in the ocean.`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953443` |
| latent_pt_mb | `20.9534` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `8310618` |
| mp4_mb | `8.3106` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4295472` | `4.2955` | `4.878` | `4.8775` | `9.884e-05` | `0.00837601` | `0.02582359` | `0.1287248` | `57.034181` | `52.239405` | `0.99551` |
| inter_delta_q8_zstd | `3845198` | `3.8452` | `5.4492` | `5.4486` | `0.00172769` | `0.03079179` | `0.33753619` | `0.8300665` | `48.939675` | `49.769023` | `0.994197` |

### 026_imaging_quality_r02_p035_busy_freeway_at_night

Prompt: `Busy freeway at night.`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953174` |
| latent_pt_mb | `20.9532` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `5297729` |
| mp4_mb | `5.2977` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4380444` | `4.3804` | `4.7833` | `4.7829` | `0.0001251` | `0.00951353` | `0.02639616` | `0.19216684` | `55.294019` | `50.719202` | `0.996381` |
| inter_delta_q8_zstd | `3950011` | `3.95` | `5.3046` | `5.304` | `0.00114759` | `0.02473033` | `0.27869987` | `0.89350486` | `48.619834` | `48.878025` | `0.996002` |

### 027_imaging_quality_r03_p089_hyper_realistic_spaceship_landing_on_mars

Prompt: `Hyper-realistic spaceship landing on Mars`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953634` |
| latent_pt_mb | `20.9536` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `14664476` |
| mp4_mb | `14.6645` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4637021` | `4.637` | `4.5188` | `4.5182` | `6.934e-05` | `0.00705322` | `0.02150285` | `0.51535183` | `51.009765` | `47.13031` | `0.990472` |
| inter_delta_q8_zstd | `4083966` | `4.084` | `5.1307` | `5.1301` | `0.00096152` | `0.02257684` | `0.24517155` | `2.5648098` | `44.040252` | `44.694113` | `0.98664` |

### 028_object_class_r00_p019_a_cow

Prompt: `a cow`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953041` |
| latent_pt_mb | `20.953` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `17149933` |
| mp4_mb | `17.1499` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4502508` | `4.5025` | `4.6536` | `4.6532` | `0.00010404` | `0.00860858` | `0.02972424` | `1.61692381` | `46.043908` | `45.074881` | `0.992899` |
| inter_delta_q8_zstd | `3940429` | `3.9404` | `5.3175` | `5.3169` | `0.00173924` | `0.03053899` | `0.37804645` | `11.11420345` | `37.67202` | `40.016165` | `0.98548` |

### 029_object_class_r01_p035_a_baseball_glove

Prompt: `a baseball glove`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953118` |
| latent_pt_mb | `20.9531` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `8781088` |
| mp4_mb | `8.7811` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4618855` | `4.6189` | `4.5364` | `4.536` | `0.00011376` | `0.00901341` | `0.02606976` | `0.59806222` | `50.36334` | `46.863997` | `0.994449` |
| inter_delta_q8_zstd | `3891645` | `3.8916` | `5.3841` | `5.3836` | `0.00200198` | `0.03291089` | `0.4441213` | `4.58051968` | `41.521656` | `43.299268` | `0.992321` |

### 030_object_class_r02_p043_a_knife

Prompt: `a knife`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953055` |
| latent_pt_mb | `20.9531` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `11561307` |
| mp4_mb | `11.5613` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4676826` | `4.6768` | `4.4802` | `4.4798` | `8.228e-05` | `0.00761937` | `0.02388227` | `0.51872635` | `50.981421` | `46.747931` | `0.994227` |
| inter_delta_q8_zstd | `3916338` | `3.9163` | `5.3502` | `5.3497` | `0.00136116` | `0.02681568` | `0.30959153` | `3.6245265` | `42.538291` | `43.643022` | `0.992144` |

### 031_object_class_r03_p054_a_donut

Prompt: `a donut`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953055` |
| latent_pt_mb | `20.9531` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `15980100` |
| mp4_mb | `15.9801` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4665990` | `4.666` | `4.4906` | `4.4902` | `7.81e-05` | `0.00752681` | `0.02047816` | `0.76714003` | `49.282057` | `46.12901` | `0.993674` |
| inter_delta_q8_zstd | `4086350` | `4.0864` | `5.1276` | `5.1271` | `0.00136572` | `0.02709918` | `0.33573398` | `6.14165831` | `40.247947` | `41.876891` | `0.989045` |

### 032_multiple_objects_r00_p011_a_couch_and_a_potted_plant

Prompt: `a couch and a potted plant`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953344` |
| latent_pt_mb | `20.9533` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `7438851` |
| mp4_mb | `7.4389` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4746153` | `4.7462` | `4.4148` | `4.4143` | `8.764e-05` | `0.00791647` | `0.02366185` | `0.70737296` | `49.634319` | `48.024687` | `0.993926` |
| inter_delta_q8_zstd | `3643135` | `3.6431` | `5.7515` | `5.7508` | `0.00042999` | `0.01530791` | `0.17825663` | `1.53931081` | `46.25754` | `46.613512` | `0.992887` |

### 033_multiple_objects_r01_p013_a_tv_and_a_laptop

Prompt: `a tv and a laptop`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953153` |
| latent_pt_mb | `20.9532` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `9458950` |
| mp4_mb | `9.4589` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4579469` | `4.5795` | `4.5755` | `4.575` | `9.181e-05` | `0.00813391` | `0.02579606` | `2.17509604` | `44.756019` | `45.66339` | `0.993852` |
| inter_delta_q8_zstd | `3489339` | `3.4893` | `6.0049` | `6.0043` | `0.00120925` | `0.02571788` | `0.28210533` | `5.34467173` | `40.851593` | `43.007611` | `0.991691` |

### 034_multiple_objects_r02_p027_a_teddy_bear_and_a_frisbee

Prompt: `a teddy bear and a frisbee`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953344` |
| latent_pt_mb | `20.9533` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `13284992` |
| mp4_mb | `13.285` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4494848` | `4.4948` | `4.6616` | `4.6611` | `0.00012191` | `0.00935536` | `0.02844441` | `1.81451309` | `45.543203` | `45.185281` | `0.993921` |
| inter_delta_q8_zstd | `3701399` | `3.7014` | `5.6609` | `5.6603` | `0.00229487` | `0.03545649` | `0.40606612` | `10.65371227` | `37.855794` | `40.095899` | `0.987664` |

### 035_multiple_objects_r03_p043_a_car_and_a_motorcycle

Prompt: `a car and a motorcycle`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953188` |
| latent_pt_mb | `20.9532` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `15037690` |
| mp4_mb | `15.0377` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4543633` | `4.5436` | `4.6115` | `4.6111` | `9.644e-05` | `0.00836799` | `0.0219625` | `0.84060806` | `48.884868` | `46.282292` | `0.993291` |
| inter_delta_q8_zstd | `4014657` | `4.0147` | `5.2192` | `5.2186` | `0.0012201` | `0.02532271` | `0.29076242` | `4.69495058` | `41.414493` | `42.795269` | `0.988827` |

### 036_human_action_r00_p012_a_person_is_skateboarding

Prompt: `A person is skateboarding`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953181` |
| latent_pt_mb | `20.9532` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `11040251` |
| mp4_mb | `11.0403` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4539306` | `4.5393` | `4.6159` | `4.6155` | `0.00012508` | `0.00942417` | `0.03068084` | `0.29464692` | `53.437785` | `48.772208` | `0.992562` |
| inter_delta_q8_zstd | `4157095` | `4.1571` | `5.0403` | `5.0398` | `0.00273667` | `0.03815823` | `0.50484633` | `2.8522892` | `43.578868` | `45.243384` | `0.989297` |

### 037_human_action_r01_p044_a_person_is_planting_trees

Prompt: `A person is planting trees`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953188` |
| latent_pt_mb | `20.9532` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `20566654` |
| mp4_mb | `20.5667` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4700010` | `4.7` | `4.4581` | `4.4577` | `9.741e-05` | `0.00833431` | `0.02287221` | `1.57091165` | `46.169286` | `44.215321` | `0.993415` |
| inter_delta_q8_zstd | `4107620` | `4.1076` | `5.1011` | `5.1005` | `0.00158764` | `0.02944197` | `0.31386334` | `12.00742912` | `37.336303` | `39.319188` | `0.988411` |

### 038_human_action_r02_p045_a_person_is_sharpening_knives

Prompt: `A person is sharpening knives`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953337` |
| latent_pt_mb | `20.9533` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `7488126` |
| mp4_mb | `7.4881` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4589475` | `4.5895` | `4.5655` | `4.565` | `9.857e-05` | `0.00836585` | `0.02729225` | `0.21541873` | `54.797969` | `49.549754` | `0.993469` |
| inter_delta_q8_zstd | `4067623` | `4.0676` | `5.1512` | `5.1507` | `0.00223193` | `0.03471573` | `0.41542482` | `1.95994711` | `45.20836` | `46.40958` | `0.99137` |

### 039_human_action_r03_p048_a_person_is_hula_hooping

Prompt: `A person is hula hooping`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953174` |
| latent_pt_mb | `20.9532` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `15924288` |
| mp4_mb | `15.9243` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4615998` | `4.616` | `4.5393` | `4.5388` | `0.00010634` | `0.00866256` | `0.03114557` | `0.51236522` | `51.035007` | `46.307038` | `0.991952` |
| inter_delta_q8_zstd | `4168274` | `4.1683` | `5.0268` | `5.0263` | `0.00219514` | `0.0347556` | `0.3714667` | `5.48041916` | `40.742666` | `42.324327` | `0.987124` |

### 040_color_r00_p005_a_purple_bicycle

Prompt: `a purple bicycle`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953069` |
| latent_pt_mb | `20.9531` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `15815144` |
| mp4_mb | `15.8151` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4661047` | `4.661` | `4.4954` | `4.4949` | `8.589e-05` | `0.00786725` | `0.02366376` | `0.97570103` | `48.237636` | `45.843418` | `0.994193` |
| inter_delta_q8_zstd | `4051392` | `4.0514` | `5.1718` | `5.1713` | `0.00146325` | `0.02810693` | `0.33756411` | `7.62236547` | `39.309906` | `41.283904` | `0.990149` |

### 041_color_r01_p033_a_blue_umbrella

Prompt: `a blue umbrella`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953062` |
| latent_pt_mb | `20.9531` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `14267481` |
| mp4_mb | `14.2675` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4620616` | `4.6206` | `4.5347` | `4.5343` | `0.00011482` | `0.00906915` | `0.02672291` | `1.9209069` | `45.29574` | `44.889925` | `0.99421` |
| inter_delta_q8_zstd | `3859225` | `3.8592` | `5.4293` | `5.4288` | `0.00214266` | `0.03402585` | `0.38130355` | `12.08673573` | `37.307713` | `39.870375` | `0.989682` |

### 042_color_r02_p058_a_red_chair

Prompt: `a red chair`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953034` |
| latent_pt_mb | `20.953` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `6731773` |
| mp4_mb | `6.7318` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4760169` | `4.7602` | `4.4017` | `4.4013` | `0.00010142` | `0.00849255` | `0.02790967` | `0.55660444` | `50.675337` | `47.446649` | `0.994157` |
| inter_delta_q8_zstd | `3535123` | `3.5351` | `5.9271` | `5.9265` | `0.00110467` | `0.02428728` | `0.2834928` | `2.32372117` | `44.468963` | `45.408276` | `0.993102` |

### 043_color_r03_p077_a_green_vase

Prompt: `a green vase`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953041` |
| latent_pt_mb | `20.953` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `4750295` |
| mp4_mb | `4.7503` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4824473` | `4.8245` | `4.3431` | `4.3427` | `7.627e-05` | `0.00737104` | `0.02202159` | `0.23661347` | `54.390409` | `50.116907` | `0.995072` |
| inter_delta_q8_zstd | `3680756` | `3.6808` | `5.6926` | `5.692` | `0.00037839` | `0.01434796` | `0.16562891` | `0.55246103` | `50.707787` | `49.247239` | `0.99471` |

### 044_spatial_relationship_r00_p010_a_bird_on_the_left_of_a_cat_front_view

Prompt: `a bird on the left of a cat, front view`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953712` |
| latent_pt_mb | `20.9537` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `8526562` |
| mp4_mb | `8.5266` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4670871` | `4.6709` | `4.486` | `4.4855` | `9.585e-05` | `0.00829378` | `0.02278213` | `0.96809632` | `48.271618` | `46.740691` | `0.995035` |
| inter_delta_q8_zstd | `3476505` | `3.4765` | `6.0272` | `6.0265` | `0.00195504` | `0.03253933` | `0.38307148` | `6.89520311` | `39.745333` | `41.994873` | `0.992094` |

### 045_spatial_relationship_r01_p015_a_cow_on_the_right_of_an_elephant_front_view

Prompt: `a cow on the right of an elephant, front view`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953754` |
| latent_pt_mb | `20.9538` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `12016409` |
| mp4_mb | `12.0164` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4641475` | `4.6415` | `4.5145` | `4.5139` | `9.597e-05` | `0.00829931` | `0.02361917` | `0.84335333` | `48.870708` | `46.433428` | `0.993725` |
| inter_delta_q8_zstd | `3897967` | `3.898` | `5.3756` | `5.3749` | `0.00168454` | `0.02986214` | `0.37671047` | `5.58406878` | `40.661296` | `42.64225` | `0.989991` |

### 046_spatial_relationship_r02_p048_a_train_on_the_right_of_a_boat_front_view

Prompt: `a train on the right of a boat, front view`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953733` |
| latent_pt_mb | `20.9537` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `12799328` |
| mp4_mb | `12.7993` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4482354` | `4.4824` | `4.6747` | `4.6741` | `0.0001147` | `0.00906668` | `0.02884912` | `1.10388076` | `47.701582` | `45.894971` | `0.994323` |
| inter_delta_q8_zstd | `4140901` | `4.1409` | `5.0602` | `5.0595` | `0.0011412` | `0.02410429` | `0.33709908` | `5.48720121` | `40.737295` | `42.376247` | `0.991207` |

### 047_spatial_relationship_r03_p068_a_pizza_on_the_top_of_a_donut_front_view

Prompt: `a pizza on the top of a donut, front view`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953726` |
| latent_pt_mb | `20.9537` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `5024186` |
| mp4_mb | `5.0242` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4735260` | `4.7353` | `4.425` | `4.4245` | `8.913e-05` | `0.00800343` | `0.02230197` | `0.26826078` | `53.845232` | `49.639759` | `0.995493` |
| inter_delta_q8_zstd | `3508746` | `3.5087` | `5.9719` | `5.9711` | `0.00111527` | `0.02452926` | `0.2860316` | `1.14837992` | `47.529948` | `47.929291` | `0.99474` |

### 048_scene_r00_p037_golf_course

Prompt: `golf course`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953034` |
| latent_pt_mb | `20.953` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `13373887` |
| mp4_mb | `13.3739` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4214746` | `4.2147` | `4.9714` | `4.9709` | `0.00013661` | `0.00994694` | `0.03022307` | `1.24392164` | `47.182873` | `46.32349` | `0.990202` |
| inter_delta_q8_zstd | `3668229` | `3.6682` | `5.712` | `5.7115` | `0.00131153` | `0.02636414` | `0.33395183` | `4.10089254` | `42.00202` | `43.590656` | `0.984788` |

### 049_scene_r01_p070_sky

Prompt: `sky`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20952978` |
| latent_pt_mb | `20.953` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `25009167` |
| mp4_mb | `25.0092` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4168068` | `4.1681` | `5.027` | `5.0266` | `0.00011016` | `0.00896515` | `0.02417016` | `2.16457582` | `44.777076` | `43.472519` | `0.990227` |
| inter_delta_q8_zstd | `3849399` | `3.8494` | `5.4432` | `5.4427` | `0.00121307` | `0.02556152` | `0.33935022` | `12.53730774` | `37.148761` | `38.609466` | `0.979416` |

### 050_scene_r02_p079_train_railway

Prompt: `train railway`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953048` |
| latent_pt_mb | `20.953` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `20093707` |
| mp4_mb | `20.0937` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4497421` | `4.4974` | `4.6589` | `4.6585` | `9.77e-05` | `0.00842009` | `0.02311784` | `1.32893515` | `46.895766` | `44.750633` | `0.992551` |
| inter_delta_q8_zstd | `4059243` | `4.0592` | `5.1618` | `5.1613` | `0.0017786` | `0.03131594` | `0.34527302` | `13.08779907` | `36.962137` | `39.212853` | `0.984436` |

### 051_scene_r03_p080_train_station_platform

Prompt: `train station platform`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953111` |
| latent_pt_mb | `20.9531` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `15689645` |
| mp4_mb | `15.6896` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4452286` | `4.4523` | `4.7061` | `4.7057` | `0.00012585` | `0.00948941` | `0.03041166` | `2.20449662` | `44.697709` | `44.350936` | `0.992412` |
| inter_delta_q8_zstd | `3826194` | `3.8262` | `5.4762` | `5.4757` | `0.00143668` | `0.02772328` | `0.33423689` | `7.9026165` | `39.153095` | `41.062649` | `0.98874` |

### 052_temporal_style_r00_p024_a_shark_is_swimming_in_the_ocean_pan_right

Prompt: `a shark is swimming in the ocean, pan right`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953634` |
| latent_pt_mb | `20.9536` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `7870360` |
| mp4_mb | `7.8704` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4351994` | `4.352` | `4.8147` | `4.8141` | `0.00010881` | `0.00882241` | `0.02609277` | `0.13873763` | `56.708861` | `52.040083` | `0.995484` |
| inter_delta_q8_zstd | `4101818` | `4.1018` | `5.1084` | `5.1077` | `0.00186206` | `0.03179273` | `0.39693773` | `0.96361375` | `48.291774` | `49.489466` | `0.994262` |

### 053_temporal_style_r01_p046_a_cute_happy_corgi_playing_in_park_sunset_tilt_down

Prompt: `A cute happy Corgi playing in park, sunset, tilt down`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953761` |
| latent_pt_mb | `20.9538` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `12813468` |
| mp4_mb | `12.8135` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4628130` | `4.6281` | `4.5275` | `4.5269` | `9.741e-05` | `0.0083668` | `0.02415633` | `0.52718669` | `50.911159` | `46.737373` | `0.993833` |
| inter_delta_q8_zstd | `4057079` | `4.0571` | `5.1647` | `5.1641` | `0.00164647` | `0.02980032` | `0.38446787` | `5.22946215` | `40.946233` | `42.364817` | `0.990328` |

### 054_temporal_style_r02_p073_a_couple_in_formal_evening_wear_going_home_get_caught_in_a_heavy_downpour_with_u

Prompt: `A couple in formal evening wear going home get caught in a heavy downpour with umbrellas, pan left`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20954092` |
| latent_pt_mb | `20.9541` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `9259362` |
| mp4_mb | `9.2594` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4629449` | `4.6294` | `4.5263` | `4.5256` | `8.889e-05` | `0.00800825` | `0.02242291` | `0.25088885` | `54.13599` | `49.301137` | `0.993917` |
| inter_delta_q8_zstd | `4197975` | `4.198` | `4.9915` | `4.9907` | `0.00166148` | `0.02975595` | `0.35364652` | `1.85225713` | `45.453791` | `46.528651` | `0.992023` |

### 055_temporal_style_r03_p090_snow_rocky_mountains_peaks_canyon_snow_blanketed_rocky_mountains_surround_and_sh

Prompt: `Snow rocky mountains peaks canyon. snow blanketed rocky mountains surround and shadow deep canyons. the canyons twist and bend through the high elevated mountain peaks, in super slow motion`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20954220` |
| latent_pt_mb | `20.9542` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `14214003` |
| mp4_mb | `14.214` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4669625` | `4.6696` | `4.4873` | `4.4867` | `6.039e-05` | `0.00662824` | `0.01866627` | `7.54406309` | `39.35475` | `41.297606` | `0.991119` |
| inter_delta_q8_zstd | `3444399` | `3.4444` | `6.0836` | `6.0826` | `0.00037477` | `0.01423787` | `0.17811602` | `5.42593384` | `40.786059` | `42.492406` | `0.992355` |

### 056_appearance_style_r00_p005_a_beautiful_coastal_beach_in_spring_waves_lapping_on_sand_in_cyberpunk_style

Prompt: `A beautiful coastal beach in spring, waves lapping on sand, in cyberpunk style`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20954078` |
| latent_pt_mb | `20.9541` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `14586478` |
| mp4_mb | `14.5865` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4585821` | `4.5858` | `4.5693` | `4.5687` | `6.649e-05` | `0.00693903` | `0.01875627` | `0.5864045` | `50.448831` | `46.759253` | `0.992892` |
| inter_delta_q8_zstd | `4061818` | `4.0618` | `5.1588` | `5.158` | `0.00084841` | `0.02136487` | `0.23607016` | `3.59308076` | `42.576134` | `43.671548` | `0.98948` |

### 057_appearance_style_r01_p008_a_beautiful_coastal_beach_in_spring_waves_lapping_on_sand_surrealism_style

Prompt: `A beautiful coastal beach in spring, waves lapping on sand, surrealism style`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20954064` |
| latent_pt_mb | `20.9541` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `8895725` |
| mp4_mb | `8.8957` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4534184` | `4.5342` | `4.6214` | `4.6207` | `6.996e-05` | `0.00710384` | `0.02165723` | `0.26171395` | `53.952535` | `49.083085` | `0.992858` |
| inter_delta_q8_zstd | `4127688` | `4.1277` | `5.0765` | `5.0757` | `0.00073336` | `0.01980261` | `0.22719753` | `1.08328712` | `47.783368` | `47.491003` | `0.991427` |

### 058_appearance_style_r02_p029_a_panda_drinking_coffee_in_a_cafe_in_paris_by_hokusai_in_the_style_of_ukiyo

Prompt: `A panda drinking coffee in a cafe in Paris by Hokusai, in the style of Ukiyo`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20954071` |
| latent_pt_mb | `20.9541` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `8506249` |
| mp4_mb | `8.5062` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4648243` | `4.6482` | `4.508` | `4.5073` | `0.00010825` | `0.0088222` | `0.02722999` | `1.80570138` | `45.564344` | `45.636565` | `0.993065` |
| inter_delta_q8_zstd | `3194205` | `3.1942` | `6.56` | `6.5591` | `0.0014475` | `0.02818068` | `0.35730737` | `9.69910812` | `38.263486` | `40.697815` | `0.989505` |

### 059_appearance_style_r03_p084_snow_rocky_mountains_peaks_canyon_snow_blanketed_rocky_mountains_surround_and_sh

Prompt: `Snow rocky mountains peaks canyon. snow blanketed rocky mountains surround and shadow deep canyons. the canyons twist and bend through the high elevated mountain peaks, black and white`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20954234` |
| latent_pt_mb | `20.9542` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `15675974` |
| mp4_mb | `15.676` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4700443` | `4.7004` | `4.4579` | `4.4572` | `6.322e-05` | `0.00676928` | `0.01890206` | `8.43135262` | `38.871831` | `40.805285` | `0.991383` |
| inter_delta_q8_zstd | `3546961` | `3.547` | `5.9077` | `5.9068` | `0.00038175` | `0.01435225` | `0.17232466` | `6.76683712` | `39.826946` | `41.690592` | `0.992248` |

### 060_overall_consistency_r00_p010_fireworks

Prompt: `Fireworks.`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953118` |
| latent_pt_mb | `20.9531` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `21274771` |
| mp4_mb | `21.2748` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4751530` | `4.7515` | `4.4098` | `4.4093` | `0.00013847` | `0.00989159` | `0.02919066` | `1.97465551` | `45.17589` | `44.690656` | `0.995904` |
| inter_delta_q8_zstd | `4428718` | `4.4287` | `4.7312` | `4.7307` | `0.00229766` | `0.03509945` | `0.37348324` | `23.70106506` | `34.383125` | `37.492347` | `0.9901` |

### 061_overall_consistency_r01_p012_flying_through_fantasy_landscapes

Prompt: `Flying through fantasy landscapes.`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20953542` |
| latent_pt_mb | `20.9535` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `23854304` |
| mp4_mb | `23.8543` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4481666` | `4.4817` | `4.6754` | `4.6748` | `9.909e-05` | `0.0084869` | `0.02354413` | `1.32515299` | `46.908143` | `44.840041` | `0.990305` |
| inter_delta_q8_zstd | `4173996` | `4.174` | `5.02` | `5.0194` | `0.00154117` | `0.0291101` | `0.31482878` | `14.86688232` | `36.408605` | `38.62013` | `0.979035` |

### 062_overall_consistency_r02_p029_campfire_at_night_in_a_snowy_forest_with_starry_sky_in_the_background

Prompt: `Campfire at night in a snowy forest with starry sky in the background.`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20954050` |
| latent_pt_mb | `20.954` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `6808900` |
| mp4_mb | `6.8089` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4615462` | `4.6155` | `4.54` | `4.5393` | `0.00013509` | `0.00975429` | `0.0318917` | `0.33520573` | `52.877689` | `49.786367` | `0.995531` |
| inter_delta_q8_zstd | `3419731` | `3.4197` | `6.1274` | `6.1265` | `0.00222397` | `0.03429649` | `0.40682757` | `2.96797705` | `43.406198` | `46.072555` | `0.993562` |

### 063_overall_consistency_r03_p037_an_astronaut_is_riding_a_horse_in_the_space_in_a_photorealistic_style

Prompt: `An astronaut is riding a horse in the space in a photorealistic style.`


| Video config | Value |
|---|---:|
| fps | `24` |
| frame_num | `121` |
| size | `1280*704` |
| latent_shape | `[48, 31, 44, 80]` |

| Baseline metric | Value |
|---|---:|
| raw_frame_bytes | `327106560` |
| raw_frame_mb | `327.1066` |
| latent_pt_bytes | `20954050` |
| latent_pt_mb | `20.954` |
| latent_raw_bytes | `20951040` |
| latent_raw_mb | `20.951` |
| mp4_bytes | `10213380` |
| mp4_mb | `10.2134` |

| Codec | latent_bytes | latent_mb | vs_pt_ratio | vs_raw_ratio | latent_mse | latent_mae | latent_max_abs | raw_frame_mse | raw_psnr_db | mp4_psnr_db | mp4_ssim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| intra_q8_zstd | `4742676` | `4.7427` | `4.4182` | `4.4176` | `9.288e-05` | `0.0081795` | `0.02452874` | `0.93359846` | `48.429202` | `46.958159` | `0.995594` |
| inter_delta_q8_zstd | `4280739` | `4.2807` | `4.895` | `4.8943` | `0.00134606` | `0.02677224` | `0.29299283` | `7.00753641` | `39.67515` | `41.844574` | `0.993786` |
