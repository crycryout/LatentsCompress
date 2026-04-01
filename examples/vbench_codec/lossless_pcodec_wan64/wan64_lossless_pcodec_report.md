# Wan2.2 64 Latents Lossless PCodec Report

This report compresses the original Wan2.2 latent tensor payload directly with lossless `pcodec`, then compares the resulting size against the previously measured lossless `zstd` and `openzl` baselines.

Important note:

- `pcodec` operates on the numeric latent tensor payload (`latents`) rather than the full serialized `.pt` container.
- For these files the `.pt` metadata overhead is tiny, so the comparison is still useful, but it is not a byte-identical container-vs-container benchmark.

## Summary

- sample_count: `64`
- compression_level: `8`
- total_original_pt_mb: `1341.023296`
- total_original_tensor_mb: `1340.86656`
- total_pcodec_mb: `1085.278405`
- total_zstd_mb: `1224.206334`
- total_openzl_mb: `1224.839651`
- total_pcodec_saved_vs_tensor_mb: `255.588155`
- total_pcodec_delta_vs_zstd_mb: `-138.927929`
- total_pcodec_delta_vs_openzl_mb: `-139.561246`
- mean_original_tensor_mb: `20.95104`
- mean_pcodec_mb: `16.957475`
- mean_zstd_mb: `19.128224`
- mean_openzl_mb: `19.13812`
- mean_pcodec_delta_vs_zstd_mb: `-2.170749`
- mean_pcodec_delta_vs_openzl_mb: `-2.180644`
- count_pcodec_smaller_than_zstd: `64`
- count_pcodec_smaller_than_openzl: `64`
- all_verified_lossless: `True`

## Rows

| name | tensor_mb | pcodec_mb | zstd_mb | openzl_mb | pcodec_vs_zstd_mb | pcodec_vs_openzl_mb | verified_lossless |
|---|---:|---:|---:|---:|---:|---:|---:|
| `000_subject_consistency_r00_p003_a_person_eating_a_burger` | `20.95104` | `17.037039` | `19.196455` | `19.192452` | `-2.159416` | `-2.155413` | `True` |
| `001_subject_consistency_r01_p014_a_car_accelerating_to_gain_speed` | `20.95104` | `16.942656` | `19.248919` | `19.252123` | `-2.306263` | `-2.309467` | `True` |
| `002_subject_consistency_r02_p031_a_truck_anchored_in_a_tranquil_bay` | `20.95104` | `16.348334` | `18.946847` | `18.984823` | `-2.598513` | `-2.636489` | `True` |
| `003_subject_consistency_r03_p035_a_boat_sailing_smoothly_on_a_calm_lake` | `20.95104` | `16.538767` | `19.056187` | `19.080351` | `-2.51742` | `-2.541584` | `True` |
| `004_background_consistency_r00_p013_bridge` | `20.95104` | `16.560867` | `19.015279` | `19.046269` | `-2.454412` | `-2.485402` | `True` |
| `005_background_consistency_r01_p017_campus` | `20.95104` | `16.83274` | `19.133089` | `19.156024` | `-2.300349` | `-2.323284` | `True` |
| `006_background_consistency_r02_p028_downtown` | `20.95104` | `17.056508` | `19.140739` | `19.151372` | `-2.084231` | `-2.094864` | `True` |
| `007_background_consistency_r03_p069_ski_slope` | `20.95104` | `16.652053` | `19.06913` | `19.097315` | `-2.417077` | `-2.445262` | `True` |
| `008_temporal_flickering_r00_p003_a_tranquil_tableau_of_alley` | `20.95104` | `17.00013` | `19.09579` | `19.106864` | `-2.09566` | `-2.106734` | `True` |
| `009_temporal_flickering_r01_p004_a_tranquil_tableau_of_bar` | `20.95104` | `17.035004` | `19.13838` | `19.144557` | `-2.103376` | `-2.109553` | `True` |
| `010_temporal_flickering_r02_p011_a_tranquil_tableau_of_house` | `20.95104` | `16.808801` | `19.074892` | `19.095883` | `-2.266091` | `-2.287082` | `True` |
| `011_temporal_flickering_r03_p054_a_tranquil_tableau_of_in_the_heart_of_plaka_the_neoclassical_architecture_of_the` | `20.95104` | `16.866316` | `19.074794` | `19.090601` | `-2.208478` | `-2.224285` | `True` |
| `012_motion_smoothness_r00_p011_a_car_stuck_in_traffic_during_rush_hour` | `20.95104` | `16.912936` | `19.205602` | `19.207137` | `-2.292666` | `-2.294201` | `True` |
| `013_motion_smoothness_r01_p027_a_train_speeding_down_the_tracks` | `20.95104` | `16.963467` | `19.229296` | `19.239311` | `-2.265829` | `-2.275844` | `True` |
| `014_motion_smoothness_r02_p029_a_train_accelerating_to_gain_speed` | `20.95104` | `16.849134` | `19.272225` | `19.270374` | `-2.423091` | `-2.42124` | `True` |
| `015_motion_smoothness_r03_p064_a_bear_climbing_a_tree` | `20.95104` | `17.402995` | `19.249323` | `19.245059` | `-1.846328` | `-1.842064` | `True` |
| `016_dynamic_degree_r00_p003_a_person_eating_a_burger` | `20.95104` | `17.037039` | `19.195698` | `19.192351` | `-2.158659` | `-2.155312` | `True` |
| `017_dynamic_degree_r01_p025_a_bus_stuck_in_traffic_during_rush_hour` | `20.95104` | `17.042109` | `19.219874` | `19.223248` | `-2.177765` | `-2.181139` | `True` |
| `018_dynamic_degree_r02_p069_a_giraffe_bending_down_to_drink_water_from_a_river` | `20.95104` | `17.196082` | `19.163242` | `19.182935` | `-1.96716` | `-1.986853` | `True` |
| `019_dynamic_degree_r03_p071_a_giraffe_running_to_join_a_herd_of_its_kind` | `20.95104` | `16.727583` | `19.037801` | `19.074254` | `-2.310218` | `-2.346671` | `True` |
| `020_aesthetic_quality_r00_p028_origami_dancers_in_white_paper_3d_render_on_white_background_studio_shot_dancing` | `20.95104` | `16.976261` | `19.002682` | `19.00382` | `-2.026421` | `-2.027559` | `True` |
| `021_aesthetic_quality_r01_p053_a_jellyfish_floating_through_the_ocean_with_bioluminescent_tentacles` | `20.95104` | `16.544369` | `18.648128` | `18.675305` | `-2.103759` | `-2.130936` | `True` |
| `022_aesthetic_quality_r02_p057_a_steam_train_moving_on_a_mountainside` | `20.95104` | `16.913273` | `19.220958` | `19.231604` | `-2.307685` | `-2.318331` | `True` |
| `023_aesthetic_quality_r03_p075_a_happy_fuzzy_panda_playing_guitar_nearby_a_campfire_snow_mountain_in_the_backgr` | `20.95104` | `17.159358` | `19.270686` | `19.268386` | `-2.111328` | `-2.109028` | `True` |
| `024_imaging_quality_r00_p000_close_up_of_grapes_on_a_rotating_table` | `20.95104` | `17.116759` | `19.273631` | `19.271063` | `-2.156872` | `-2.154304` | `True` |
| `025_imaging_quality_r01_p020_a_shark_is_swimming_in_the_ocean` | `20.95104` | `16.211276` | `18.629673` | `18.634337` | `-2.418397` | `-2.423061` | `True` |
| `026_imaging_quality_r02_p035_busy_freeway_at_night` | `20.95104` | `16.32756` | `18.933355` | `18.959889` | `-2.605795` | `-2.632329` | `True` |
| `027_imaging_quality_r03_p089_hyper_realistic_spaceship_landing_on_mars` | `20.95104` | `17.041338` | `19.149625` | `19.14842` | `-2.108287` | `-2.107082` | `True` |
| `028_object_class_r00_p019_a_cow` | `20.95104` | `16.967301` | `19.167705` | `19.179334` | `-2.200404` | `-2.212033` | `True` |
| `029_object_class_r01_p035_a_baseball_glove` | `20.95104` | `17.035784` | `19.239508` | `19.23984` | `-2.203724` | `-2.204056` | `True` |
| `030_object_class_r02_p043_a_knife` | `20.95104` | `17.398196` | `19.229142` | `19.227965` | `-1.830946` | `-1.829769` | `True` |
| `031_object_class_r03_p054_a_donut` | `20.95104` | `17.168694` | `19.186297` | `19.192086` | `-2.017603` | `-2.023392` | `True` |
| `032_multiple_objects_r00_p011_a_couch_and_a_potted_plant` | `20.95104` | `17.307884` | `19.229808` | `19.229126` | `-1.921924` | `-1.921242` | `True` |
| `033_multiple_objects_r01_p013_a_tv_and_a_laptop` | `20.95104` | `17.135436` | `19.221691` | `19.219888` | `-2.086255` | `-2.084452` | `True` |
| `034_multiple_objects_r02_p027_a_teddy_bear_and_a_frisbee` | `20.95104` | `17.085601` | `19.207647` | `19.212046` | `-2.122046` | `-2.126445` | `True` |
| `035_multiple_objects_r03_p043_a_car_and_a_motorcycle` | `20.95104` | `17.063776` | `19.241339` | `19.243629` | `-2.177563` | `-2.179853` | `True` |
| `036_human_action_r00_p012_a_person_is_skateboarding` | `20.95104` | `17.03343` | `19.237382` | `19.242333` | `-2.203952` | `-2.208903` | `True` |
| `037_human_action_r01_p044_a_person_is_planting_trees` | `20.95104` | `17.269273` | `19.236221` | `19.23452` | `-1.966948` | `-1.965247` | `True` |
| `038_human_action_r02_p045_a_person_is_sharpening_knives` | `20.95104` | `17.266179` | `19.314131` | `19.3084` | `-2.047952` | `-2.042221` | `True` |
| `039_human_action_r03_p048_a_person_is_hula_hooping` | `20.95104` | `17.127138` | `19.223637` | `19.231328` | `-2.096499` | `-2.10419` | `True` |
| `040_color_r00_p005_a_purple_bicycle` | `20.95104` | `17.172337` | `19.240626` | `19.242142` | `-2.068289` | `-2.069805` | `True` |
| `041_color_r01_p033_a_blue_umbrella` | `20.95104` | `17.056087` | `19.139307` | `19.15759` | `-2.08322` | `-2.101503` | `True` |
| `042_color_r02_p058_a_red_chair` | `20.95104` | `17.194735` | `19.248359` | `19.246897` | `-2.053624` | `-2.052162` | `True` |
| `043_color_r03_p077_a_green_vase` | `20.95104` | `17.270546` | `19.143487` | `19.147303` | `-1.872941` | `-1.876757` | `True` |
| `044_spatial_relationship_r00_p010_a_bird_on_the_left_of_a_cat_front_view` | `20.95104` | `17.349844` | `19.229812` | `19.231261` | `-1.879968` | `-1.881417` | `True` |
| `045_spatial_relationship_r01_p015_a_cow_on_the_right_of_an_elephant_front_view` | `20.95104` | `17.205875` | `19.207504` | `19.206861` | `-2.001629` | `-2.000986` | `True` |
| `046_spatial_relationship_r02_p048_a_train_on_the_right_of_a_boat_front_view` | `20.95104` | `16.91779` | `19.203575` | `19.214237` | `-2.285785` | `-2.296447` | `True` |
| `047_spatial_relationship_r03_p068_a_pizza_on_the_top_of_a_donut_front_view` | `20.95104` | `17.183839` | `19.157555` | `19.154281` | `-1.973716` | `-1.970442` | `True` |
| `048_scene_r00_p037_golf_course` | `20.95104` | `16.568119` | `19.049102` | `19.080097` | `-2.480983` | `-2.511978` | `True` |
| `049_scene_r01_p070_sky` | `20.95104` | `16.377131` | `18.947163` | `18.983457` | `-2.570032` | `-2.606326` | `True` |
| `050_scene_r02_p079_train_railway` | `20.95104` | `16.913895` | `19.17579` | `19.190907` | `-2.261895` | `-2.277012` | `True` |
| `051_scene_r03_p080_train_station_platform` | `20.95104` | `16.94811` | `19.156069` | `19.167587` | `-2.207959` | `-2.219477` | `True` |
| `052_temporal_style_r00_p024_a_shark_is_swimming_in_the_ocean_pan_right` | `20.95104` | `16.435321` | `18.67452` | `18.717815` | `-2.239199` | `-2.282494` | `True` |
| `053_temporal_style_r01_p046_a_cute_happy_corgi_playing_in_park_sunset_tilt_down` | `20.95104` | `16.990895` | `19.165886` | `19.185851` | `-2.174991` | `-2.194956` | `True` |
| `054_temporal_style_r02_p073_a_couple_in_formal_evening_wear_going_home_get_caught_in_a_heavy_downpour_with_u` | `20.95104` | `17.280264` | `19.139664` | `19.143834` | `-1.8594` | `-1.86357` | `True` |
| `055_temporal_style_r03_p090_snow_rocky_mountains_peaks_canyon_snow_blanketed_rocky_mountains_surround_and_sh` | `20.95104` | `16.977897` | `19.086813` | `19.097279` | `-2.108916` | `-2.119382` | `True` |
| `056_appearance_style_r00_p005_a_beautiful_coastal_beach_in_spring_waves_lapping_on_sand_in_cyberpunk_style` | `20.95104` | `16.818661` | `19.075789` | `19.089971` | `-2.257128` | `-2.27131` | `True` |
| `057_appearance_style_r01_p008_a_beautiful_coastal_beach_in_spring_waves_lapping_on_sand_surrealism_style` | `20.95104` | `16.615434` | `18.965461` | `18.985347` | `-2.350027` | `-2.369913` | `True` |
| `058_appearance_style_r02_p029_a_panda_drinking_coffee_in_a_cafe_in_paris_by_hokusai_in_the_style_of_ukiyo` | `20.95104` | `17.093049` | `19.208306` | `19.211916` | `-2.115257` | `-2.118867` | `True` |
| `059_appearance_style_r03_p084_snow_rocky_mountains_peaks_canyon_snow_blanketed_rocky_mountains_surround_and_sh` | `20.95104` | `17.01584` | `19.091608` | `19.091858` | `-2.075768` | `-2.076018` | `True` |
| `060_overall_consistency_r00_p010_fireworks` | `20.95104` | `17.135031` | `19.052537` | `19.063622` | `-1.917506` | `-1.928591` | `True` |
| `061_overall_consistency_r01_p012_flying_through_fantasy_landscapes` | `20.95104` | `16.872999` | `19.179482` | `19.195843` | `-2.306483` | `-2.322844` | `True` |
| `062_overall_consistency_r02_p029_campfire_at_night_in_a_snowy_forest_with_starry_sky_in_the_background` | `20.95104` | `16.969327` | `19.066068` | `19.06524` | `-2.096741` | `-2.095913` | `True` |
| `063_overall_consistency_r03_p037_an_astronaut_is_riding_a_horse_in_the_space_in_a_photorealistic_style` | `20.95104` | `16.955933` | `18.975043` | `18.981833` | `-2.01911` | `-2.0259` | `True` |
