# Wan2.2 64 Latents Lossless Zstd Report

This report compresses the saved Wan2.2 latent `.pt` files directly with `zstd`.

Properties:

- compression target: original `.pt` latent file bytes
- compression type: lossless `zstd`
- verification: decompressed bytes checked against original bytes

## Summary

- sample_count: `64`
- zstd_level: `19`
- total_original_bytes: `1341023296`
- total_compressed_bytes: `1224206334`
- total_original_mb: `1341.023296`
- total_compressed_mb: `1224.206334`
- total_bytes_saved: `116816962`
- total_mb_saved: `116.816962`
- total_compression_ratio: `1.095423`
- mean_original_mb: `20.953489`
- mean_compressed_mb: `19.128224`
- mean_bytes_saved_mb: `1.825265`
- mean_compression_ratio: `1.095482`
- all_verified_lossless: `True`

## Rows

| name | original_mb | compressed_mb | saved_mb | compression_ratio | saved_percent | verified_lossless |
|---|---:|---:|---:|---:|---:|---:|
| `000_subject_consistency_r00_p003_a_person_eating_a_burger` | `20.953351` | `19.196455` | `1.756896` | `1.091522` | `8.3848` | `True` |
| `001_subject_consistency_r01_p014_a_car_accelerating_to_gain_speed` | `20.953471` | `19.248919` | `1.704552` | `1.088553` | `8.1349` | `True` |
| `002_subject_consistency_r02_p031_a_truck_anchored_in_a_tranquil_bay` | `20.953613` | `18.946847` | `2.006766` | `1.105916` | `9.5772` | `True` |
| `003_subject_consistency_r03_p035_a_boat_sailing_smoothly_on_a_calm_lake` | `20.953705` | `19.056187` | `1.897518` | `1.099575` | `9.0558` | `True` |
| `004_background_consistency_r00_p013_bridge` | `20.953118` | `19.015279` | `1.937839` | `1.10191` | `9.2485` | `True` |
| `005_background_consistency_r01_p017_campus` | `20.953118` | `19.133089` | `1.820029` | `1.095125` | `8.6862` | `True` |
| `006_background_consistency_r02_p028_downtown` | `20.953132` | `19.140739` | `1.812393` | `1.094688` | `8.6497` | `True` |
| `007_background_consistency_r03_p069_ski_slope` | `20.953139` | `19.06913` | `1.884009` | `1.098799` | `8.9915` | `True` |
| `008_temporal_flickering_r00_p003_a_tranquil_tableau_of_alley` | `20.953436` | `19.09579` | `1.857646` | `1.09728` | `8.8656` | `True` |
| `009_temporal_flickering_r01_p004_a_tranquil_tableau_of_bar` | `20.953358` | `19.13838` | `1.814978` | `1.094834` | `8.662` | `True` |
| `010_temporal_flickering_r02_p011_a_tranquil_tableau_of_house` | `20.953436` | `19.074892` | `1.878544` | `1.098483` | `8.9653` | `True` |
| `011_temporal_flickering_r03_p054_a_tranquil_tableau_of_in_the_heart_of_plaka_the_neoclassical_architecture_of_the` | `20.954191` | `19.074794` | `1.879397` | `1.098528` | `8.9691` | `True` |
| `012_motion_smoothness_r00_p011_a_car_stuck_in_traffic_during_rush_hour` | `20.953634` | `19.205602` | `1.748032` | `1.091017` | `8.3424` | `True` |
| `013_motion_smoothness_r01_p027_a_train_speeding_down_the_tracks` | `20.953457` | `19.229296` | `1.724161` | `1.089663` | `8.2285` | `True` |
| `014_motion_smoothness_r02_p029_a_train_accelerating_to_gain_speed` | `20.953471` | `19.272225` | `1.681246` | `1.087237` | `8.0237` | `True` |
| `015_motion_smoothness_r03_p064_a_bear_climbing_a_tree` | `20.953195` | `19.249323` | `1.703872` | `1.088516` | `8.1318` | `True` |
| `016_dynamic_degree_r00_p003_a_person_eating_a_burger` | `20.953188` | `19.195698` | `1.75749` | `1.091556` | `8.3877` | `True` |
| `017_dynamic_degree_r01_p025_a_bus_stuck_in_traffic_during_rush_hour` | `20.953613` | `19.219874` | `1.733739` | `1.090206` | `8.2742` | `True` |
| `018_dynamic_degree_r02_p069_a_giraffe_bending_down_to_drink_water_from_a_river` | `20.953754` | `19.163242` | `1.790512` | `1.093435` | `8.5451` | `True` |
| `019_dynamic_degree_r03_p071_a_giraffe_running_to_join_a_herd_of_its_kind` | `20.953712` | `19.037801` | `1.915911` | `1.100637` | `9.1435` | `True` |
| `020_aesthetic_quality_r00_p028_origami_dancers_in_white_paper_3d_render_on_white_background_studio_shot_dancing` | `20.954177` | `19.002682` | `1.951495` | `1.102696` | `9.3132` | `True` |
| `021_aesthetic_quality_r01_p053_a_jellyfish_floating_through_the_ocean_with_bioluminescent_tentacles` | `20.954029` | `18.648128` | `2.305901` | `1.123653` | `11.0046` | `True` |
| `022_aesthetic_quality_r02_p057_a_steam_train_moving_on_a_mountainside` | `20.953627` | `19.220958` | `1.732669` | `1.090145` | `8.2691` | `True` |
| `023_aesthetic_quality_r03_p075_a_happy_fuzzy_panda_playing_guitar_nearby_a_campfire_snow_mountain_in_the_backgr` | `20.954113` | `19.270686` | `1.683427` | `1.087357` | `8.0339` | `True` |
| `024_imaging_quality_r00_p000_close_up_of_grapes_on_a_rotating_table` | `20.953613` | `19.273631` | `1.679982` | `1.087165` | `8.0176` | `True` |
| `025_imaging_quality_r01_p020_a_shark_is_swimming_in_the_ocean` | `20.953443` | `18.629673` | `2.32377` | `1.124735` | `11.0902` | `True` |
| `026_imaging_quality_r02_p035_busy_freeway_at_night` | `20.953174` | `18.933355` | `2.019819` | `1.10668` | `9.6397` | `True` |
| `027_imaging_quality_r03_p089_hyper_realistic_spaceship_landing_on_mars` | `20.953634` | `19.149625` | `1.804009` | `1.094206` | `8.6095` | `True` |
| `028_object_class_r00_p019_a_cow` | `20.953041` | `19.167705` | `1.785336` | `1.093143` | `8.5207` | `True` |
| `029_object_class_r01_p035_a_baseball_glove` | `20.953118` | `19.239508` | `1.71361` | `1.089067` | `8.1783` | `True` |
| `030_object_class_r02_p043_a_knife` | `20.953055` | `19.229142` | `1.723913` | `1.089651` | `8.2275` | `True` |
| `031_object_class_r03_p054_a_donut` | `20.953055` | `19.186297` | `1.766758` | `1.092084` | `8.432` | `True` |
| `032_multiple_objects_r00_p011_a_couch_and_a_potted_plant` | `20.953344` | `19.229808` | `1.723536` | `1.089628` | `8.2256` | `True` |
| `033_multiple_objects_r01_p013_a_tv_and_a_laptop` | `20.953153` | `19.221691` | `1.731462` | `1.090079` | `8.2635` | `True` |
| `034_multiple_objects_r02_p027_a_teddy_bear_and_a_frisbee` | `20.953344` | `19.207647` | `1.745697` | `1.090886` | `8.3314` | `True` |
| `035_multiple_objects_r03_p043_a_car_and_a_motorcycle` | `20.953188` | `19.241339` | `1.711849` | `1.088967` | `8.1699` | `True` |
| `036_human_action_r00_p012_a_person_is_skateboarding` | `20.953181` | `19.237382` | `1.715799` | `1.089191` | `8.1887` | `True` |
| `037_human_action_r01_p044_a_person_is_planting_trees` | `20.953188` | `19.236221` | `1.716967` | `1.089257` | `8.1943` | `True` |
| `038_human_action_r02_p045_a_person_is_sharpening_knives` | `20.953337` | `19.314131` | `1.639206` | `1.084871` | `7.8231` | `True` |
| `039_human_action_r03_p048_a_person_is_hula_hooping` | `20.953174` | `19.223637` | `1.729537` | `1.089969` | `8.2543` | `True` |
| `040_color_r00_p005_a_purple_bicycle` | `20.953069` | `19.240626` | `1.712443` | `1.089001` | `8.1728` | `True` |
| `041_color_r01_p033_a_blue_umbrella` | `20.953062` | `19.139307` | `1.813755` | `1.094766` | `8.6563` | `True` |
| `042_color_r02_p058_a_red_chair` | `20.953034` | `19.248359` | `1.704675` | `1.088562` | `8.1357` | `True` |
| `043_color_r03_p077_a_green_vase` | `20.953041` | `19.143487` | `1.809554` | `1.094526` | `8.6362` | `True` |
| `044_spatial_relationship_r00_p010_a_bird_on_the_left_of_a_cat_front_view` | `20.953712` | `19.229812` | `1.7239` | `1.089647` | `8.2272` | `True` |
| `045_spatial_relationship_r01_p015_a_cow_on_the_right_of_an_elephant_front_view` | `20.953754` | `19.207504` | `1.74625` | `1.090915` | `8.3338` | `True` |
| `046_spatial_relationship_r02_p048_a_train_on_the_right_of_a_boat_front_view` | `20.953733` | `19.203575` | `1.750158` | `1.091137` | `8.3525` | `True` |
| `047_spatial_relationship_r03_p068_a_pizza_on_the_top_of_a_donut_front_view` | `20.953726` | `19.157555` | `1.796171` | `1.093758` | `8.5721` | `True` |
| `048_scene_r00_p037_golf_course` | `20.953034` | `19.049102` | `1.903932` | `1.099949` | `9.0867` | `True` |
| `049_scene_r01_p070_sky` | `20.952978` | `18.947163` | `2.005815` | `1.105864` | `9.5729` | `True` |
| `050_scene_r02_p079_train_railway` | `20.953048` | `19.17579` | `1.777258` | `1.092682` | `8.4821` | `True` |
| `051_scene_r03_p080_train_station_platform` | `20.953111` | `19.156069` | `1.797042` | `1.093811` | `8.5765` | `True` |
| `052_temporal_style_r00_p024_a_shark_is_swimming_in_the_ocean_pan_right` | `20.953634` | `18.67452` | `2.279114` | `1.122044` | `10.8769` | `True` |
| `053_temporal_style_r01_p046_a_cute_happy_corgi_playing_in_park_sunset_tilt_down` | `20.953761` | `19.165886` | `1.787875` | `1.093284` | `8.5325` | `True` |
| `054_temporal_style_r02_p073_a_couple_in_formal_evening_wear_going_home_get_caught_in_a_heavy_downpour_with_u` | `20.954092` | `19.139664` | `1.814428` | `1.094799` | `8.6591` | `True` |
| `055_temporal_style_r03_p090_snow_rocky_mountains_peaks_canyon_snow_blanketed_rocky_mountains_surround_and_sh` | `20.95422` | `19.086813` | `1.867407` | `1.097838` | `8.9118` | `True` |
| `056_appearance_style_r00_p005_a_beautiful_coastal_beach_in_spring_waves_lapping_on_sand_in_cyberpunk_style` | `20.954078` | `19.075789` | `1.878289` | `1.098465` | `8.9638` | `True` |
| `057_appearance_style_r01_p008_a_beautiful_coastal_beach_in_spring_waves_lapping_on_sand_surrealism_style` | `20.954064` | `18.965461` | `1.988603` | `1.104854` | `9.4903` | `True` |
| `058_appearance_style_r02_p029_a_panda_drinking_coffee_in_a_cafe_in_paris_by_hokusai_in_the_style_of_ukiyo` | `20.954071` | `19.208306` | `1.745765` | `1.090886` | `8.3314` | `True` |
| `059_appearance_style_r03_p084_snow_rocky_mountains_peaks_canyon_snow_blanketed_rocky_mountains_surround_and_sh` | `20.954234` | `19.091608` | `1.862626` | `1.097563` | `8.889` | `True` |
| `060_overall_consistency_r00_p010_fireworks` | `20.953118` | `19.052537` | `1.900581` | `1.099755` | `9.0706` | `True` |
| `061_overall_consistency_r01_p012_flying_through_fantasy_landscapes` | `20.953542` | `19.179482` | `1.77406` | `1.092498` | `8.4666` | `True` |
| `062_overall_consistency_r02_p029_campfire_at_night_in_a_snowy_forest_with_starry_sky_in_the_background` | `20.95405` | `19.066068` | `1.887982` | `1.099023` | `9.0101` | `True` |
| `063_overall_consistency_r03_p037_an_astronaut_is_riding_a_horse_in_the_space_in_a_photorealistic_style` | `20.95405` | `18.975043` | `1.979007` | `1.104295` | `9.4445` | `True` |
