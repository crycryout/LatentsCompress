# Wan2.2 64 Latents Lossless OpenZL Report

This report compresses the original Wan2.2 latent `.pt` bytes directly with lossless `openzl` generic serial compression, and compares the resulting container size against the previously measured lossless `zstd` baseline.

## Summary

- sample_count: `64`
- total_original_mb: `1341.023296`
- total_openzl_mb: `1224.839651`
- total_zstd_mb: `1224.206334`
- total_saved_vs_original_mb: `116.183645`
- total_delta_vs_zstd_mb: `0.633317`
- total_ratio_original_to_openzl: `1.094856`
- mean_original_mb: `20.953489`
- mean_openzl_mb: `19.13812`
- mean_zstd_mb: `19.128224`
- mean_delta_vs_zstd_mb: `0.009896`
- count_openzl_smaller_than_zstd: `16`
- count_openzl_larger_than_zstd: `48`
- count_openzl_equal_to_zstd: `0`
- all_verified_lossless: `True`

## Rows

| name | original_mb | openzl_mb | zstd_mb | delta_vs_zstd_mb | saved_vs_original_mb | ratio_original_to_openzl | verified_lossless |
|---|---:|---:|---:|---:|---:|---:|---:|
| `000_subject_consistency_r00_p003_a_person_eating_a_burger` | `20.953351` | `19.192452` | `19.196455` | `-0.004003` | `1.760899` | `1.09175` | `True` |
| `001_subject_consistency_r01_p014_a_car_accelerating_to_gain_speed` | `20.953471` | `19.252123` | `19.248919` | `0.003204` | `1.701348` | `1.088372` | `True` |
| `002_subject_consistency_r02_p031_a_truck_anchored_in_a_tranquil_bay` | `20.953613` | `18.984823` | `18.946847` | `0.037976` | `1.96879` | `1.103703` | `True` |
| `003_subject_consistency_r03_p035_a_boat_sailing_smoothly_on_a_calm_lake` | `20.953705` | `19.080351` | `19.056187` | `0.024164` | `1.873354` | `1.098182` | `True` |
| `004_background_consistency_r00_p013_bridge` | `20.953118` | `19.046269` | `19.015279` | `0.03099` | `1.906849` | `1.100117` | `True` |
| `005_background_consistency_r01_p017_campus` | `20.953118` | `19.156024` | `19.133089` | `0.022935` | `1.797094` | `1.093814` | `True` |
| `006_background_consistency_r02_p028_downtown` | `20.953132` | `19.151372` | `19.140739` | `0.010633` | `1.80176` | `1.09408` | `True` |
| `007_background_consistency_r03_p069_ski_slope` | `20.953139` | `19.097315` | `19.06913` | `0.028185` | `1.855824` | `1.097177` | `True` |
| `008_temporal_flickering_r00_p003_a_tranquil_tableau_of_alley` | `20.953436` | `19.106864` | `19.09579` | `0.011074` | `1.846572` | `1.096644` | `True` |
| `009_temporal_flickering_r01_p004_a_tranquil_tableau_of_bar` | `20.953358` | `19.144557` | `19.13838` | `0.006177` | `1.808801` | `1.094481` | `True` |
| `010_temporal_flickering_r02_p011_a_tranquil_tableau_of_house` | `20.953436` | `19.095883` | `19.074892` | `0.020991` | `1.857553` | `1.097275` | `True` |
| `011_temporal_flickering_r03_p054_a_tranquil_tableau_of_in_the_heart_of_plaka_the_neoclassical_architecture_of_the` | `20.954191` | `19.090601` | `19.074794` | `0.015807` | `1.86359` | `1.097618` | `True` |
| `012_motion_smoothness_r00_p011_a_car_stuck_in_traffic_during_rush_hour` | `20.953634` | `19.207137` | `19.205602` | `0.001535` | `1.746497` | `1.09093` | `True` |
| `013_motion_smoothness_r01_p027_a_train_speeding_down_the_tracks` | `20.953457` | `19.239311` | `19.229296` | `0.010015` | `1.714146` | `1.089096` | `True` |
| `014_motion_smoothness_r02_p029_a_train_accelerating_to_gain_speed` | `20.953471` | `19.270374` | `19.272225` | `-0.001851` | `1.683097` | `1.087341` | `True` |
| `015_motion_smoothness_r03_p064_a_bear_climbing_a_tree` | `20.953195` | `19.245059` | `19.249323` | `-0.004264` | `1.708136` | `1.088757` | `True` |
| `016_dynamic_degree_r00_p003_a_person_eating_a_burger` | `20.953188` | `19.192351` | `19.195698` | `-0.003347` | `1.760837` | `1.091747` | `True` |
| `017_dynamic_degree_r01_p025_a_bus_stuck_in_traffic_during_rush_hour` | `20.953613` | `19.223248` | `19.219874` | `0.003374` | `1.730365` | `1.090014` | `True` |
| `018_dynamic_degree_r02_p069_a_giraffe_bending_down_to_drink_water_from_a_river` | `20.953754` | `19.182935` | `19.163242` | `0.019693` | `1.770819` | `1.092312` | `True` |
| `019_dynamic_degree_r03_p071_a_giraffe_running_to_join_a_herd_of_its_kind` | `20.953712` | `19.074254` | `19.037801` | `0.036453` | `1.879458` | `1.098534` | `True` |
| `020_aesthetic_quality_r00_p028_origami_dancers_in_white_paper_3d_render_on_white_background_studio_shot_dancing` | `20.954177` | `19.00382` | `19.002682` | `0.001138` | `1.950357` | `1.10263` | `True` |
| `021_aesthetic_quality_r01_p053_a_jellyfish_floating_through_the_ocean_with_bioluminescent_tentacles` | `20.954029` | `18.675305` | `18.648128` | `0.027177` | `2.278724` | `1.122018` | `True` |
| `022_aesthetic_quality_r02_p057_a_steam_train_moving_on_a_mountainside` | `20.953627` | `19.231604` | `19.220958` | `0.010646` | `1.722023` | `1.089541` | `True` |
| `023_aesthetic_quality_r03_p075_a_happy_fuzzy_panda_playing_guitar_nearby_a_campfire_snow_mountain_in_the_backgr` | `20.954113` | `19.268386` | `19.270686` | `-0.0023` | `1.685727` | `1.087487` | `True` |
| `024_imaging_quality_r00_p000_close_up_of_grapes_on_a_rotating_table` | `20.953613` | `19.271063` | `19.273631` | `-0.002568` | `1.68255` | `1.08731` | `True` |
| `025_imaging_quality_r01_p020_a_shark_is_swimming_in_the_ocean` | `20.953443` | `18.634337` | `18.629673` | `0.004664` | `2.319106` | `1.124453` | `True` |
| `026_imaging_quality_r02_p035_busy_freeway_at_night` | `20.953174` | `18.959889` | `18.933355` | `0.026534` | `1.993285` | `1.105132` | `True` |
| `027_imaging_quality_r03_p089_hyper_realistic_spaceship_landing_on_mars` | `20.953634` | `19.14842` | `19.149625` | `-0.001205` | `1.805214` | `1.094275` | `True` |
| `028_object_class_r00_p019_a_cow` | `20.953041` | `19.179334` | `19.167705` | `0.011629` | `1.773707` | `1.09248` | `True` |
| `029_object_class_r01_p035_a_baseball_glove` | `20.953118` | `19.23984` | `19.239508` | `0.000332` | `1.713278` | `1.089048` | `True` |
| `030_object_class_r02_p043_a_knife` | `20.953055` | `19.227965` | `19.229142` | `-0.001177` | `1.72509` | `1.089718` | `True` |
| `031_object_class_r03_p054_a_donut` | `20.953055` | `19.192086` | `19.186297` | `0.005789` | `1.760969` | `1.091755` | `True` |
| `032_multiple_objects_r00_p011_a_couch_and_a_potted_plant` | `20.953344` | `19.229126` | `19.229808` | `-0.000682` | `1.724218` | `1.089667` | `True` |
| `033_multiple_objects_r01_p013_a_tv_and_a_laptop` | `20.953153` | `19.219888` | `19.221691` | `-0.001803` | `1.733265` | `1.090181` | `True` |
| `034_multiple_objects_r02_p027_a_teddy_bear_and_a_frisbee` | `20.953344` | `19.212046` | `19.207647` | `0.004399` | `1.741298` | `1.090636` | `True` |
| `035_multiple_objects_r03_p043_a_car_and_a_motorcycle` | `20.953188` | `19.243629` | `19.241339` | `0.00229` | `1.709559` | `1.088838` | `True` |
| `036_human_action_r00_p012_a_person_is_skateboarding` | `20.953181` | `19.242333` | `19.237382` | `0.004951` | `1.710848` | `1.088911` | `True` |
| `037_human_action_r01_p044_a_person_is_planting_trees` | `20.953188` | `19.23452` | `19.236221` | `-0.001701` | `1.718668` | `1.089353` | `True` |
| `038_human_action_r02_p045_a_person_is_sharpening_knives` | `20.953337` | `19.3084` | `19.314131` | `-0.005731` | `1.644937` | `1.085193` | `True` |
| `039_human_action_r03_p048_a_person_is_hula_hooping` | `20.953174` | `19.231328` | `19.223637` | `0.007691` | `1.721846` | `1.089533` | `True` |
| `040_color_r00_p005_a_purple_bicycle` | `20.953069` | `19.242142` | `19.240626` | `0.001516` | `1.710927` | `1.088916` | `True` |
| `041_color_r01_p033_a_blue_umbrella` | `20.953062` | `19.15759` | `19.139307` | `0.018283` | `1.795472` | `1.093721` | `True` |
| `042_color_r02_p058_a_red_chair` | `20.953034` | `19.246897` | `19.248359` | `-0.001462` | `1.706137` | `1.088645` | `True` |
| `043_color_r03_p077_a_green_vase` | `20.953041` | `19.147303` | `19.143487` | `0.003816` | `1.805738` | `1.094308` | `True` |
| `044_spatial_relationship_r00_p010_a_bird_on_the_left_of_a_cat_front_view` | `20.953712` | `19.231261` | `19.229812` | `0.001449` | `1.722451` | `1.089565` | `True` |
| `045_spatial_relationship_r01_p015_a_cow_on_the_right_of_an_elephant_front_view` | `20.953754` | `19.206861` | `19.207504` | `-0.000643` | `1.746893` | `1.090952` | `True` |
| `046_spatial_relationship_r02_p048_a_train_on_the_right_of_a_boat_front_view` | `20.953733` | `19.214237` | `19.203575` | `0.010662` | `1.739496` | `1.090532` | `True` |
| `047_spatial_relationship_r03_p068_a_pizza_on_the_top_of_a_donut_front_view` | `20.953726` | `19.154281` | `19.157555` | `-0.003274` | `1.799445` | `1.093945` | `True` |
| `048_scene_r00_p037_golf_course` | `20.953034` | `19.080097` | `19.049102` | `0.030995` | `1.872937` | `1.098162` | `True` |
| `049_scene_r01_p070_sky` | `20.952978` | `18.983457` | `18.947163` | `0.036294` | `1.969521` | `1.103749` | `True` |
| `050_scene_r02_p079_train_railway` | `20.953048` | `19.190907` | `19.17579` | `0.015117` | `1.762141` | `1.091822` | `True` |
| `051_scene_r03_p080_train_station_platform` | `20.953111` | `19.167587` | `19.156069` | `0.011518` | `1.785524` | `1.093153` | `True` |
| `052_temporal_style_r00_p024_a_shark_is_swimming_in_the_ocean_pan_right` | `20.953634` | `18.717815` | `18.67452` | `0.043295` | `2.235819` | `1.119449` | `True` |
| `053_temporal_style_r01_p046_a_cute_happy_corgi_playing_in_park_sunset_tilt_down` | `20.953761` | `19.185851` | `19.165886` | `0.019965` | `1.76791` | `1.092147` | `True` |
| `054_temporal_style_r02_p073_a_couple_in_formal_evening_wear_going_home_get_caught_in_a_heavy_downpour_with_u` | `20.954092` | `19.143834` | `19.139664` | `0.00417` | `1.810258` | `1.094561` | `True` |
| `055_temporal_style_r03_p090_snow_rocky_mountains_peaks_canyon_snow_blanketed_rocky_mountains_surround_and_sh` | `20.95422` | `19.097279` | `19.086813` | `0.010466` | `1.856941` | `1.097236` | `True` |
| `056_appearance_style_r00_p005_a_beautiful_coastal_beach_in_spring_waves_lapping_on_sand_in_cyberpunk_style` | `20.954078` | `19.089971` | `19.075789` | `0.014182` | `1.864107` | `1.097648` | `True` |
| `057_appearance_style_r01_p008_a_beautiful_coastal_beach_in_spring_waves_lapping_on_sand_surrealism_style` | `20.954064` | `18.985347` | `18.965461` | `0.019886` | `1.968717` | `1.103697` | `True` |
| `058_appearance_style_r02_p029_a_panda_drinking_coffee_in_a_cafe_in_paris_by_hokusai_in_the_style_of_ukiyo` | `20.954071` | `19.211916` | `19.208306` | `0.00361` | `1.742155` | `1.090681` | `True` |
| `059_appearance_style_r03_p084_snow_rocky_mountains_peaks_canyon_snow_blanketed_rocky_mountains_surround_and_sh` | `20.954234` | `19.091858` | `19.091608` | `0.00025` | `1.862376` | `1.097548` | `True` |
| `060_overall_consistency_r00_p010_fireworks` | `20.953118` | `19.063622` | `19.052537` | `0.011085` | `1.889496` | `1.099115` | `True` |
| `061_overall_consistency_r01_p012_flying_through_fantasy_landscapes` | `20.953542` | `19.195843` | `19.179482` | `0.016361` | `1.757699` | `1.091567` | `True` |
| `062_overall_consistency_r02_p029_campfire_at_night_in_a_snowy_forest_with_starry_sky_in_the_background` | `20.95405` | `19.06524` | `19.066068` | `-0.000828` | `1.88881` | `1.099071` | `True` |
| `063_overall_consistency_r03_p037_an_astronaut_is_riding_a_horse_in_the_space_in_a_photorealistic_style` | `20.95405` | `18.981833` | `18.975043` | `0.00679` | `1.972217` | `1.1039` | `True` |
