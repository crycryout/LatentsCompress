# Wan2.2 64 Latents Lossless Zstd Then OpenZL Report

This report first compresses each Wan2.2 latent `.pt` file with lossless `zstd`, then compresses the resulting zstd byte stream again with lossless `openzl` generic serial compression.

Properties:

- stage 1: lossless `zstd` on original `.pt` bytes
- stage 2: lossless `openzl` on the `zstd` payload bytes
- verification: `openzl -> zstd -> original` round-trip checked for every sample

## Summary

- sample_count: `64`
- zstd_level: `19`
- total_original_mb: `1341.023296`
- total_zstd_mb: `1224.139234`
- total_zstd_then_openzl_mb: `1224.169846`
- total_saved_vs_original_mb: `116.85345`
- total_saved_vs_zstd_mb: `-0.030612`
- total_ratio_original_to_zstd_then_openzl: `1.095455`
- mean_original_mb: `20.953489`
- mean_zstd_mb: `19.127176`
- mean_zstd_then_openzl_mb: `19.127654`
- mean_delta_vs_zstd_mb: `0.000478`
- all_verified_lossless: `True`

## Rows

| name | original_mb | zstd_mb | zstd_then_openzl_mb | delta_vs_zstd_mb | saved_vs_original_mb | ratio_original_to_final | verified_lossless |
|---|---:|---:|---:|---:|---:|---:|---:|
| `000_subject_consistency_r00_p003_a_person_eating_a_burger` | `20.953351` | `19.191728` | `19.192208` | `0.00048` | `1.761143` | `1.091763` | `True` |
| `001_subject_consistency_r01_p014_a_car_accelerating_to_gain_speed` | `20.953471` | `19.252407` | `19.252887` | `0.00048` | `1.700584` | `1.088329` | `True` |
| `002_subject_consistency_r02_p031_a_truck_anchored_in_a_tranquil_bay` | `20.953613` | `18.947021` | `18.947495` | `0.000474` | `2.006118` | `1.105878` | `True` |
| `003_subject_consistency_r03_p035_a_boat_sailing_smoothly_on_a_calm_lake` | `20.953705` | `19.054364` | `19.054841` | `0.000477` | `1.898864` | `1.099653` | `True` |
| `004_background_consistency_r00_p013_bridge` | `20.953118` | `19.015413` | `19.01589` | `0.000477` | `1.937228` | `1.101874` | `True` |
| `005_background_consistency_r01_p017_campus` | `20.953118` | `19.132671` | `19.133148` | `0.000477` | `1.81997` | `1.095121` | `True` |
| `006_background_consistency_r02_p028_downtown` | `20.953132` | `19.151721` | `19.152201` | `0.00048` | `1.800931` | `1.094033` | `True` |
| `007_background_consistency_r03_p069_ski_slope` | `20.953139` | `19.068285` | `19.068762` | `0.000477` | `1.884377` | `1.09882` | `True` |
| `008_temporal_flickering_r00_p003_a_tranquil_tableau_of_alley` | `20.953436` | `19.096228` | `19.096705` | `0.000477` | `1.856731` | `1.097228` | `True` |
| `009_temporal_flickering_r01_p004_a_tranquil_tableau_of_bar` | `20.953358` | `19.143582` | `19.144062` | `0.00048` | `1.809296` | `1.09451` | `True` |
| `010_temporal_flickering_r02_p011_a_tranquil_tableau_of_house` | `20.953436` | `19.073251` | `19.073728` | `0.000477` | `1.879708` | `1.09855` | `True` |
| `011_temporal_flickering_r03_p054_a_tranquil_tableau_of_in_the_heart_of_plaka_the_neoclassical_architecture_of_the` | `20.954191` | `19.075827` | `19.076304` | `0.000477` | `1.877887` | `1.098441` | `True` |
| `012_motion_smoothness_r00_p011_a_car_stuck_in_traffic_during_rush_hour` | `20.953634` | `19.204035` | `19.204515` | `0.00048` | `1.749119` | `1.091079` | `True` |
| `013_motion_smoothness_r01_p027_a_train_speeding_down_the_tracks` | `20.953457` | `19.234155` | `19.234635` | `0.00048` | `1.718822` | `1.089361` | `True` |
| `014_motion_smoothness_r02_p029_a_train_accelerating_to_gain_speed` | `20.953471` | `19.271783` | `19.272266` | `0.000483` | `1.681205` | `1.087234` | `True` |
| `015_motion_smoothness_r03_p064_a_bear_climbing_a_tree` | `20.953195` | `19.244866` | `19.245346` | `0.00048` | `1.707849` | `1.088741` | `True` |
| `016_dynamic_degree_r00_p003_a_person_eating_a_burger` | `20.953188` | `19.192273` | `19.192753` | `0.00048` | `1.760435` | `1.091724` | `True` |
| `017_dynamic_degree_r01_p025_a_bus_stuck_in_traffic_during_rush_hour` | `20.953613` | `19.218049` | `19.218529` | `0.00048` | `1.735084` | `1.090282` | `True` |
| `018_dynamic_degree_r02_p069_a_giraffe_bending_down_to_drink_water_from_a_river` | `20.953754` | `19.159625` | `19.160105` | `0.00048` | `1.793649` | `1.093614` | `True` |
| `019_dynamic_degree_r03_p071_a_giraffe_running_to_join_a_herd_of_its_kind` | `20.953712` | `19.037187` | `19.037664` | `0.000477` | `1.916048` | `1.100645` | `True` |
| `020_aesthetic_quality_r00_p028_origami_dancers_in_white_paper_3d_render_on_white_background_studio_shot_dancing` | `20.954177` | `18.999492` | `18.999966` | `0.000474` | `1.954211` | `1.102853` | `True` |
| `021_aesthetic_quality_r01_p053_a_jellyfish_floating_through_the_ocean_with_bioluminescent_tentacles` | `20.954029` | `18.645538` | `18.646006` | `0.000468` | `2.308023` | `1.123781` | `True` |
| `022_aesthetic_quality_r02_p057_a_steam_train_moving_on_a_mountainside` | `20.953627` | `19.221951` | `19.222431` | `0.00048` | `1.731196` | `1.090061` | `True` |
| `023_aesthetic_quality_r03_p075_a_happy_fuzzy_panda_playing_guitar_nearby_a_campfire_snow_mountain_in_the_backgr` | `20.954113` | `19.269715` | `19.270198` | `0.000483` | `1.683915` | `1.087384` | `True` |
| `024_imaging_quality_r00_p000_close_up_of_grapes_on_a_rotating_table` | `20.953613` | `19.270844` | `19.271327` | `0.000483` | `1.682286` | `1.087295` | `True` |
| `025_imaging_quality_r01_p020_a_shark_is_swimming_in_the_ocean` | `20.953443` | `18.61022` | `18.610685` | `0.000465` | `2.342758` | `1.125882` | `True` |
| `026_imaging_quality_r02_p035_busy_freeway_at_night` | `20.953174` | `18.932082` | `18.932556` | `0.000474` | `2.020618` | `1.106727` | `True` |
| `027_imaging_quality_r03_p089_hyper_realistic_spaceship_landing_on_mars` | `20.953634` | `19.149325` | `19.149805` | `0.00048` | `1.803829` | `1.094196` | `True` |
| `028_object_class_r00_p019_a_cow` | `20.953041` | `19.165474` | `19.165954` | `0.00048` | `1.787087` | `1.093243` | `True` |
| `029_object_class_r01_p035_a_baseball_glove` | `20.953118` | `19.239241` | `19.239721` | `0.00048` | `1.713397` | `1.089055` | `True` |
| `030_object_class_r02_p043_a_knife` | `20.953055` | `19.229689` | `19.230169` | `0.00048` | `1.722886` | `1.089593` | `True` |
| `031_object_class_r03_p054_a_donut` | `20.953055` | `19.188879` | `19.189359` | `0.00048` | `1.763696` | `1.09191` | `True` |
| `032_multiple_objects_r00_p011_a_couch_and_a_potted_plant` | `20.953344` | `19.230093` | `19.230573` | `0.00048` | `1.722771` | `1.089585` | `True` |
| `033_multiple_objects_r01_p013_a_tv_and_a_laptop` | `20.953153` | `19.221455` | `19.221935` | `0.00048` | `1.731218` | `1.090065` | `True` |
| `034_multiple_objects_r02_p027_a_teddy_bear_and_a_frisbee` | `20.953344` | `19.205562` | `19.206042` | `0.00048` | `1.747302` | `1.090977` | `True` |
| `035_multiple_objects_r03_p043_a_car_and_a_motorcycle` | `20.953188` | `19.24074` | `19.24122` | `0.00048` | `1.711968` | `1.088974` | `True` |
| `036_human_action_r00_p012_a_person_is_skateboarding` | `20.953181` | `19.231792` | `19.232272` | `0.00048` | `1.720909` | `1.08948` | `True` |
| `037_human_action_r01_p044_a_person_is_planting_trees` | `20.953188` | `19.234076` | `19.234556` | `0.00048` | `1.718632` | `1.089351` | `True` |
| `038_human_action_r02_p045_a_person_is_sharpening_knives` | `20.953337` | `19.314472` | `19.314955` | `0.000483` | `1.638382` | `1.084825` | `True` |
| `039_human_action_r03_p048_a_person_is_hula_hooping` | `20.953174` | `19.212564` | `19.213044` | `0.00048` | `1.74013` | `1.09057` | `True` |
| `040_color_r00_p005_a_purple_bicycle` | `20.953069` | `19.238332` | `19.238812` | `0.00048` | `1.714257` | `1.089104` | `True` |
| `041_color_r01_p033_a_blue_umbrella` | `20.953062` | `19.138406` | `19.138886` | `0.00048` | `1.814176` | `1.09479` | `True` |
| `042_color_r02_p058_a_red_chair` | `20.953034` | `19.247158` | `19.247638` | `0.00048` | `1.705396` | `1.088603` | `True` |
| `043_color_r03_p077_a_green_vase` | `20.953041` | `19.142483` | `19.142963` | `0.00048` | `1.810078` | `1.094556` | `True` |
| `044_spatial_relationship_r00_p010_a_bird_on_the_left_of_a_cat_front_view` | `20.953712` | `19.228674` | `19.229154` | `0.00048` | `1.724558` | `1.089685` | `True` |
| `045_spatial_relationship_r01_p015_a_cow_on_the_right_of_an_elephant_front_view` | `20.953754` | `19.205867` | `19.206347` | `0.00048` | `1.747407` | `1.090981` | `True` |
| `046_spatial_relationship_r02_p048_a_train_on_the_right_of_a_boat_front_view` | `20.953733` | `19.203674` | `19.204154` | `0.00048` | `1.749579` | `1.091104` | `True` |
| `047_spatial_relationship_r03_p068_a_pizza_on_the_top_of_a_donut_front_view` | `20.953726` | `19.154367` | `19.154847` | `0.00048` | `1.798879` | `1.093912` | `True` |
| `048_scene_r00_p037_golf_course` | `20.953034` | `19.050442` | `19.050919` | `0.000477` | `1.902115` | `1.099844` | `True` |
| `049_scene_r01_p070_sky` | `20.952978` | `18.947586` | `18.94806` | `0.000474` | `2.004918` | `1.105811` | `True` |
| `050_scene_r02_p079_train_railway` | `20.953048` | `19.174515` | `19.174995` | `0.00048` | `1.778053` | `1.092728` | `True` |
| `051_scene_r03_p080_train_station_platform` | `20.953111` | `19.154404` | `19.154884` | `0.00048` | `1.798227` | `1.093878` | `True` |
| `052_temporal_style_r00_p024_a_shark_is_swimming_in_the_ocean_pan_right` | `20.953634` | `18.673711` | `18.674179` | `0.000468` | `2.279455` | `1.122065` | `True` |
| `053_temporal_style_r01_p046_a_cute_happy_corgi_playing_in_park_sunset_tilt_down` | `20.953761` | `19.165832` | `19.166312` | `0.00048` | `1.787449` | `1.09326` | `True` |
| `054_temporal_style_r02_p073_a_couple_in_formal_evening_wear_going_home_get_caught_in_a_heavy_downpour_with_u` | `20.954092` | `19.135118` | `19.135595` | `0.000477` | `1.818497` | `1.095032` | `True` |
| `055_temporal_style_r03_p090_snow_rocky_mountains_peaks_canyon_snow_blanketed_rocky_mountains_surround_and_sh` | `20.95422` | `19.087679` | `19.088156` | `0.000477` | `1.866064` | `1.09776` | `True` |
| `056_appearance_style_r00_p005_a_beautiful_coastal_beach_in_spring_waves_lapping_on_sand_in_cyberpunk_style` | `20.954078` | `19.083355` | `19.083832` | `0.000477` | `1.870246` | `1.098002` | `True` |
| `057_appearance_style_r01_p008_a_beautiful_coastal_beach_in_spring_waves_lapping_on_sand_surrealism_style` | `20.954064` | `18.963642` | `18.964116` | `0.000474` | `1.989948` | `1.104932` | `True` |
| `058_appearance_style_r02_p029_a_panda_drinking_coffee_in_a_cafe_in_paris_by_hokusai_in_the_style_of_ukiyo` | `20.954071` | `19.208263` | `19.208743` | `0.00048` | `1.745328` | `1.090861` | `True` |
| `059_appearance_style_r03_p084_snow_rocky_mountains_peaks_canyon_snow_blanketed_rocky_mountains_surround_and_sh` | `20.954234` | `19.083646` | `19.084123` | `0.000477` | `1.870111` | `1.097993` | `True` |
| `060_overall_consistency_r00_p010_fireworks` | `20.953118` | `19.050551` | `19.051028` | `0.000477` | `1.90209` | `1.099842` | `True` |
| `061_overall_consistency_r01_p012_flying_through_fantasy_landscapes` | `20.953542` | `19.193767` | `19.194247` | `0.00048` | `1.759295` | `1.091657` | `True` |
| `062_overall_consistency_r02_p029_campfire_at_night_in_a_snowy_forest_with_starry_sky_in_the_background` | `20.95405` | `19.059677` | `19.060154` | `0.000477` | `1.893896` | `1.099364` | `True` |
| `063_overall_consistency_r03_p037_an_astronaut_is_riding_a_horse_in_the_space_in_a_photorealistic_style` | `20.95405` | `18.97041` | `18.970884` | `0.000474` | `1.983166` | `1.104537` | `True` |
