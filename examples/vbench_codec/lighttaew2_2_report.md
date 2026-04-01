# Wan2.2 LightTAE Sample Report

This report compares `lighttaew2_2` MP4 outputs against the official `Wan2.2_VAE` MP4 baseline for the 10 curated sample prompts already committed in `examples/vbench_codec/video_samples/`.

## Summary

- sample_count: `10`
- mean_official_decode_sec: `16.8667`
- mean_lighttae_decode_sec: `0.1542`
- mean_speedup_vs_official: `110.5952x`
- mean_raw_frame_psnr_db: `30.7227`
- mean_mp4_psnr_db: `34.7777`
- mean_mp4_ssim: `0.959747`

## Per-sample

| stem | official_decode_sec | lighttae_decode_sec | speedup | raw_psnr_db | mp4_psnr_db | mp4_ssim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `000_subject_consistency_r00_p003_a_person_eating_a_burger` | `17.2300` | `0.2119` | `81.3048x` | `32.0155` | `36.3307` | `0.956708` |
| `020_aesthetic_quality_r00_p028_origami_dancers_in_white_paper_3d_render_on_white_background_studio_shot_dancing` | `16.8311` | `0.1448` | `116.2087x` | `27.4161` | `30.5854` | `0.962842` |
| `026_imaging_quality_r02_p035_busy_freeway_at_night` | `16.8282` | `0.1480` | `113.6986x` | `27.9875` | `32.6217` | `0.963241` |
| `032_multiple_objects_r00_p011_a_couch_and_a_potted_plant` | `16.8260` | `0.1480` | `113.7251x` | `30.9425` | `34.3944` | `0.939787` |
| `038_human_action_r02_p045_a_person_is_sharpening_knives` | `16.8258` | `0.1473` | `114.1891x` | `33.4673` | `37.0233` | `0.966193` |
| `043_color_r03_p077_a_green_vase` | `16.8254` | `0.1476` | `113.9954x` | `34.7526` | `38.4253` | `0.973100` |
| `047_spatial_relationship_r03_p068_a_pizza_on_the_top_of_a_donut_front_view` | `16.8252` | `0.1489` | `113.0093x` | `34.1658` | `38.8794` | `0.979304` |
| `052_temporal_style_r00_p024_a_shark_is_swimming_in_the_ocean_pan_right` | `16.8251` | `0.1474` | `114.1101x` | `33.1693` | `37.6776` | `0.978715` |
| `058_appearance_style_r02_p029_a_panda_drinking_coffee_in_a_cafe_in_paris_by_hokusai_in_the_style_of_ukiyo` | `16.8254` | `0.1480` | `113.7168x` | `24.4506` | `27.9055` | `0.915422` |
| `062_overall_consistency_r02_p029_campfire_at_night_in_a_snowy_forest_with_starry_sky_in_the_background` | `16.8252` | `0.1502` | `111.9943x` | `28.8594` | `33.9338` | `0.962161` |
