# SkyReels Latent Visualization Outputs

This folder collects the visualization artifacts added on 2026-04-07 for the three deduplicated SkyReels long-video latent samples:

- `wingsuit`
- `neon`
- `avalanche`

## Included analyses

- `frame_vs_latent_offset_*.json`
  - Direct time-axis offset statistics comparing raw decoded video frames and latent steps.
- `frame_vs_latent_locality_metrics*.json`
  - Locality / concentration metrics showing that frame differences are spatially sparse while latent differences are spatially dense.
- `heatmaps_neon_t140/`
  - A hand-picked example with still images and heatmaps for:
    - video frame `140 -> 141`
    - video frame `140 -> 144`
    - latent step `35 -> 36`

## Video heatmap outputs

- `heatmap_videos/<sample>/<sample>_frame_diff_t1_heatmap.mp4`
  - Pixel-space frame-difference heatmap video.
- `heatmap_videos/<sample>/<sample>_latent_diff_t1_heatmap.mp4`
  - Latent absolute-difference heatmap video.
- `heatmap_videos/<sample>/<sample>_preview_triptych.png`
  - A quick static preview comparing the two modalities.

## Latent visualization outputs

- `latent_visualization_videos/<sample>/<sample>_latent_grid.mp4`
  - 16-channel latent grid visualization with signed colormap.
- `latent_visualization_videos/<sample>/<sample>_latent_heatmap.mp4`
  - Absolute-activation heatmap collapsed over channels.
- `latent_visualization_videos/<sample>/<sample>_latent_pca_rgb.mp4`
  - PCA projection of the 16-channel latent tensor into pseudo-RGB.
- `latent_visualization_videos/<sample>/<sample>_preview_triptych.png`
  - A side-by-side still preview of the three modes.

## Relative latent-difference outputs

- `latent_relative_diff_videos/<sample>/<sample>_latent_relative_diff.mp4`
  - Per-step latent difference map normalized by its own framewise mean:
    - blue = below that step's mean change
    - white = near that step's mean change
    - red = above that step's mean change
- `latent_relative_diff_videos/<sample>/<sample>_latent_relative_diff_preview.png`
  - A single-frame preview with legend.

## Notes

- These assets intentionally keep the final MP4s, previews, metadata, and index files in Git.
- Large intermediate PNG frame directories used during rendering are excluded by `.gitignore`.
