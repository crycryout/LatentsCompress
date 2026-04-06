# SkyReels V2 Async 720P 30s

This folder tracks the repo-local prompt and launcher used for a `SkyReels-V2-DF-14B-720P` run with:

- async diffusion forcing mode
- `720p`
- `24fps`
- training-aligned `30s` target via `737` frames
- VRAM reduction via `--offload` and a reduced `base_num_frames`

Recommended workflow:

1. Install the Python environment:
   - `bash scripts/install_skyreels_v2_env.sh`
2. If your local `SkyReels-V2` checkout does not have `flash-attn`, apply the fallback patch:
   - `git -C /root/SkyReels-V2 apply /root/LatentsCompress/examples/skyreels_generation/skyreels_v2_no_flash_attn.patch`
3. Start the job with `nohup` protection:
   - `bash scripts/launch_skyreels_v2_async_720p_30s_nohup.sh`
4. Watch progress:
   - `tail -f examples/skyreels_generation/logs/skyreels_v2_df_async_720p_30s_high_motion*.log`

Default memory-saving settings:

- `--offload`
- `BASE_NUM_FRAMES=57`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`
- `TOKENIZERS_PARALLELISM=false`

More aggressive fallback if the default still OOMs:

- `BASE_NUM_FRAMES=37 bash scripts/launch_skyreels_v2_async_720p_30s_nohup.sh`

Validated local run:

- output: `720p`, `24fps`, `737` frames, about `30.708s`
- mode: async long-video generation
- stable single-GPU setting on this machine: `BASE_NUM_FRAMES=57`

Prompt files:

- `async_720p_30s_duration_prompt.txt`: keeps the model focused on sustained action across the full `30s`.
- `async_720p_30s_high_motion_prompt.txt`: emphasizes aggressive subject and camera dynamics.
- `async_720p_30s_full_prompt.txt`: the combined prompt used by the launcher.
