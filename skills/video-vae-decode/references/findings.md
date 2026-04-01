# Findings

## Wan

- Stateful time streaming decode was implemented successfully.
- With `latent_group_size=1`, Wan outputs:
  - first group: `1` frame
  - later groups: `4` frames each
- Total frame count matches full decode, but stream decode is not numerically identical to full decode.
- Stream decode was slower than full decode by roughly `5%` to `7%` in eager mode.
- `torch.compile` helped full decode more than streaming decode.

## Open-Sora 2.0

- Native temporal tiled decode was compared against full decode.
- In the tested `hunyuan_vae` path, `overlap=0` and `overlap=0.25` both matched full decode exactly in the tested cases.
- This means Open-Sora's native temporal tiled path behaved like an official exact decode path in those experiments.

## HunyuanVideo 1.5

- Official GPU VAE decode of a saved `121`-frame latent OOMed even after:
  - fp16 decode
  - spatial tiling
  - smaller tile sizes
  - memory-efficient context
- The root cause was not another process on the GPU.
- The larger issue was that the official VAE path does not support temporal tiling, so time remains a full-length activation burden.
- An unofficial temporal chunk decode prototype succeeded in decoding the saved latent to MP4 on GPU.
- Example output is under `examples/streaming/hunyuan/`.

## Practical takeaway

If official Hunyuan decode OOMs on long videos, the real decision is usually:
- accept CPU decode
- reduce clip length
- or use a non-official temporal chunk workaround

Do not assume that spatial tiling alone is enough for long-video GPU decode.
