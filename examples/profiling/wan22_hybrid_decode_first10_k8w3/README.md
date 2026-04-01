# Wan2.2 Hybrid Decode Pilot

This pilot evaluates a simple hybrid decode strategy for saved, uncompressed `Wan2.2 TI2V-5B` latents:

- `tae_only`: decode every latent step with `lighttaew2_2`
- `k8_w3_keyreplace`: decode every step with `lighttaew2_2`, but replace every 8th latent-step group with a short-window official `Wan2.2_VAE` decode over a 3-step causal window

The baseline for quality is the official full-sequence `Wan2.2_VAE` decode of the same uncompressed latent, followed by MP4 encoding. Metrics are reported in two domains:

- `raw-vs-raw`: uint8 frames from decoded videos before MP4
- `mp4-vs-baseline-mp4`: FFmpeg PSNR/SSIM after both sides are encoded to MP4

## Research Check

I did not find a paper that directly matches the exact `TAE + sparse VAE anchor + causal repair` idea.

Closest adjacent work:

- `Semantic-Aware Adaptive Video Streaming Using Latent Diffusion Models for Wireless Networks` (`arXiv:2502.05695`): uses I-frame latent compression plus P/B-frame refinement metadata inside a streaming system, but not a `TAE/VAE` hybrid decoder.
- `DLFR-VAE: Dynamic Latent Frame Rate VAE for Video Generation` (`arXiv:2502.11897`): adapts temporal latent rate for faster video generation, but still stays within the VAE family.
- `LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models` (`arXiv:2503.14325`): replaces heavy video VAEs with a lighter reconstruction VAE.
- `I^2VC: A Unified Framework for Intra- & Inter-frame Video Compression` (`arXiv:2405.14336`): relevant for the keyframe/inter-frame repair framing, though it is not a Wan/TAE decoder paper.

## 10-sample Result

Dataset:

- first 10 saved uncompressed latents from `/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents`
- output report: `hybrid_decode_report.json`

Aggregate outcome for `k8_w3_keyreplace` vs `tae_only`:

- `raw_frame_psnr_db` gain: mean `+0.2297 dB`, min `+0.0280 dB`, max `+0.3179 dB`
- `mp4_psnr_db` gain: mean `+0.2274 dB`, min `+0.0291 dB`, max `+0.3292 dB`
- all `10/10` samples improved in both raw and MP4 domains
- amortized decode cost: mean `34.87 ms/frame`, below the `24 fps` budget of `41.67 ms/frame`
- stricter serial playback model: `0/10` samples can sustain playback with zero prefetch
- required initial prebuffer in the serial model: mean `5.9` frames, range `4-9` frames

Anchor quality for the sparse VAE groups themselves:

- anchor raw PSNR vs full VAE: mean `29.33 dB`
- anchor raw PSNR minimum over key groups: mean `28.41 dB`
- each sample had `1` exact-match key group at the first latent step

## Interpretation

This first pilot says the idea is viable enough to keep pushing:

- sparse `VAE` anchors can consistently pull `TAE` quality upward without re-running denoising
- the current `k8_w3_keyreplace` strategy clears the average throughput budget on all 10 tested clips
- but it is not a zero-buffer stream; a practical player still needs a small prebuffer or asynchronous/background key decode

The current pilot is intentionally simple:

- it uses `keyreplace`, not a learned repair model
- it uses short-window VAE anchors, not a persistent long-cache sparse VAE stream
- it does not yet do motion-aware or content-aware correction between key groups

## Recommended Next Steps

- Replace the naive residual path with a causal low-frequency or tile-wise affine correction.
- Add adaptive key scheduling based on TAE error or temporal complexity instead of a fixed interval.
- Explore a persistent sparse-VAE stream so key anchors are closer to full-VAE behavior than cold-start window decodes.
- Validate with a true async pipeline model if the end goal is zero-stall playback on a single GPU.
