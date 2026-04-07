# Downstream-Loss-Aware Wan2.2 Experiment

Sample: `latent`

| Scheme | Archive MB | vs latent .pt | vs baseline MP4 | Latent MAE | Raw frame PSNR | MP4 PSNR | MP4 SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|
| intra_q4 | 1.652 | 12.69x | 1.76x | 0.1149 | 35.578 dB | 39.049 dB | 0.974857 |
| inter_q4_k8 | 1.731 | 12.10x | 1.68x | 0.1169 | 35.727 dB | 39.189 dB | 0.975190 |
| inter_q6_k8 | 3.555 | 5.89x | 0.82x | 0.0261 | 47.412 dB | 46.019 dB | 0.988632 |
| intra_q6 | 3.563 | 5.88x | 0.82x | 0.0259 | 47.223 dB | 45.996 dB | 0.988649 |
| inter_q8_k8 | 4.591 | 4.56x | 0.63x | 0.0064 | 54.879 dB | 47.190 dB | 0.989846 |
| intra_q8 | 4.841 | 4.33x | 0.60x | 0.0063 | 54.756 dB | 47.184 dB | 0.989848 |
| intra_fp16 | 8.388 | 2.50x | 0.35x | 0.0002 | 59.329 dB | 47.269 dB | 0.989907 |
| inter_fp16_k8 | 8.925 | 2.35x | 0.33x | 0.0001 | 59.708 dB | 47.347 dB | 0.990050 |
