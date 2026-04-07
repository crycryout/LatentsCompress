# Channel Sensitivity Experiment

Sample: `latent`

## Scheme Comparison

| Scheme | Archive MB | vs `.pt` | Latent MAE | Raw Frame PSNR | MP4 PSNR | MP4 SSIM |
|---|---:|---:|---:|---:|---:|---:|
| uniform_inter_q8 | 4.587 | 4.57x | 0.0064 | 54.883 dB | 47.230 dB | 0.989908 |
| uniform_inter_q6 | 3.555 | 5.89x | 0.0261 | 47.416 dB | 46.049 dB | 0.988686 |
| mixed_864 | 3.483 | 6.02x | 0.0417 | 44.756 dB | 45.213 dB | 0.987539 |

## Most Sensitive Channels

| Rank | Channel | Raw Frame PSNR After Single-Channel Q4 |
|---|---:|---:|
| 1 | 21 | 44.911 dB |
| 2 | 39 | 46.153 dB |
| 3 | 37 | 46.581 dB |
| 4 | 11 | 48.238 dB |
| 5 | 41 | 48.560 dB |
| 6 | 1 | 48.852 dB |
| 7 | 22 | 49.293 dB |
| 8 | 30 | 49.466 dB |
| 9 | 3 | 49.488 dB |
| 10 | 9 | 49.847 dB |
