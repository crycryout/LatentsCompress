# Wan2.2 Relation Side-Info V1

This proof-of-concept tests a new server/client architecture:

- server stores the original Wan2.2 latents
- server also computes compact `TAE -> VAE` relation side information
- client receives latents plus side information
- client only runs `TAE` decode locally, then applies the side information to approximate the full-VAE result

No client-side VAE decode is used in this setup.

## Schemes

- `tile_affine_f8_4x4_fp16`
  - very compact per-frame low-resolution tile-affine metadata
- `lowres_residual_f8_q8`
  - dense low-resolution residual field at `1/8` spatial scale, quantized to `q8`
- `lowres_residual_f4_q8`
  - denser low-resolution residual field at `1/4` spatial scale, quantized to `q8`

## 2-sample Result

Compared with `TAE-only`:

- `tile_affine_f8_4x4_fp16`
  - mean raw PSNR gain: about `+0.20 dB`
  - mean MP4 PSNR gain: about `+0.20 dB`
  - mean side-info size: about `0.018 MB`
  - mean side-info / latent ratio: about `0.09%`

- `lowres_residual_f8_q8`
  - mean raw PSNR gain: about `+2.27 dB`
  - mean MP4 PSNR gain: about `+2.00 dB`
  - mean side-info size: about `3.67 MB`
  - mean side-info / latent ratio: about `18.37%`

- `lowres_residual_f4_q8`
  - mean raw PSNR gain: about `+4.40 dB`
  - mean MP4 PSNR gain: about `+4.00 dB`
  - mean side-info size: about `12.77 MB`
  - mean side-info / latent ratio: about `63.92%`

## Interpretation

This strongly suggests the earlier low gains were not only a repair-algorithm issue in the old sparse-key setup.

If the server is allowed to send richer per-group relation information, quality can move by multiple dB.

But the trade-off is clear:

- tiny parametric metadata stays compact, but gains are small
- dense low-resolution residual side-info gives much larger gains
- as the residual field becomes stronger, the method starts to look more like a compact residual stream rather than “just a little metadata”

So the promising operating point is currently around:

- `lowres_residual_f8_q8`

It is clearly stronger than the old sparse-key heuristic route, but it is still expensive enough that it should be compared against simpler baselines such as low-res residual transmission or direct video residual coding.
