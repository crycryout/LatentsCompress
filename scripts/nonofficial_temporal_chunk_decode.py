import argparse
import sys
from pathlib import Path
from types import MethodType

import torch

sys.path.insert(0, "/root/HunyuanVideo-1.5")

from generate import save_video
from hyvideo.models.autoencoders import hunyuanvideo_15_vae


def _blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
    blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
    for x in range(blend_extent):
        b[:, :, x, :, :] = (
            a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent)
            + b[:, :, x, :, :] * (x / blend_extent)
        )
    return b


def _temporal_tiled_decode_enabled(self, z: torch.Tensor):
    _, _, t, _, _ = z.shape
    overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
    blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
    t_limit = self.tile_sample_min_tsize - blend_extent
    assert 0 < overlap_size < self.tile_latent_min_tsize

    row = []
    for i in range(0, t, overlap_size):
        tile = z[:, :, i : i + self.tile_latent_min_tsize + 1, :, :]
        if self.use_spatial_tiling and (
            tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size
        ):
            decoded = self.spatial_tiled_decode(tile)
        else:
            decoded = self.decoder(tile)
        if i > 0:
            decoded = decoded[:, :, 1:, :, :]
        row.append(decoded)

    result_row = []
    for i, tile in enumerate(row):
        if i > 0:
            tile = _blend_t(self, row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])
        else:
            result_row.append(tile[:, :, : t_limit + 1, :, :])
    return torch.cat(result_row, dim=-3)


def enable_nonofficial_temporal_tiling(vae, sample_tsize: int, overlap_factor: float):
    if sample_tsize % vae.ffactor_temporal != 0:
        raise ValueError(
            f"sample_tsize must be divisible by temporal compression factor {vae.ffactor_temporal}, got {sample_tsize}"
        )
    vae.tile_sample_min_tsize = sample_tsize
    vae.tile_latent_min_tsize = sample_tsize // vae.ffactor_temporal
    vae.tile_overlap_factor = overlap_factor
    vae.temporal_tiled_decode = MethodType(_temporal_tiled_decode_enabled, vae)
    vae.use_temporal_tiling = True
    return vae


def load_latents(latent_path: Path) -> torch.Tensor:
    payload = torch.load(latent_path, map_location="cpu")
    latents = payload["latents"] if isinstance(payload, dict) and "latents" in payload else payload
    if latents.ndim != 5:
        raise ValueError(f"Expected 5D latents [B,C,T,H,W], got shape {tuple(latents.shape)}")
    return latents


def maybe_unscale_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
        return latents / vae.config.scaling_factor + vae.config.shift_factor
    return latents / vae.config.scaling_factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-path", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--spatial-tile-size", type=int, default=128)
    parser.add_argument("--temporal-tile-size", type=int, default=16)
    parser.add_argument("--tile-overlap-factor", type=float, default=0.25)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print(f"[info] loading latents from {args.latent_path}")
    latents = load_latents(args.latent_path)
    print(f"[info] latent shape={tuple(latents.shape)} dtype={latents.dtype}")

    print("[info] loading official Hunyuan VAE")
    vae = hunyuanvideo_15_vae.AutoencoderKLConv3D.from_pretrained(
        str(args.model_root / "vae"),
        torch_dtype=dtype,
    ).to(args.device)
    vae.set_tile_sample_min_size(args.spatial_tile_size, args.tile_overlap_factor)
    vae.enable_tiling()
    enable_nonofficial_temporal_tiling(
        vae, sample_tsize=args.temporal_tile_size, overlap_factor=args.tile_overlap_factor
    )
    print(
        "[info] enabled non-official temporal tiling:",
        f"sample_tsize={vae.tile_sample_min_tsize}",
        f"latent_tsize={vae.tile_latent_min_tsize}",
        f"spatial_tile={vae.tile_sample_min_size}",
        f"overlap={vae.tile_overlap_factor}",
    )

    latents = latents.to(args.device, dtype=dtype)
    latents = maybe_unscale_latents(latents, vae)

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=args.device.startswith("cuda")):
            with vae.memory_efficient_context():
                video = vae.decode(latents, return_dict=False, generator=None)[0]

    video = (video / 2 + 0.5).clamp(0, 1).cpu().float()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(args.output_path))
    print(f"[ok] saved {args.output_path}")


if __name__ == "__main__":
    main()
