#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


LATENT_FPS = 6
TITLE_BAR = 22
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
HEATMAP_SCALE = 8
GRID_SCALE = 2
PCA_SCALE = 8
PCA_SAMPLE_POINTS = 40000

INFERNO = np.array(
    [
        [0, 0, 4],
        [31, 12, 72],
        [85, 15, 109],
        [136, 34, 106],
        [186, 54, 85],
        [227, 89, 51],
        [249, 140, 10],
        [252, 195, 65],
        [252, 255, 164],
    ],
    dtype=np.float32,
)

COOLWARM = np.array(
    [
        [59, 76, 192],
        [98, 130, 234],
        [141, 176, 254],
        [184, 208, 249],
        [221, 221, 221],
        [245, 196, 173],
        [244, 154, 123],
        [229, 103, 73],
        [180, 4, 38],
    ],
    dtype=np.float32,
)


@dataclass
class Sample:
    name: str
    latent_path: str


SAMPLES = [
    Sample(
        name="wingsuit",
        latent_path="/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/wingsuit_rescue_glacier_pullup/full_video_latents_dedup.pt",
    ),
    Sample(
        name="neon",
        latent_path="/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/neon_hoverbike_chain_reaction/full_video_latents_dedup.pt",
    ),
    Sample(
        name="avalanche",
        latent_path="/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/avalanche_snowmobile_bridge_escape/full_video_latents_dedup.pt",
    ),
]


def evenly_spaced_indices(total: int, count: int) -> np.ndarray:
    if total <= count:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, count, dtype=np.int64)


def colorize_positive(arr: np.ndarray, vmax: float) -> np.ndarray:
    vmax = max(float(vmax), 1e-8)
    y = np.clip(arr.astype(np.float32) / vmax, 0.0, 1.0)
    pos = y * (len(INFERNO) - 1)
    i = np.floor(pos).astype(np.int32)
    j = np.clip(i + 1, 0, len(INFERNO) - 1)
    t = (pos - i)[..., None]
    return ((1.0 - t) * INFERNO[i] + t * INFERNO[j]).astype(np.uint8)


def colorize_signed(arr: np.ndarray, vmax: float) -> np.ndarray:
    vmax = max(float(vmax), 1e-8)
    y = np.clip((arr.astype(np.float32) / (2.0 * vmax)) + 0.5, 0.0, 1.0)
    pos = y * (len(COOLWARM) - 1)
    i = np.floor(pos).astype(np.int32)
    j = np.clip(i + 1, 0, len(COOLWARM) - 1)
    t = (pos - i)[..., None]
    return ((1.0 - t) * COOLWARM[i] + t * COOLWARM[j]).astype(np.uint8)


def add_title(img: Image.Image, title: str, font: ImageFont.ImageFont) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + TITLE_BAR), BG_COLOR)
    canvas.paste(img, (0, TITLE_BAR))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, fill=TEXT_COLOR, font=font)
    return canvas


def save_frame(path: str, img: Image.Image, title: str, font: ImageFont.ImageFont) -> None:
    add_title(img, title, font).save(path)


def encode_mp4(frame_dir: str, fps: int, out_path: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(frame_dir, "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            out_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def render_colorbar(width: int, height: int, palette: np.ndarray, font: ImageFont.ImageFont, signed: bool) -> Image.Image:
    if signed:
        grad = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :].repeat(height, axis=0)
        rgb = colorize_signed(grad, 1.0)
    else:
        grad = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :].repeat(height, axis=0)
        rgb = colorize_positive(grad, 1.0)
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)
    if signed:
        draw.text((2, 2), "-", fill=(255, 255, 255), font=font)
        draw.text((max(2, width // 2 - 4), 2), "0", fill=(0, 0, 0), font=font)
        draw.text((max(2, width - 10), 2), "+", fill=(0, 0, 0), font=font)
    else:
        draw.text((2, 2), "low", fill=(255, 255, 255), font=font)
        draw.text((max(2, width - 26), 2), "high", fill=(0, 0, 0), font=font)
    return img


def fit_pca(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, c, h, w = seq.shape
    rng = np.random.default_rng(0)
    total_points = t * h * w
    take = min(PCA_SAMPLE_POINTS, total_points)
    flat = np.transpose(seq, (0, 2, 3, 1)).reshape(total_points, c)
    idx = rng.choice(total_points, size=take, replace=False)
    sample = flat[idx].astype(np.float32)
    mean = sample.mean(axis=0)
    centered = sample - mean
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:3]
    basis = vecs[:, order].astype(np.float32)
    proj = centered @ basis
    low = np.percentile(proj, 1.0, axis=0).astype(np.float32)
    high = np.percentile(proj, 99.0, axis=0).astype(np.float32)
    return mean.astype(np.float32), basis, low, high


def project_pca(frame: np.ndarray, mean: np.ndarray, basis: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    h, w, c = frame.shape
    flat = frame.reshape(-1, c).astype(np.float32)
    proj = (flat - mean) @ basis
    rgb = (proj - low) / np.maximum(high - low, 1e-8)
    rgb = np.clip(rgb, 0.0, 1.0)
    return (rgb.reshape(h, w, 3) * 255.0).astype(np.uint8)


def make_grid_image(frame: np.ndarray, vmax: float) -> Image.Image:
    tiles = []
    for ch in range(frame.shape[0]):
        rgb = colorize_signed(frame[ch], vmax)
        tile = Image.fromarray(rgb, mode="RGB")
        tiles.append(tile)
    tile_w, tile_h = tiles[0].size
    canvas = Image.new("RGB", (tile_w * 4, tile_h * 4))
    for ch, tile in enumerate(tiles):
        y = (ch // 4) * tile_h
        x = (ch % 4) * tile_w
        canvas.paste(tile, (x, y))
    return canvas.resize((canvas.width * GRID_SCALE, canvas.height * GRID_SCALE), Image.Resampling.NEAREST)


def make_heatmap_image(frame: np.ndarray, vmax: float) -> Image.Image:
    heat = np.abs(frame).mean(axis=0)
    rgb = colorize_positive(heat, vmax)
    img = Image.fromarray(rgb, mode="RGB")
    return img.resize((img.width * HEATMAP_SCALE, img.height * HEATMAP_SCALE), Image.Resampling.NEAREST)


def make_pca_image(frame: np.ndarray, mean: np.ndarray, basis: np.ndarray, low: np.ndarray, high: np.ndarray) -> Image.Image:
    hwc = np.transpose(frame, (1, 2, 0))
    rgb = project_pca(hwc, mean, basis, low, high)
    img = Image.fromarray(rgb, mode="RGB")
    return img.resize((img.width * PCA_SCALE, img.height * PCA_SCALE), Image.Resampling.NEAREST)


def make_preview_strip(grid_img: Image.Image, heat_img: Image.Image, pca_img: Image.Image, out_path: str, font: ImageFont.ImageFont) -> None:
    panels = [
        add_title(grid_img, "Latent channel grid", font),
        add_title(heat_img, "Latent magnitude heatmap", font),
        add_title(pca_img, "Latent PCA -> RGB", font),
    ]
    bar_signed = render_colorbar(panels[0].width, 18, COOLWARM, font, signed=True)
    bar_heat = render_colorbar(panels[1].width, 18, INFERNO, font, signed=False)
    width = sum(p.width for p in panels) + 40
    height = max(p.height for p in panels) + 70
    canvas = Image.new("RGB", (width, height), BG_COLOR)
    x = 10
    for p in panels:
        canvas.paste(p, (x, 10))
        x += p.width + 10
    canvas.paste(bar_signed, (10, height - 48))
    canvas.paste(bar_heat, (20 + bar_signed.width, height - 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, height - 22), "Grid uses a signed scale. Heatmap and PCA use positive intensity / RGB scales.", fill=TEXT_COLOR, font=font)
    canvas.save(out_path)


def main() -> None:
    out_root = "/root/LatentsCompress/examples/vbench_codec/latent_visualization_videos"
    os.makedirs(out_root, exist_ok=True)
    font = ImageFont.load_default()
    index = {"samples": []}

    for sample in SAMPLES:
        sample_dir = os.path.join(out_root, sample.name)
        os.makedirs(sample_dir, exist_ok=True)
        seq = torch.load(sample.latent_path, map_location="cpu").squeeze(0).float().numpy()
        # seq: C,T,H,W -> T,C,H,W
        seq = np.transpose(seq, (1, 0, 2, 3))

        grid_dir = os.path.join(sample_dir, "grid_frames")
        heat_dir = os.path.join(sample_dir, "heatmap_frames")
        pca_dir = os.path.join(sample_dir, "pca_frames")
        for d in [grid_dir, heat_dir, pca_dir]:
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)

        grid_vmax = float(np.percentile(np.abs(seq), 99.5))
        heat_vmax = float(np.percentile(np.abs(seq).mean(axis=1), 99.5))
        pca_mean, pca_basis, pca_low, pca_high = fit_pca(seq)

        for idx in range(seq.shape[0]):
            frame = seq[idx]
            grid_img = make_grid_image(frame, grid_vmax)
            heat_img = make_heatmap_image(frame, heat_vmax)
            pca_img = make_pca_image(frame, pca_mean, pca_basis, pca_low, pca_high)

            save_frame(os.path.join(grid_dir, f"{idx:06d}.png"), grid_img, f"{sample.name} latent grid step {idx}", font)
            save_frame(os.path.join(heat_dir, f"{idx:06d}.png"), heat_img, f"{sample.name} latent heatmap step {idx}", font)
            save_frame(os.path.join(pca_dir, f"{idx:06d}.png"), pca_img, f"{sample.name} latent PCA step {idx}", font)

        grid_video = os.path.join(sample_dir, f"{sample.name}_latent_grid.mp4")
        heat_video = os.path.join(sample_dir, f"{sample.name}_latent_heatmap.mp4")
        pca_video = os.path.join(sample_dir, f"{sample.name}_latent_pca_rgb.mp4")
        encode_mp4(grid_dir, LATENT_FPS, grid_video)
        encode_mp4(heat_dir, LATENT_FPS, heat_video)
        encode_mp4(pca_dir, LATENT_FPS, pca_video)

        preview_idx = min(35, seq.shape[0] - 1)
        preview_strip = os.path.join(sample_dir, f"{sample.name}_preview_triptych.png")
        make_preview_strip(
            make_grid_image(seq[preview_idx], grid_vmax),
            make_heatmap_image(seq[preview_idx], heat_vmax),
            make_pca_image(seq[preview_idx], pca_mean, pca_basis, pca_low, pca_high),
            preview_strip,
            font,
        )

        meta = {
            "name": sample.name,
            "latent_steps": int(seq.shape[0]),
            "fps": LATENT_FPS,
            "duration_sec": float(seq.shape[0] / LATENT_FPS),
            "grid_vmax_99_5_abs": grid_vmax,
            "heat_vmax_99_5_absmean": heat_vmax,
            "pca_mean": pca_mean.tolist(),
            "pca_basis": pca_basis.tolist(),
            "pca_low_p1": pca_low.tolist(),
            "pca_high_p99": pca_high.tolist(),
            "grid_video": grid_video,
            "heatmap_video": heat_video,
            "pca_rgb_video": pca_video,
            "preview_triptych": preview_strip,
        }
        with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        index["samples"].append(meta)

    with open(os.path.join(out_root, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
