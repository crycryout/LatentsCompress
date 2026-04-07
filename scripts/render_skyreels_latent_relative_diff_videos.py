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
UPSCALE = 8
TITLE_BAR = 22
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)
SAMPLE_COUNT_FOR_SCALE = 96

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


def render_colorbar(width: int, height: int, font: ImageFont.ImageFont) -> Image.Image:
    grad = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :].repeat(height, axis=0)
    rgb = colorize_signed(grad, 1.0)
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), "below avg", fill=(255, 255, 255), font=font)
    draw.text((max(2, width // 2 - 4), 2), "0", fill=(0, 0, 0), font=font)
    draw.text((max(2, width - 54), 2), "above avg", fill=(0, 0, 0), font=font)
    return img


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


def latent_diff_maps(seq: np.ndarray) -> np.ndarray:
    return np.abs(seq[1:] - seq[:-1]).mean(axis=1)


def relative_maps(diff_maps: np.ndarray) -> np.ndarray:
    per_frame_mean = diff_maps.mean(axis=(1, 2), keepdims=True)
    return (diff_maps / np.maximum(per_frame_mean, 1e-8)) - 1.0


def estimate_vmax(rel_maps: np.ndarray) -> float:
    idxs = evenly_spaced_indices(rel_maps.shape[0], min(SAMPLE_COUNT_FOR_SCALE, rel_maps.shape[0]))
    vals = np.abs(rel_maps[idxs]).reshape(-1)
    return float(np.percentile(vals, 99.5))


def make_panel(img: Image.Image, title: str, font: ImageFont.ImageFont) -> Image.Image:
    return add_title(img, title, font)


def save_frame(path: str, rel_map: np.ndarray, vmax: float, idx: int, font: ImageFont.ImageFont, sample_name: str) -> None:
    rgb = colorize_signed(rel_map, vmax)
    img = Image.fromarray(rgb, mode="RGB").resize((rgb.shape[1] * UPSCALE, rgb.shape[0] * UPSCALE), Image.Resampling.NEAREST)
    make_panel(img, f"{sample_name} relative latent diff {idx} -> {idx + 1}", font).save(path)


def make_preview(rel_map: np.ndarray, vmax: float, out_path: str, font: ImageFont.ImageFont, sample_name: str) -> None:
    rgb = colorize_signed(rel_map, vmax)
    img = Image.fromarray(rgb, mode="RGB").resize((rgb.shape[1] * UPSCALE, rgb.shape[0] * UPSCALE), Image.Resampling.NEAREST)
    panel = make_panel(img, f"{sample_name}: latent relative-diff heatmap", font)
    bar = render_colorbar(panel.width, 20, font)
    canvas = Image.new("RGB", (panel.width, panel.height + 40), BG_COLOR)
    canvas.paste(panel, (0, 0))
    canvas.paste(bar, (0, panel.height + 4))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, panel.height + 26), "Blue: below each frame's mean change, Red: above each frame's mean change", fill=TEXT_COLOR, font=font)
    canvas.save(out_path)


def main() -> None:
    out_root = "/root/LatentsCompress/examples/vbench_codec/latent_relative_diff_videos"
    os.makedirs(out_root, exist_ok=True)
    font = ImageFont.load_default()
    index = {"samples": []}

    for sample in SAMPLES:
        seq = torch.load(sample.latent_path, map_location="cpu").squeeze(0).float().numpy()
        seq = np.transpose(seq, (1, 0, 2, 3))
        diff = latent_diff_maps(seq)
        rel = relative_maps(diff)
        vmax = estimate_vmax(rel)

        sample_dir = os.path.join(out_root, sample.name)
        frame_dir = os.path.join(sample_dir, "relative_diff_frames")
        shutil.rmtree(frame_dir, ignore_errors=True)
        os.makedirs(frame_dir, exist_ok=True)

        for idx in range(rel.shape[0]):
            save_frame(os.path.join(frame_dir, f"{idx:06d}.png"), rel[idx], vmax, idx, font, sample.name)

        mp4_path = os.path.join(sample_dir, f"{sample.name}_latent_relative_diff.mp4")
        preview_path = os.path.join(sample_dir, f"{sample.name}_latent_relative_diff_preview.png")
        encode_mp4(frame_dir, LATENT_FPS, mp4_path)
        preview_idx = min(35, rel.shape[0] - 1)
        make_preview(rel[preview_idx], vmax, preview_path, font, sample.name)

        meta = {
            "name": sample.name,
            "latent_steps": int(seq.shape[0]),
            "diff_pairs": int(rel.shape[0]),
            "fps": LATENT_FPS,
            "duration_sec": float(rel.shape[0] / LATENT_FPS),
            "relative_vmax_99_5_abs": vmax,
            "relative_mean_abs": float(np.mean(np.abs(rel))),
            "mp4": mp4_path,
            "preview": preview_path,
            "description": "Per-step latent abs-diff map normalized by its own spatial mean, then minus 1.0",
        }
        os.makedirs(sample_dir, exist_ok=True)
        with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        index["samples"].append(meta)

    with open(os.path.join(out_root, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
