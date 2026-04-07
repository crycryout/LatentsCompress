#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


POOL = 8
FRAME_FPS = 24
LATENT_FPS = 6
SAMPLE_FRAMES_FOR_SCALE = 96
TITLE_BAR = 22
BG_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 0, 0)

PALETTE = np.array(
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


@dataclass
class Sample:
    name: str
    latent_path: str
    frame_path: str


SAMPLES = [
    Sample(
        name="wingsuit",
        latent_path="/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/wingsuit_rescue_glacier_pullup/full_video_latents_dedup.pt",
        frame_path="/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/wingsuit_rescue_glacier_pullup/raw_frames_uint8.npy",
    ),
    Sample(
        name="neon",
        latent_path="/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/neon_hoverbike_chain_reaction/full_video_latents_dedup.pt",
        frame_path="/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/neon_hoverbike_chain_reaction/raw_frames_uint8.npy",
    ),
    Sample(
        name="avalanche",
        latent_path="/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/avalanche_snowmobile_bridge_escape/full_video_latents_dedup.pt",
        frame_path="/root/SkyReels-V2/result/skyreels_v2_dynamic5_720p24_async/avalanche_snowmobile_bridge_escape/raw_frames_uint8.npy",
    ),
]


def evenly_spaced_indices(total: int, count: int) -> np.ndarray:
    if total <= count:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, count, dtype=np.int64)


def colorize(arr: np.ndarray, vmax: float) -> np.ndarray:
    vmax = max(float(vmax), 1e-8)
    y = np.clip(arr.astype(np.float32) / vmax, 0.0, 1.0)
    pos = y * (len(PALETTE) - 1)
    i = np.floor(pos).astype(np.int32)
    j = np.clip(i + 1, 0, len(PALETTE) - 1)
    t = (pos - i)[..., None]
    return ((1.0 - t) * PALETTE[i] + t * PALETTE[j]).astype(np.uint8)


def add_title(rgb: np.ndarray, title: str, font: ImageFont.ImageFont) -> Image.Image:
    img = Image.fromarray(rgb, mode="RGB")
    canvas = Image.new("RGB", (img.width, img.height + TITLE_BAR), BG_COLOR)
    canvas.paste(img, (0, TITLE_BAR))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, fill=TEXT_COLOR, font=font)
    return canvas


def save_frame(path: str, rgb: np.ndarray, title: str, font: ImageFont.ImageFont) -> None:
    add_title(rgb, title, font).save(path)


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


def render_colorbar(width: int, height: int, font: ImageFont.ImageFont) -> Image.Image:
    grad = np.linspace(0, 1, width, dtype=np.float32)[None, :].repeat(height, axis=0)
    rgb = colorize(grad, 1.0)
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), "low", fill=(255, 255, 255), font=font)
    draw.text((max(2, width - 26), 2), "high", fill=(0, 0, 0), font=font)
    return img


def video_pair_diff(frames: np.memmap, idx: int, step: int) -> np.ndarray:
    a = frames[idx].astype(np.float32)
    b = frames[idx + step].astype(np.float32)
    return np.abs(b - a).mean(axis=-1) / 255.0


def pool_video_map(diff_map: np.ndarray) -> np.ndarray:
    h, w = diff_map.shape
    return diff_map.reshape(h // POOL, POOL, w // POOL, POOL).mean(axis=(1, 3))


def latent_pair_diff(seq: np.ndarray, idx: int) -> np.ndarray:
    return np.abs(seq[idx + 1] - seq[idx]).mean(axis=0)


def estimate_vmax_video(frames: np.memmap, step: int) -> float:
    total = frames.shape[0] - step
    idxs = evenly_spaced_indices(total, SAMPLE_FRAMES_FOR_SCALE)
    vals = []
    for idx in idxs:
        vals.append(video_pair_diff(frames, int(idx), step).reshape(-1))
    return float(np.percentile(np.concatenate(vals, axis=0), 99.5))


def estimate_vmax_video_pooled(frames: np.memmap, step: int) -> float:
    total = frames.shape[0] - step
    idxs = evenly_spaced_indices(total, SAMPLE_FRAMES_FOR_SCALE)
    vals = []
    for idx in idxs:
        vals.append(pool_video_map(video_pair_diff(frames, int(idx), step)).reshape(-1))
    return float(np.percentile(np.concatenate(vals, axis=0), 99.5))


def estimate_vmax_latent(seq: np.ndarray) -> float:
    total = seq.shape[0] - 1
    idxs = evenly_spaced_indices(total, min(SAMPLE_FRAMES_FOR_SCALE, total))
    vals = []
    for idx in idxs:
        vals.append(latent_pair_diff(seq, int(idx)).reshape(-1))
    return float(np.percentile(np.concatenate(vals, axis=0), 99.5))


def make_comparison_strip(
    sample_name: str,
    frame_rgb: np.ndarray,
    pooled_video_rgb: np.ndarray,
    latent_rgb: np.ndarray,
    out_path: str,
    font: ImageFont.ImageFont,
) -> None:
    scale = 5
    panels = [
        add_title(np.array(Image.fromarray(frame_rgb).resize((frame_rgb.shape[1] // POOL * scale, frame_rgb.shape[0] // POOL * scale), Image.Resampling.NEAREST)), f"{sample_name}: video pooled heatmap", font),
        add_title(np.array(Image.fromarray(pooled_video_rgb).resize((pooled_video_rgb.shape[1] * scale, pooled_video_rgb.shape[0] * scale), Image.Resampling.NEAREST)), f"{sample_name}: pooled 90x160", font),
        add_title(np.array(Image.fromarray(latent_rgb).resize((latent_rgb.shape[1] * scale, latent_rgb.shape[0] * scale), Image.Resampling.NEAREST)), f"{sample_name}: latent 90x160", font),
    ]
    bar = render_colorbar(panels[0].width, 20, font)
    width = sum(p.width for p in panels) + 40
    height = max(p.height for p in panels) + 60
    canvas = Image.new("RGB", (width, height), BG_COLOR)
    x = 10
    for p in panels:
        canvas.paste(p, (x, 10))
        x += p.width + 10
    canvas.paste(bar, ((width - bar.width) // 2, height - 30))
    ImageDraw.Draw(canvas).text((10, height - 22), "Independent 99.5th-percentile scales per modality.", fill=TEXT_COLOR, font=font)
    canvas.save(out_path)


def main() -> None:
    out_root = "/root/LatentsCompress/examples/vbench_codec/heatmap_videos"
    os.makedirs(out_root, exist_ok=True)
    font = ImageFont.load_default()
    index = {"samples": []}

    for sample in SAMPLES:
        sample_dir = os.path.join(out_root, sample.name)
        os.makedirs(sample_dir, exist_ok=True)
        frames = np.load(sample.frame_path, mmap_mode="r")
        latent = torch.load(sample.latent_path, map_location="cpu").squeeze(0).float().numpy()
        seq = np.transpose(latent, (1, 0, 2, 3))

        video_vmax = estimate_vmax_video(frames, step=1)
        video_pooled_vmax = estimate_vmax_video_pooled(frames, step=1)
        latent_vmax = estimate_vmax_latent(seq)

        frame_heat_dir = os.path.join(sample_dir, "frame_diff_frames")
        latent_heat_dir = os.path.join(sample_dir, "latent_diff_frames")
        shutil.rmtree(frame_heat_dir, ignore_errors=True)
        shutil.rmtree(latent_heat_dir, ignore_errors=True)
        os.makedirs(frame_heat_dir, exist_ok=True)
        os.makedirs(latent_heat_dir, exist_ok=True)

        for idx in range(frames.shape[0] - 1):
            diff = video_pair_diff(frames, idx, step=1)
            rgb = colorize(diff, video_vmax)
            save_frame(
                os.path.join(frame_heat_dir, f"{idx:06d}.png"),
                rgb,
                f"{sample.name} frame diff {idx} -> {idx + 1}",
                font,
            )

        for idx in range(seq.shape[0] - 1):
            diff = latent_pair_diff(seq, idx)
            rgb = colorize(diff, latent_vmax)
            rgb = np.array(Image.fromarray(rgb, mode="RGB").resize((1280, 720), Image.Resampling.NEAREST))
            save_frame(
                os.path.join(latent_heat_dir, f"{idx:06d}.png"),
                rgb,
                f"{sample.name} latent diff {idx} -> {idx + 1}",
                font,
            )

        frame_video_path = os.path.join(sample_dir, f"{sample.name}_frame_diff_t1_heatmap.mp4")
        latent_video_path = os.path.join(sample_dir, f"{sample.name}_latent_diff_t1_heatmap.mp4")
        encode_mp4(frame_heat_dir, FRAME_FPS, frame_video_path)
        encode_mp4(latent_heat_dir, LATENT_FPS, latent_video_path)

        preview_idx = min(35, frames.shape[0] - 2)
        latent_preview_idx = min(35, seq.shape[0] - 2)
        preview_video = colorize(video_pair_diff(frames, preview_idx, 1), video_vmax)
        preview_pooled = colorize(pool_video_map(video_pair_diff(frames, preview_idx, 1)), video_pooled_vmax)
        preview_latent = colorize(latent_pair_diff(seq, latent_preview_idx), latent_vmax)
        make_comparison_strip(
            sample.name,
            preview_video,
            preview_pooled,
            preview_latent,
            os.path.join(sample_dir, f"{sample.name}_preview_triptych.png"),
            font,
        )

        sample_meta = {
            "name": sample.name,
            "frame_count": int(frames.shape[0]),
            "latent_steps": int(seq.shape[0]),
            "frame_diff_pairs": int(frames.shape[0] - 1),
            "latent_diff_pairs": int(seq.shape[0] - 1),
            "frame_diff_fps": FRAME_FPS,
            "latent_diff_fps": LATENT_FPS,
            "frame_diff_duration_sec": float((frames.shape[0] - 1) / FRAME_FPS),
            "latent_diff_duration_sec": float((seq.shape[0] - 1) / LATENT_FPS),
            "video_vmax_99_5": video_vmax,
            "video_pooled_vmax_99_5": video_pooled_vmax,
            "latent_vmax_99_5": latent_vmax,
            "frame_heatmap_video": frame_video_path,
            "latent_heatmap_video": latent_video_path,
            "preview_triptych": os.path.join(sample_dir, f"{sample.name}_preview_triptych.png"),
        }
        with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
            json.dump(sample_meta, f, indent=2)
        index["samples"].append(sample_meta)

    with open(os.path.join(out_root, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
