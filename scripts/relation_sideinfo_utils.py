from __future__ import annotations

import io
import math
import zlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class RelationPacket:
    step_idx: int
    lowpass_kernel: int
    tile_rows: int
    tile_cols: int
    corr_h: int
    corr_w: int
    decay_steps: float
    global_gain: torch.Tensor
    global_bias: torch.Tensor
    tile_gain: torch.Tensor
    tile_bias: torch.Tensor
    low_residual: torch.Tensor


def avg_blur(video_cthw: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return video_cthw
    pad = kernel // 2
    tchw = video_cthw.permute(1, 0, 2, 3).contiguous()
    blurred = F.avg_pool2d(tchw, kernel_size=kernel, stride=1, padding=pad)
    return blurred.permute(1, 0, 2, 3).contiguous()


def resize_spatial(video_cthw: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    tchw = video_cthw.permute(1, 0, 2, 3).contiguous()
    resized = F.interpolate(tchw, size=out_hw, mode="bilinear", align_corners=False)
    return resized.permute(1, 0, 2, 3).contiguous()


def downsample_lowfreq(video_cthw: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    tchw = video_cthw.permute(1, 0, 2, 3).contiguous()
    resized = F.interpolate(tchw, size=out_hw, mode="area")
    return resized.permute(1, 0, 2, 3).contiguous()


def channel_affine(src: torch.Tensor, dst: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    c = src.shape[0]
    src_flat = src.view(c, -1)
    dst_flat = dst.view(c, -1)
    src_mean = src_flat.mean(dim=1).view(c, 1, 1, 1)
    dst_mean = dst_flat.mean(dim=1).view(c, 1, 1, 1)
    src_std = src_flat.std(dim=1, unbiased=False).view(c, 1, 1, 1)
    dst_std = dst_flat.std(dim=1, unbiased=False).view(c, 1, 1, 1)
    gain = (dst_std + eps) / (src_std + eps)
    bias = dst_mean - gain * src_mean
    return gain, bias


def tile_bounds(length: int, parts: int) -> list[tuple[int, int]]:
    edges = [round(i * length / parts) for i in range(parts + 1)]
    return [(edges[i], edges[i + 1]) for i in range(parts)]


def tile_affine(src: torch.Tensor, dst: torch.Tensor, rows: int, cols: int, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    c, _, h, w = src.shape
    gain = torch.ones((c, rows, cols), dtype=src.dtype, device=src.device)
    bias = torch.zeros((c, rows, cols), dtype=src.dtype, device=src.device)
    row_bounds = tile_bounds(h, rows)
    col_bounds = tile_bounds(w, cols)
    for r, (hs, he) in enumerate(row_bounds):
        for q, (ws, we) in enumerate(col_bounds):
            src_tile = src[:, :, hs:he, ws:we].contiguous().view(c, -1)
            dst_tile = dst[:, :, hs:he, ws:we].contiguous().view(c, -1)
            src_mean = src_tile.mean(dim=1)
            dst_mean = dst_tile.mean(dim=1)
            src_std = src_tile.std(dim=1, unbiased=False)
            dst_std = dst_tile.std(dim=1, unbiased=False)
            gain[:, r, q] = (dst_std + eps) / (src_std + eps)
            bias[:, r, q] = dst_mean - gain[:, r, q] * src_mean
    return gain, bias


def apply_tile_affine(x: torch.Tensor, gain: torch.Tensor, bias: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    out = x.clone()
    _, _, h, w = x.shape
    row_bounds = tile_bounds(h, rows)
    col_bounds = tile_bounds(w, cols)
    for r, (hs, he) in enumerate(row_bounds):
        for q, (ws, we) in enumerate(col_bounds):
            g = gain[:, r, q].view(-1, 1, 1, 1)
            b = bias[:, r, q].view(-1, 1, 1, 1)
            out[:, :, hs:he, ws:we] = out[:, :, hs:he, ws:we] * g + b
    return out


def build_relation_packet(
    step_idx: int,
    tae_group: torch.Tensor,
    vae_group: torch.Tensor,
    lowpass_kernel: int,
    tile_rows: int,
    tile_cols: int,
    correction_divisor: int,
    decay_steps: float,
) -> RelationPacket:
    h, w = tae_group.shape[-2:]
    corr_h = max(16, h // max(correction_divisor, 1))
    corr_w = max(16, w // max(correction_divisor, 1))
    tae_low = downsample_lowfreq(tae_group, (corr_h, corr_w))
    vae_low = downsample_lowfreq(vae_group, (corr_h, corr_w))
    global_gain, global_bias = channel_affine(tae_low, vae_low)
    tae_low_global = tae_low * global_gain + global_bias
    tile_gain, tile_bias = tile_affine(tae_low_global, vae_low, tile_rows, tile_cols)
    low_residual = (vae_low - tae_low_global).clamp(-1.0, 1.0)
    return RelationPacket(
        step_idx=step_idx,
        lowpass_kernel=lowpass_kernel,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        corr_h=corr_h,
        corr_w=corr_w,
        decay_steps=decay_steps,
        global_gain=global_gain,
        global_bias=global_bias,
        tile_gain=tile_gain,
        tile_bias=tile_bias,
        low_residual=low_residual,
    )


def apply_relation_packet(tae_group: torch.Tensor, packet: RelationPacket, age_steps: int) -> torch.Tensor:
    device = tae_group.device
    dtype = tae_group.dtype
    global_gain = packet.global_gain.to(device=device, dtype=dtype)
    global_bias = packet.global_bias.to(device=device, dtype=dtype)
    tile_gain = packet.tile_gain.to(device=device, dtype=dtype)
    tile_bias = packet.tile_bias.to(device=device, dtype=dtype)
    low_residual = packet.low_residual.to(device=device, dtype=dtype)

    low_ds = downsample_lowfreq(tae_group, (packet.corr_h, packet.corr_w))
    corrected_low = low_ds * global_gain + global_bias
    weight = math.exp(-float(age_steps) / max(packet.decay_steps, 1e-4))
    corrected_low = corrected_low + low_residual * weight
    corrected_low = apply_tile_affine(corrected_low, tile_gain, tile_bias, packet.tile_rows, packet.tile_cols)
    high = tae_group - resize_spatial(low_ds, tuple(tae_group.shape[-2:]))
    corrected = (resize_spatial(corrected_low, tuple(tae_group.shape[-2:])) + high).clamp(-1.0, 1.0)
    return corrected


def packet_to_cpu_dict(packet: RelationPacket) -> dict[str, Any]:
    return {
        "step_idx": packet.step_idx,
        "lowpass_kernel": packet.lowpass_kernel,
        "tile_rows": packet.tile_rows,
        "tile_cols": packet.tile_cols,
        "corr_h": packet.corr_h,
        "corr_w": packet.corr_w,
        "decay_steps": packet.decay_steps,
        "global_gain": packet.global_gain.detach().to(torch.float16).cpu().contiguous(),
        "global_bias": packet.global_bias.detach().to(torch.float16).cpu().contiguous(),
        "tile_gain": packet.tile_gain.detach().to(torch.float16).cpu().contiguous(),
        "tile_bias": packet.tile_bias.detach().to(torch.float16).cpu().contiguous(),
        "low_residual": packet.low_residual.detach().to(torch.float16).cpu().contiguous(),
    }


def cpu_dict_to_packet(data: dict[str, Any]) -> RelationPacket:
    return RelationPacket(
        step_idx=int(data["step_idx"]),
        lowpass_kernel=int(data["lowpass_kernel"]),
        tile_rows=int(data["tile_rows"]),
        tile_cols=int(data["tile_cols"]),
        corr_h=int(data["corr_h"]),
        corr_w=int(data["corr_w"]),
        decay_steps=float(data["decay_steps"]),
        global_gain=data["global_gain"].to(torch.float32),
        global_bias=data["global_bias"].to(torch.float32),
        tile_gain=data["tile_gain"].to(torch.float32),
        tile_bias=data["tile_bias"].to(torch.float32),
        low_residual=data["low_residual"].to(torch.float32),
    )


def serialize_sideinfo(packets: list[RelationPacket], meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "meta": meta,
        "packets": [packet_to_cpu_dict(packet) for packet in packets],
    }


def deserialize_sideinfo(blob: dict[str, Any]) -> tuple[dict[str, Any], list[RelationPacket]]:
    meta = blob["meta"]
    packets = [cpu_dict_to_packet(packet) for packet in blob["packets"]]
    return meta, packets


def measure_sideinfo_bytes(sideinfo: dict[str, Any]) -> dict[str, int]:
    buf = io.BytesIO()
    torch.save(sideinfo, buf)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, level=9)
    return {
        "sideinfo_torch_bytes": len(raw),
        "sideinfo_zlib_bytes": len(compressed),
    }
