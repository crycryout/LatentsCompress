#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


DEFAULTS: dict[str, dict[str, Any]] = {
    "wan_t2v_a14b": {
        "latent_path": "/workspace/outputs/wan22_t2v_a14b_720p16_lighthouse_49f_latents.pt",
        "ckpt_dir": "/workspace/models/Wan2.2-T2V-A14B",
        "wan_root": "/root/Wan2.2",
        "model_dtype": "bf16",
        "stream_chunk_latent_frames": 1,
    },
    "wan_ti2v_5b": {
        "latent_path": "/workspace/video_bench/wan22_ti2v5b_vbench_16x4_seed42/latents/000_subject_consistency_r00_p003_a_person_eating_a_burger.pt",
        "ckpt_dir": "/workspace/models/Wan2.2-TI2V-5B",
        "wan_root": "/root/Wan2.2",
        "model_dtype": "bf16",
        "stream_chunk_latent_frames": 1,
    },
    "opensora_v2_256px": {
        "latent_path": "/workspace/video_bench/opensora_run_t2v/video_256px/sample_0_latents.pt",
        "opensora_root": "/root/Open-Sora",
        "opensora_config": "/root/Open-Sora/configs/diffusion/inference/t2i2v_256px.py",
        "model_dtype": "bf16",
        "stream_chunk_latent_frames": 1,
    },
}


def torch_dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def load_blob(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    blob = torch.load(path, map_location="cpu")
    meta: dict[str, Any] = {"latent_path": str(path)}
    if isinstance(blob, dict):
        meta.update({k: v for k, v in blob.items() if k != "latents"})
        latents = blob["latents"]
    else:
        latents = blob
    return latents.detach().cpu(), meta


def frame_count(x: Any, time_dim: int) -> int:
    if torch.is_tensor(x):
        return int(x.shape[time_dim])
    raise TypeError(f"Unsupported decoded output type: {type(x)!r}")


def maybe_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    synchronize()
    t0 = time.perf_counter()
    out = fn()
    synchronize()
    return out, time.perf_counter() - t0


class BaseAdapter:
    output_time_dim: int

    def decode_full(self, latents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode_stream_chunk(self, latents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def split_chunks(self, latents: torch.Tensor, chunk_latent_frames: int) -> list[torch.Tensor]:
        raise NotImplementedError


class WanAdapter(BaseAdapter):
    output_time_dim = 1

    def __init__(self, family: str, ckpt_dir: str, wan_root: str, device: str, model_dtype: str):
        sys.path.insert(0, wan_root)
        dtype = torch_dtype_from_name(model_dtype)
        if family == "wan_t2v_a14b":
            from wan.modules.vae2_1 import Wan2_1_VAE  # noqa: WPS433

            self.vae = Wan2_1_VAE(
                vae_pth=str(Path(ckpt_dir) / "Wan2.1_VAE.pth"),
                dtype=dtype,
                device=device,
            )
        else:
            from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: WPS433

            self.vae = Wan2_2_VAE(
                vae_pth=str(Path(ckpt_dir) / "Wan2.2_VAE.pth"),
                dtype=dtype,
                device=device,
            )
        self.device = device

    def decode_full(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode([latents.to(self.device, dtype=torch.float32)])[0].detach().cpu()

    def decode_stream_chunk(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decode_full(latents)

    def split_chunks(self, latents: torch.Tensor, chunk_latent_frames: int) -> list[torch.Tensor]:
        total_t = int(latents.shape[1])
        return [latents[:, i : i + chunk_latent_frames].contiguous() for i in range(0, total_t, chunk_latent_frames)]


class OpenSoraAdapter(BaseAdapter):
    output_time_dim = 2

    def __init__(self, opensora_root: str, opensora_config: str, device: str, model_dtype: str, use_native_temporal_tiling: bool):
        sys.path.insert(0, opensora_root)
        from mmengine import Config  # noqa: WPS433
        from opensora.registry import MODELS, build_module  # noqa: WPS433

        cfg = Config.fromfile(opensora_config)
        root = Path(opensora_root)
        if isinstance(cfg.ae, dict) and "from_pretrained" in cfg.ae and isinstance(cfg.ae["from_pretrained"], str):
            src = cfg.ae["from_pretrained"]
            if src.startswith("./"):
                cfg.ae["from_pretrained"] = str((root / src[2:]).resolve())
        self.dtype = torch_dtype_from_name(model_dtype)
        cwd = os.getcwd()
        os.chdir(opensora_root)
        try:
            self.vae = build_module(cfg.ae, MODELS, device_map=device, torch_dtype=self.dtype).eval()
        finally:
            os.chdir(cwd)
        self.device = device
        self.use_native_temporal_tiling = use_native_temporal_tiling

        if use_native_temporal_tiling and hasattr(self.vae, "enable_temporal_tiling"):
            self.vae.enable_temporal_tiling(True)
        elif use_native_temporal_tiling and hasattr(self.vae, "use_temporal_tiling"):
            self.vae.use_temporal_tiling = True

    def decode_full(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents.to(self.device, dtype=self.dtype)).detach().cpu()

    def decode_stream_chunk(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents.to(self.device, dtype=self.dtype)).detach().cpu()

    def split_chunks(self, latents: torch.Tensor, chunk_latent_frames: int) -> list[torch.Tensor]:
        total_t = int(latents.shape[2])
        return [latents[:, :, i : i + chunk_latent_frames].contiguous() for i in range(0, total_t, chunk_latent_frames)]


def build_adapter(args: argparse.Namespace) -> BaseAdapter:
    if args.family in {"wan_t2v_a14b", "wan_ti2v_5b"}:
        return WanAdapter(
            family=args.family,
            ckpt_dir=args.ckpt_dir,
            wan_root=args.wan_root,
            device=args.device,
            model_dtype=args.model_dtype,
        )
    if args.family == "opensora_v2_256px":
        return OpenSoraAdapter(
            opensora_root=args.opensora_root,
            opensora_config=args.opensora_config,
            device=args.device,
            model_dtype=args.model_dtype,
            use_native_temporal_tiling=args.native_temporal_tiling,
        )
    raise ValueError(f"Unsupported family: {args.family}")


def apply_defaults(args: argparse.Namespace) -> None:
    defaults = DEFAULTS[args.family]
    for key, value in defaults.items():
        if getattr(args, key, None) in (None, ""):
            setattr(args, key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try to implement streaming VAE decode and report whether chunked decode works for the given latent sample."
    )
    parser.add_argument("--family", choices=list(DEFAULTS.keys()), required=True)
    parser.add_argument("--latent-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", default=None)
    parser.add_argument("--stream-chunk-latent-frames", type=int, default=None)
    parser.add_argument("--output-json")
    parser.add_argument("--ckpt-dir")
    parser.add_argument("--wan-root")
    parser.add_argument("--opensora-root")
    parser.add_argument("--opensora-config")
    parser.add_argument(
        "--native-temporal-tiling",
        action="store_true",
        help="For Open-Sora only: enable its native temporal tiling switch before decode.",
    )
    args = parser.parse_args()
    apply_defaults(args)
    return args


def main() -> None:
    args = parse_args()
    latents, meta = load_blob(args.latent_path)
    adapter = build_adapter(args)

    report: dict[str, Any] = {
        "family": args.family,
        "latent_path": args.latent_path,
        "latent_shape": list(latents.shape),
        "latent_dtype": str(latents.dtype),
        "native_temporal_tiling": bool(args.native_temporal_tiling),
        "meta": {k: v for k, v in meta.items() if not torch.is_tensor(v)},
        "full_decode": {},
        "stream_decode": {},
    }

    full_video, full_dt = timed(lambda: adapter.decode_full(latents))
    full_frames = frame_count(full_video, adapter.output_time_dim)
    report["full_decode"] = {
        "success": True,
        "time_sec": full_dt,
        "frames": full_frames,
        "output_shape": list(full_video.shape),
    }
    del full_video
    maybe_cleanup()

    chunks = adapter.split_chunks(latents, args.stream_chunk_latent_frames)
    chunk_reports = []
    total_stream_frames = 0
    stream_success = True
    first_error = None

    for idx, chunk in enumerate(chunks):
        try:
            video, dt = timed(lambda c=chunk: adapter.decode_stream_chunk(c))
            out_frames = frame_count(video, adapter.output_time_dim)
            total_stream_frames += out_frames
            chunk_reports.append(
                {
                    "chunk_index": idx,
                    "latent_shape": list(chunk.shape),
                    "success": True,
                    "time_sec": dt,
                    "frames": out_frames,
                    "output_shape": list(video.shape),
                }
            )
            del video
            maybe_cleanup()
        except Exception as exc:  # pragma: no cover - runtime path
            stream_success = False
            first_error = repr(exc)
            chunk_reports.append(
                {
                    "chunk_index": idx,
                    "latent_shape": list(chunk.shape),
                    "success": False,
                    "error": repr(exc),
                }
            )
            break

    report["stream_decode"] = {
        "stream_chunk_latent_frames": args.stream_chunk_latent_frames,
        "success": stream_success,
        "chunk_count": len(chunks),
        "total_stream_frames": total_stream_frames,
        "matches_full_frame_count": bool(stream_success and total_stream_frames == full_frames),
        "first_error": first_error,
        "chunks": chunk_reports,
    }

    out_path = Path(
        args.output_json
        or f"/root/GenLatents/examples/{args.family}_stream_decode_report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
