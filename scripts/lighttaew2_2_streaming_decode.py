from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-decode saved Wan2.2 latents with lighttaew2_2 and pipe frames to ffmpeg."
    )
    parser.add_argument("--latent-path", type=Path, required=True, help="Path to a saved Wan2.2 latent .pt file.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output target. Local `.mp4` path or an `rtmp://...` URL.",
    )
    parser.add_argument(
        "--lighttae-path",
        type=Path,
        default=Path("/root/models/vae/lighttaew2_2.safetensors"),
        help="Path to the lighttaew2_2 checkpoint.",
    )
    parser.add_argument(
        "--lightx2v-root",
        type=Path,
        default=Path("/root/LightX2V"),
        help="Path to the LightX2V checkout used to load TAEHV.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device for decoding.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp32"],
        help="Decoder dtype.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output FPS. Defaults to latent metadata fps.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="How many latent timesteps to push at once. Default 1 for true step-wise streaming.",
    )
    parser.add_argument(
        "--emit-mode",
        choices=["auto", "as_fast_as_possible", "realtime"],
        default="auto",
        help="`realtime` paces frame writes to fps. `auto` uses realtime for RTMP and fast for local files.",
    )
    parser.add_argument(
        "--max-latent-steps",
        type=int,
        default=None,
        help="Optional cap on latent steps for short streaming tests.",
    )
    parser.add_argument(
        "--ffmpeg-loglevel",
        default="error",
        help="ffmpeg loglevel.",
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help="ffmpeg x264 preset.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="CRF for local MP4 output.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save a JSON summary.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite local output files.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def load_saved_latents(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict):
        meta = {k: v for k, v in blob.items() if k != "latents"}
        latents = blob["latents"]
    else:
        meta = {}
        latents = blob
    return latents.to(torch.float32).cpu(), meta


def maybe_cleanup(device: str) -> None:
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def ensure_parent_dir(output: str) -> None:
    if output.startswith("rtmp://"):
        return
    Path(output).parent.mkdir(parents=True, exist_ok=True)


def load_lightx2v_tae_module(lightx2v_root: Path):
    for name in [
        "lightx2v",
        "lightx2v.models",
        "lightx2v.models.video_encoders",
        "lightx2v.models.video_encoders.hf",
        "lightx2v.models.video_encoders.hf.wan",
    ]:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

    def load_module(mod_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    tae_module = load_module(
        "lightx2v.models.video_encoders.hf.tae",
        lightx2v_root / "lightx2v/models/video_encoders/hf/tae.py",
    )
    return tae_module


@dataclass
class StepStats:
    latent_start: int
    latent_len: int
    emitted_frames: int
    decode_sec: float
    total_emitted_frames: int


class Wan22LightTAEStreamingDecoder:
    def __init__(
        self,
        lightx2v_root: Path,
        checkpoint_path: Path,
        device: str,
        dtype: torch.dtype,
        need_scaled: bool = True,
    ):
        tae_module = load_lightx2v_tae_module(lightx2v_root)
        self.TAEHV = tae_module.TAEHV
        self.MemBlock = tae_module.MemBlock
        self.TPool = tae_module.TPool
        self.TGrow = tae_module.TGrow
        self.TWorkItem = tae_module.TWorkItem

        self.device = device
        self.dtype = dtype
        self.need_scaled = need_scaled
        self.model = self.TAEHV(str(checkpoint_path), model_type="wan22").to(
            device=self.device,
            dtype=self.dtype,
        ).eval()
        self.batch_size = 1

        self.latents_mean = torch.tensor(
            [
                -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
                -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
                -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
                -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
                -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
                0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
            ],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 48, 1, 1, 1)
        self.latents_std = torch.tensor(
            [
                0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
                0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
                0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
                0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
                0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
                0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
            ],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 48, 1, 1, 1)
        self.reset()

    def reset(self) -> None:
        self.mem: list[Any] = [None] * len(self.model.decoder)
        self.raw_frames_seen = 0
        self.frames_emitted = 0

    def _scale_latents(self, latents_bcthw: torch.Tensor) -> torch.Tensor:
        if not self.need_scaled:
            return latents_bcthw
        return latents_bcthw * self.latents_std + self.latents_mean

    def _enqueue_successor(self, queue: list, tensor: torch.Tensor, block_index: int) -> None:
        queue.insert(0, self.TWorkItem(tensor, block_index))

    def _process_source_timestep(self, xt: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        queue = [self.TWorkItem(xt, 0)]
        while queue:
            work = queue.pop(0)
            cur = work.input_tensor
            block_index = work.block_index
            if block_index == len(self.model.decoder):
                outputs.append(cur)
                continue

            block = self.model.decoder[block_index]
            if isinstance(block, self.MemBlock):
                prev = self.mem[block_index]
                if prev is None:
                    prev_for_block = torch.zeros_like(cur)
                    nxt = block(cur, prev_for_block)
                    self.mem[block_index] = cur.detach().clone()
                else:
                    if prev.shape != cur.shape:
                        prev_for_block = torch.zeros_like(cur)
                        nxt = block(cur, prev_for_block)
                        self.mem[block_index] = cur.detach().clone()
                    else:
                        nxt = block(cur, prev)
                        self.mem[block_index].copy_(cur)
                self._enqueue_successor(queue, nxt, block_index + 1)
            elif isinstance(block, self.TPool):
                cached = self.mem[block_index]
                if cached is None:
                    cached = []
                    self.mem[block_index] = cached
                cached.append(cur)
                if len(cached) == block.stride:
                    pooled = block(torch.cat(cached, dim=1).view(self.batch_size * block.stride, cur.shape[1], cur.shape[2], cur.shape[3]))
                    self.mem[block_index] = []
                    self._enqueue_successor(queue, pooled, block_index + 1)
            elif isinstance(block, self.TGrow):
                grown = block(cur)
                grown = grown.view(self.batch_size, block.stride * grown.shape[1], grown.shape[2], grown.shape[3])
                for nxt in reversed(grown.chunk(block.stride, dim=1)):
                    self._enqueue_successor(queue, nxt, block_index + 1)
            else:
                nxt = block(cur)
                self._enqueue_successor(queue, nxt, block_index + 1)
        return outputs

    def push_latent_chunk(self, latents_cthw: torch.Tensor, return_cpu: bool = True) -> torch.Tensor:
        if latents_cthw.ndim == 3:
            latents_cthw = latents_cthw.unsqueeze(1)
        if latents_cthw.ndim != 4:
            raise ValueError(f"Expected [C,T,H,W] or [C,H,W], got {tuple(latents_cthw.shape)}")

        latents_bcthw = latents_cthw.unsqueeze(0).to(device=self.device, dtype=torch.float32)
        latents_bcthw = self._scale_latents(latents_bcthw)
        latents_ntchw = latents_bcthw.permute(0, 2, 1, 3, 4).contiguous().to(self.device, dtype=self.dtype)

        raw_outputs: list[torch.Tensor] = []
        for xt in latents_ntchw.unbind(dim=1):
            raw_outputs.extend(self._process_source_timestep(xt))

        if not raw_outputs:
            return torch.empty((0, 3, 0, 0), dtype=torch.float32)

        video = torch.stack(raw_outputs, dim=1)
        video = video.clamp_(0, 1)
        if self.model.patch_size > 1:
            video = F.pixel_shuffle(video, self.model.patch_size)

        raw_before = self.raw_frames_seen
        raw_count = int(video.shape[1])
        self.raw_frames_seen += raw_count
        trim = max(self.model.frames_to_trim - raw_before, 0)
        if trim >= raw_count:
            return torch.empty((0, video.shape[2], video.shape[3], video.shape[4]), dtype=torch.float32)

        emitted = video[:, trim:]
        self.frames_emitted += int(emitted.shape[1])
        emitted0 = emitted[0].detach()
        if return_cpu:
            return emitted0.cpu()
        return emitted0


class FfmpegStreamSink:
    def __init__(
        self,
        output: str,
        width: int,
        height: int,
        fps: float,
        loglevel: str,
        preset: str,
        crf: int,
        overwrite: bool,
    ):
        self.output = output
        self.width = width
        self.height = height
        self.fps = fps
        self.loglevel = loglevel
        self.preset = preset
        self.crf = crf
        self.overwrite = overwrite
        self.frames_written = 0
        self.started_at = time.perf_counter()
        self.process = subprocess.Popen(
            self._build_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def _build_cmd(self) -> list[str]:
        cmd = ["ffmpeg"]
        if self.overwrite and not self.output.startswith("rtmp://"):
            cmd.append("-y")
        cmd.extend(
            [
                "-loglevel",
                self.loglevel,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{self.width}x{self.height}",
                "-r",
                f"{self.fps:.6f}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                self.preset,
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if self.output.startswith("rtmp://"):
            cmd.extend(
                [
                    "-tune",
                    "zerolatency",
                    "-f",
                    "flv",
                    self.output,
                ]
            )
        else:
            cmd.extend(
                [
                    "-crf",
                    str(self.crf),
                    "-movflags",
                    "+frag_keyframe+empty_moov+default_base_moof+faststart",
                    self.output,
                ]
            )
        return cmd

    def write_frames(self, frames_tchw: torch.Tensor, realtime: bool) -> None:
        if frames_tchw.numel() == 0:
            return
        frames_u8 = (
            frames_tchw.clamp(0, 1)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")

        if realtime:
            for frame in frames_u8:
                deadline = self.started_at + (self.frames_written / self.fps)
                now = time.perf_counter()
                if deadline > now:
                    time.sleep(deadline - now)
                self.process.stdin.write(frame.tobytes())
                self.process.stdin.flush()
                self.frames_written += 1
        else:
            self.process.stdin.write(frames_u8.tobytes())
            self.process.stdin.flush()
            self.frames_written += int(frames_u8.shape[0])

    def close(self) -> None:
        stderr_text = ""
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stderr is not None:
            stderr_text = self.process.stderr.read().decode("utf-8", errors="ignore")
        returncode = self.process.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {returncode}\n{stderr_text}")


def resolved_emit_mode(requested: str, output: str) -> str:
    if requested != "auto":
        return requested
    return "realtime" if output.startswith("rtmp://") else "as_fast_as_possible"


def main() -> None:
    args = parse_args()
    if not args.lighttae_path.exists():
        raise SystemExit(f"Missing checkpoint: {args.lighttae_path}")
    ensure_parent_dir(args.output)

    latents, meta = load_saved_latents(args.latent_path)
    total_latent_steps = int(latents.shape[1])
    if args.max_latent_steps is not None:
        total_latent_steps = min(total_latent_steps, int(args.max_latent_steps))
        latents = latents[:, :total_latent_steps].contiguous()

    fps = float(args.fps or meta.get("fps") or 24.0)
    emit_mode = resolved_emit_mode(args.emit_mode, args.output)
    realtime = emit_mode == "realtime"

    decoder = Wan22LightTAEStreamingDecoder(
        lightx2v_root=args.lightx2v_root,
        checkpoint_path=args.lighttae_path,
        device=args.device,
        dtype=torch_dtype(args.dtype),
        need_scaled=True,
    )

    sink: FfmpegStreamSink | None = None
    step_stats: list[dict[str, Any]] = []
    stream_started = time.perf_counter()

    try:
        for latent_start in range(0, total_latent_steps, args.chunk_size):
            chunk = latents[:, latent_start:latent_start + args.chunk_size].contiguous()
            synchronize(args.device)
            t0 = time.perf_counter()
            frames_tchw = decoder.push_latent_chunk(chunk)
            synchronize(args.device)
            dt = time.perf_counter() - t0

            if sink is None and frames_tchw.numel() > 0:
                sink = FfmpegStreamSink(
                    output=args.output,
                    width=int(frames_tchw.shape[-1]),
                    height=int(frames_tchw.shape[-2]),
                    fps=fps,
                    loglevel=args.ffmpeg_loglevel,
                    preset=args.preset,
                    crf=args.crf,
                    overwrite=args.overwrite,
                )

            emitted = int(frames_tchw.shape[0]) if frames_tchw.ndim == 4 else 0
            if sink is not None and emitted > 0:
                sink.write_frames(frames_tchw, realtime=realtime)

            step_stats.append(
                StepStats(
                    latent_start=latent_start,
                    latent_len=int(chunk.shape[1]),
                    emitted_frames=emitted,
                    decode_sec=dt,
                    total_emitted_frames=decoder.frames_emitted,
                ).__dict__
            )
            print(
                f"latent[{latent_start}:{latent_start + int(chunk.shape[1])}] "
                f"decoded in {dt:.4f}s, emitted {emitted} frame(s), total={decoder.frames_emitted}"
            )
    finally:
        if sink is not None:
            sink.close()
        maybe_cleanup(args.device)

    total_sec = time.perf_counter() - stream_started
    summary = {
        "latent_path": str(args.latent_path),
        "output": args.output,
        "fps": fps,
        "emit_mode": emit_mode,
        "latent_shape": list(latents.shape),
        "meta": meta,
        "total_latent_steps": total_latent_steps,
        "total_frames_emitted": decoder.frames_emitted,
        "total_stream_wall_sec": total_sec,
        "stream_fps_equivalent": (decoder.frames_emitted / total_sec) if total_sec > 0 else None,
        "steps": step_stats,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "steps"}, indent=2))


if __name__ == "__main__":
    main()
