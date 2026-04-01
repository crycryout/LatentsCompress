#!/usr/bin/env python3
import importlib
import sys


REQUIRED = {
    "hunyuan": [
        "HunyuanVideo15Pipeline",
        "HunyuanVideo15Transformer3DModel",
        "AutoencoderKLHunyuanVideo15",
    ],
    "mochi": [
        "MochiPipeline",
    ],
    "ltx": [
        "LTXPipeline",
        "LTX2Pipeline",
    ],
}


def main() -> int:
    diffusers = importlib.import_module("diffusers")
    print(f"diffusers=={diffusers.__version__}")
    missing = False
    for family, attrs in REQUIRED.items():
        print(f"[{family}]")
        for attr in attrs:
            ok = hasattr(diffusers, attr)
            print(f"  {attr}: {'OK' if ok else 'MISSING'}")
            missing = missing or not ok
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
