#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venvs/skyreels-v2}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

"$PYTHON_BIN" -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install \
  "diffusers>=0.31.0" \
  "transformers==4.49.0" \
  "tokenizers==0.21.1" \
  "accelerate==1.6.0" \
  "numpy<2" \
  einops \
  sentencepiece \
  tqdm \
  imageio \
  imageio-ffmpeg \
  easydict \
  ftfy \
  dashscope \
  huggingface_hub \
  safetensors \
  "moviepy<2" \
  decord \
  opencv-python

echo "SkyReels environment ready at $VENV_PATH"
