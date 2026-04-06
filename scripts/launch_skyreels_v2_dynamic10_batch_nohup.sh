#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKYREELS_ROOT="${SKYREELS_ROOT:-/root/SkyReels-V2}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venvs/skyreels-v2}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PATH/bin/python}"
MANIFEST_PATH="${MANIFEST_PATH:-$REPO_ROOT/examples/skyreels_generation/skyreels_dynamic10_720p24_async_manifest.json}"
RUN_NAME="${RUN_NAME:-skyreels_v2_dynamic10_720p24_async}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/examples/skyreels_generation/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/SkyReels-V2/result/$RUN_NAME}"
HF_HOME_PATH="${HF_HOME_PATH:-/root/.cache/huggingface}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BASE_NUM_FRAMES="${BASE_NUM_FRAMES:-57}"
NUM_FRAMES="${NUM_FRAMES:-737}"
FPS="${FPS:-24}"
INFERENCE_STEPS="${INFERENCE_STEPS:-30}"

if [[ ! -d "$SKYREELS_ROOT" ]]; then
  echo "SkyReels repo not found at $SKYREELS_ROOT" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found at $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest not found at $MANIFEST_PATH" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$HF_HOME_PATH" "$OUTPUT_ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/${RUN_NAME}_${STAMP}.log"
PID_FILE="$LOG_DIR/${RUN_NAME}.pid"

CMD=(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/run_skyreels_v2_artifact_batch.py"
  --skyreels-root "$SKYREELS_ROOT"
  --manifest "$MANIFEST_PATH"
  --output-root "$OUTPUT_ROOT"
  --resolution 720P
  --num-frames "$NUM_FRAMES"
  --fps "$FPS"
  --base-num-frames "$BASE_NUM_FRAMES"
  --overlap-history 17
  --ar-step 5
  --causal-block-size 5
  --addnoise-condition 20
  --guidance-scale 6.0
  --shift 8.0
  --inference-steps "$INFERENCE_STEPS"
  --offload
  --save-raw-frames
  --save-latent-chunks
)

printf -v CMD_STR '%q ' "${CMD[@]}"
printf -v LOG_FILE_Q '%q' "$LOG_FILE"
printf -v PID_FILE_Q '%q' "$PID_FILE"
printf -v HF_HOME_Q '%q' "$HF_HOME_PATH"
printf -v ALLOC_CONF_Q '%q' "$ALLOC_CONF"
printf -v CUDA_DEVICES_Q '%q' "$CUDA_DEVICES"

setsid -f bash -lc "echo \$\$ > $PID_FILE_Q && exec nohup env PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=$HF_HOME_Q HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF_Q CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_Q $CMD_STR >> $LOG_FILE_Q 2>&1 < /dev/null"

sleep 1
PID="$(cat "$PID_FILE")"

echo "Started SkyReels dynamic10 batch"
echo "PID: $PID"
echo "Output root: $OUTPUT_ROOT"
echo "Log: $LOG_FILE"
echo "Manifest: $MANIFEST_PATH"
