#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKYREELS_ROOT="${SKYREELS_ROOT:-/root/SkyReels-V2}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venvs/skyreels-v2}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_PATH/bin/python}"
PROMPT_FILE="${PROMPT_FILE:-$REPO_ROOT/examples/skyreels_generation/async_720p_30s_full_prompt.txt}"
RUN_NAME="${RUN_NAME:-skyreels_v2_df_async_720p_30s_high_motion}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/examples/skyreels_generation/logs}"
MODEL_ID="${MODEL_ID:-Skywork/SkyReels-V2-DF-14B-720P}"
HF_HOME_PATH="${HF_HOME_PATH:-/root/.cache/huggingface}"
BASE_NUM_FRAMES="${BASE_NUM_FRAMES:-57}"
NUM_FRAMES="${NUM_FRAMES:-737}"
FPS="${FPS:-24}"
SEED="${SEED:-20260406}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
SHIFT="${SHIFT:-8.0}"
INFERENCE_STEPS="${INFERENCE_STEPS:-30}"
OVERLAP_HISTORY="${OVERLAP_HISTORY:-17}"
ADDNOISE_CONDITION="${ADDNOISE_CONDITION:-20}"
AR_STEP="${AR_STEP:-5}"
CAUSAL_BLOCK_SIZE="${CAUSAL_BLOCK_SIZE:-5}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -d "$SKYREELS_ROOT" ]]; then
  echo "SkyReels repo not found at $SKYREELS_ROOT" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found at $PYTHON_BIN" >&2
  echo "Run scripts/install_skyreels_v2_env.sh first or set PYTHON_BIN." >&2
  exit 1
fi

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found at $PROMPT_FILE" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
mkdir -p "$HF_HOME_PATH"
PROMPT="$(tr '\n' ' ' < "$PROMPT_FILE" | sed 's/  */ /g')"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/${RUN_NAME}_${STAMP}.log"
PID_FILE="$LOG_DIR/${RUN_NAME}.pid"

CMD=(
  "$PYTHON_BIN" generate_video_df.py
  --model_id "$MODEL_ID"
  --resolution 720P
  --ar_step "$AR_STEP"
  --causal_block_size "$CAUSAL_BLOCK_SIZE"
  --base_num_frames "$BASE_NUM_FRAMES"
  --num_frames "$NUM_FRAMES"
  --overlap_history "$OVERLAP_HISTORY"
  --prompt "$PROMPT"
  --addnoise_condition "$ADDNOISE_CONDITION"
  --guidance_scale "$GUIDANCE_SCALE"
  --shift "$SHIFT"
  --inference_steps "$INFERENCE_STEPS"
  --fps "$FPS"
  --seed "$SEED"
  --outdir "$RUN_NAME"
  --offload
)

printf -v CMD_STR '%q ' "${CMD[@]}"
printf -v SKYREELS_ROOT_Q '%q' "$SKYREELS_ROOT"
printf -v HF_HOME_Q '%q' "$HF_HOME_PATH"
printf -v LOG_FILE_Q '%q' "$LOG_FILE"
printf -v PID_FILE_Q '%q' "$PID_FILE"
printf -v ALLOC_CONF_Q '%q' "$ALLOC_CONF"
printf -v CUDA_DEVICES_Q '%q' "$CUDA_DEVICES"

setsid -f bash -lc "cd $SKYREELS_ROOT_Q && echo \$\$ > $PID_FILE_Q && exec nohup env PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HOME=$HF_HOME_Q HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=$ALLOC_CONF_Q CUDA_VISIBLE_DEVICES=$CUDA_DEVICES_Q $CMD_STR >> $LOG_FILE_Q 2>&1 < /dev/null"

sleep 1
if [[ ! -f "$PID_FILE" ]]; then
  echo "Failed to create PID file at $PID_FILE" >&2
  exit 1
fi

PID="$(cat "$PID_FILE")"

echo "Started SkyReels async run"
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "PID file: $PID_FILE"
echo "Key settings: resolution=720P fps=$FPS num_frames=$NUM_FRAMES base_num_frames=$BASE_NUM_FRAMES ar_step=$AR_STEP causal_block_size=$CAUSAL_BLOCK_SIZE"
echo "If this still OOMs, retry with a smaller BASE_NUM_FRAMES such as 37."
