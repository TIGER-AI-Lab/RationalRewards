#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for test-time prompt tuning.
# It runs:
#   1) prompt refinement
#   2) refined-image generation
#
# Usage:
#   GENERATED_IMAGE_DIR=/path/to/first_pass_outputs \
#   DATASET_PATH=/path/to/pica.parquet \
#   PRETRAINED_NAME_OR_PATH=/path/to/Qwen-Image-Edit-2509 \
#   OUTPUT_DIR=/path/to/refined_outputs \
#   bash run_test_time_tuning.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

GENERATED_IMAGE_DIR="${GENERATED_IMAGE_DIR:-/path/to/first_pass_outputs}"
REFINED_PROMPT_DIR="${REFINED_PROMPT_DIR:-${GENERATED_IMAGE_DIR}_refined_instruction}"
SERVER_HOST="${SERVER_HOST:-http://localhost}"
SERVER_PORT="${SERVER_PORT:-6868}"
CONCURRENCY="${CONCURRENCY:-32}"

MODEL_FAMILY="${MODEL_FAMILY:-qwen}"  # qwen | flux
PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-/path/to/model_checkpoint}"
LORA_PATH="${LORA_PATH:-}"
DATANAME="${DATANAME:-pica}"          # pica | imgedit | gedit
DATASET_PATH="${DATASET_PATH:-/path/to/dataset.parquet_or_dir}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/refined_outputs}"
NUM_GPUS="${NUM_GPUS:-8}"
SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
MAX_EDGE="${MAX_EDGE:-1024}"
SKIP_GOOD_THRESHOLD="${SKIP_GOOD_THRESHOLD:-3.0}"

python3 prompt_tuning.py \
  --generated-image-dir "${GENERATED_IMAGE_DIR}" \
  --dataset-path "${DATASET_PATH}" \
  --dataname "${DATANAME}" \
  --server-host "${SERVER_HOST}" \
  --server-port "${SERVER_PORT}" \
  --concurrency "${CONCURRENCY}" \
  --output-dir "${REFINED_PROMPT_DIR}"

CMD=(
  python3 inference_with_refinedprompt.py
  --model-family "${MODEL_FAMILY}"
  --pretrained-name-or-path "${PRETRAINED_NAME_OR_PATH}"
  --dataname "${DATANAME}"
  --dataset-path "${DATASET_PATH}"
  --refineprompt-path "${REFINED_PROMPT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --num-gpus "${NUM_GPUS}"
  --seed "${SEED}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --guidance-scale "${GUIDANCE_SCALE}"
  --true-cfg-scale "${TRUE_CFG_SCALE}"
  --max-edge "${MAX_EDGE}"
  --skip-good-threshold "${SKIP_GOOD_THRESHOLD}"
)

if [[ -n "${LORA_PATH}" ]]; then
  CMD+=(--lora-path "${LORA_PATH}")
fi

"${CMD[@]}"
