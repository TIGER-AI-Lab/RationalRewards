#!/usr/bin/env bash
set -euo pipefail

# Start a local vLLM reward server.
# Usage:
#   VLLM_ENV=/path/to/env \
#   bash start_server.sh /path/to/rationalrewards_checkpoint

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <model_path>"
  exit 1
fi

MODEL_PATH="$1"
VLLM_ENV="${VLLM_ENV:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-8B-Instruct}"
SERVER_PORT="${SERVER_PORT:-6868}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"

if [[ -n "${VLLM_ENV}" ]]; then
  # shellcheck disable=SC1090
  source activate "${VLLM_ENV}"
fi

vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --limit-mm-per-prompt '{"image": 16, "video": 0}' \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --port "${SERVER_PORT}"