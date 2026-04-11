#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for RationalRewards SFT training.
# Usage:
#   MODEL_PATH=/path/to/base_model \
#   OUTPUT_DIR=/path/to/output \
#   bash run_sft.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
CONFIG_PATH="${PROJECT_DIR}/examples/train_full/rationalrewards_full_sft.yaml"

# Optional: activate your environment before running.
# source activate /path/to/llamafactory/env

MODEL_PATH="${MODEL_PATH:-/path/to/base_model}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/data}"
TOKENIZED_PATH="${TOKENIZED_PATH:-/path/to/tokenized_cache}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/output_dir}"

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "Error: llamafactory-cli not found in PATH."
  exit 1
fi

echo "Running SFT with config: ${CONFIG_PATH}"
echo "Tip: edit ${CONFIG_PATH} for full hyperparameter control."

cd "${PROJECT_DIR}"
llamafactory-cli train "${CONFIG_PATH}" \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_dir "${DATASET_DIR}" \
  --tokenized_path "${TOKENIZED_PATH}" \
  --output_dir "${OUTPUT_DIR}"
