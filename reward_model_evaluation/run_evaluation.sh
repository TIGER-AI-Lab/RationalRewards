#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for reward-model evaluation.
# Usage:
#   DATA_FILE=/path/to/data.parquet \
#   OUTPUT_DIR=evalresults/rr_eval \
#   TASK_TYPE=edit \
#   bash run_evaluation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_FILE="${DATA_FILE:-/path/to/data.parquet_or_dir}"
TASK_TYPE="${TASK_TYPE:-edit}"              # edit | gen
DATASET_TYPE="${DATASET_TYPE:-auto}"
EVALSETTING="${EVALSETTING:-rr_eval_s4000}"
OUTPUT_DIR="${OUTPUT_DIR:-evalresults/${EVALSETTING}}"
SERVER_HOST="${SERVER_HOST:-http://localhost}"
SERVER_PORT="${SERVER_PORT:-6868}"
CONCURRENCY="${CONCURRENCY:-32}"
MODE="${MODE:-all}"                         # all | text | visual
LABEL_SOURCE="${LABEL_SOURCE:-chosen}"      # chosen | ground_truth_fields

python3 run_pairwise_inference.py \
  --data-file "${DATA_FILE}" \
  --task-type "${TASK_TYPE}" \
  --dataset-type "${DATASET_TYPE}" \
  --evalsetting "${EVALSETTING}" \
  --output-dir "${OUTPUT_DIR}" \
  --server-host "${SERVER_HOST}" \
  --server-port "${SERVER_PORT}" \
  --concurrency "${CONCURRENCY}" \
  --resume

python3 compute_pairwise_accuracy.py \
  --result-dir "${OUTPUT_DIR}" \
  --task-type "${TASK_TYPE}" \
  --mode "${MODE}" \
  --label-source "${LABEL_SOURCE}"
