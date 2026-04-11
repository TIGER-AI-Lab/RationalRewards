#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for diffusion RL training with RationalRewards.
# Starts RM server, waits briefly, then launches training.
#
# Usage:
#   REWARD_MODEL_PATH=/path/to/rationalrewards_ckpt \
#   TRAIN_TARGET=flux \
#   bash run_rl_training.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-/path/to/rationalrewards_checkpoint}"
TRAIN_TARGET="${TRAIN_TARGET:-flux}"   # flux | qwen
SERVER_WARMUP_SEC="${SERVER_WARMUP_SEC:-30}"

echo "Starting reward model server..."
bash start_server.sh "${REWARD_MODEL_PATH}" &
SERVER_PID=$!

cleanup() {
  echo "Stopping reward model server (pid=${SERVER_PID})"
  kill "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting ${SERVER_WARMUP_SEC}s for server warmup..."
sleep "${SERVER_WARMUP_SEC}"

if [[ "${TRAIN_TARGET}" == "flux" ]]; then
  bash train_flux_kontext.sh
elif [[ "${TRAIN_TARGET}" == "qwen" ]]; then
  bash train_qwen_edit.sh
else
  echo "Unsupported TRAIN_TARGET=${TRAIN_TARGET}. Use flux or qwen."
  exit 1
fi
