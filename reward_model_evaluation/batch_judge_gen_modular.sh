#!/bin/bash

# Batch evaluation script (modular pipeline) for image generation datasets.
# This script starts a dedicated VLM server for reward-model judging and
# evaluates each checkpoint sequentially.
#
# Before running, set env vars or edit defaults below:
# export MODEL_BASEFOLDER="/path/to/checkpoints/"
# export MODEL_BASENAME="your_model"
# export GENAIBENCH_IMGGEN_DATA="data/genaibench_t2i.parquet"
# export MMRB2_IMGGEN_DATA="data/mmrb2/t2i/"

set -euo pipefail

MODEL_BASEFOLDER="${MODEL_BASEFOLDER:-}"
MODEL_BASENAME="${MODEL_BASENAME:-}"
CHECKPOINTS="${CHECKPOINTS:-4800,7200,6400,5600}"

GENAIBENCH_IMGGEN_DATA="${GENAIBENCH_IMGGEN_DATA:-data/genaibench_t2i.parquet}"
MMRB2_IMGGEN_DATA="${MMRB2_IMGGEN_DATA:-data/mmrb2/t2i/}"

SERVER_HOST="${SERVER_HOST:-http://localhost}"
SERVER_PORT="${SERVER_PORT:-6868}"
CONCURRENCY="${CONCURRENCY:-32}"
SERVE_SCRIPT="${SERVE_SCRIPT:-serve.sh}"

# Output roots for each dataset (kept separate for cleaner bookkeeping)
GENAIBENCH_OUTPUT_ROOT="${GENAIBENCH_OUTPUT_ROOT:-evalresults/genaibench_gen}"
MMRB2_OUTPUT_ROOT="${MMRB2_OUTPUT_ROOT:-evalresults/mmrb2_gen}"

check_server_ready() {
    local max_attempts=240
    local attempt=1
    local url="${SERVER_HOST}:${SERVER_PORT}/v1/models"

    echo "Waiting for server at ${url} ..."
    while [ "${attempt}" -le "${max_attempts}" ]; do
        local response
        response=$(curl -s "${url}" 2>/dev/null || true)
        if [ -n "${response}" ]; then
            echo "Server is ready."
            return 0
        fi
        echo "Attempt ${attempt}/${max_attempts}: not ready yet..."
        sleep 5
        attempt=$((attempt + 1))
    done

    echo "Server failed to start within timeout."
    return 1
}

if [ -z "${MODEL_BASEFOLDER}" ] || [ -z "${MODEL_BASENAME}" ]; then
    echo "ERROR: Please set MODEL_BASEFOLDER and MODEL_BASENAME."
    exit 1
fi

IFS=',' read -ra CHECKPOINT_ARRAY <<< "${CHECKPOINTS}"

for checkpoint in "${CHECKPOINT_ARRAY[@]}"; do
    echo "=================================================="
    echo "Processing checkpoint: ${checkpoint}"
    model_path="${MODEL_BASEFOLDER}checkpoint-${checkpoint}"
    eval_setting="${MODEL_BASENAME}_checkpoint_${checkpoint}_gen"

    echo "Starting dedicated RM server with model: ${model_path}"
    bash "${SERVE_SCRIPT}" "${model_path}" &
    SERVER_PID=$!

    if ! check_server_ready; then
        echo "Failed to start server for checkpoint ${checkpoint}"
        kill "${SERVER_PID}" 2>/dev/null || true
        continue
    fi

    echo "Running modular inference (GenAI-Bench imggen)"
    python3 ./run_pairwise_inference.py \
      --data-file "${GENAIBENCH_IMGGEN_DATA}" \
      --task-type gen \
      --dataset-type genaibench_imggen \
      --evalsetting "${eval_setting}" \
      --output-dir "${GENAIBENCH_OUTPUT_ROOT}/${eval_setting}" \
      --server-host "${SERVER_HOST}" \
      --server-port "${SERVER_PORT}" \
      --concurrency "${CONCURRENCY}" \
      --resume

    sleep 10

    echo "Running modular inference (MMRB2 imggen)"
    python3 ./run_pairwise_inference.py \
      --data-file "${MMRB2_IMGGEN_DATA}" \
      --task-type gen \
      --dataset-type mmrb2_imggen \
      --evalsetting "${eval_setting}" \
      --output-dir "${MMRB2_OUTPUT_ROOT}/${eval_setting}" \
      --server-host "${SERVER_HOST}" \
      --server-port "${SERVER_PORT}" \
      --concurrency "${CONCURRENCY}" \
      --resume

    echo "Stopping RM server"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    pkill -f vllm || true
    sleep 20
done

echo "All checkpoints processed."

