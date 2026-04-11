#!/bin/bash

# Batch evaluation script for image generation models
# Before running, set the following environment variables or edit the values below:
# export MODEL_BASEFOLDER="/path/to/your/model/checkpoints"
# export MODEL_BASENAME="your_model_name"
# export GENAIBENCH_IMGGEN_DATA="data/genaibench_t2i.parquet"
# export MMRB2_IMGGEN_DATA="data/mmrb2/t2i/"

# Input parameters - SET THESE ENVIRONMENT VARIABLES OR EDIT THE VALUES BELOW
model_basefolder="${MODEL_BASEFOLDER:-}"  # Path to the base directory containing model checkpoints
checkpoints="4800,7200,6400,5600"
model_basename="${MODEL_BASENAME:-}"  # Model name prefix for evaluation settings

# Data file paths - SET THESE ENVIRONMENT VARIABLES OR EDIT THE VALUES BELOW
GENAIBENCH_IMGGEN_DATA="${GENAIBENCH_IMGGEN_DATA:-data/genaibench_t2i.parquet}"
MMRB2_IMGGEN_DATA="${MMRB2_IMGGEN_DATA:-data/mmrb2/t2i/}"

# Change to the directory where serve.sh is located

# Function to check if server is ready
check_server_ready() {
    local max_attempts=240  # Maximum attempts (60 * 5s = 5 minutes)
    local attempt=1

    echo "Waiting for server to be ready..."
    while [ $attempt -le $max_attempts ]; do
        response=$(curl -s http://localhost:6868/v1/models 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$response" ]; then
            echo "Server response: $response"
            echo "Server is ready!"
            return 0
        elif [ $? -eq 0 ] && [ -z "$response" ]; then
            echo "Server response: empty"
            echo "Attempt $attempt/$max_attempts: Server returned empty response, waiting..."
        else
            echo "Attempt $attempt/$max_attempts: Server not ready yet, waiting..."
        fi
        sleep 5
        ((attempt++))
    done

    echo "Server failed to start within timeout period"
    return 1
}

# Split checkpoints by comma and process each one
IFS=',' read -ra CHECKPOINT_ARRAY <<< "$checkpoints"

for checkpoint in "${CHECKPOINT_ARRAY[@]}"; do
    echo "Processing checkpoint: $checkpoint"

    # Construct full model path
    model_path="${model_basefolder}checkpoint-${checkpoint}"
    eval_setting="${model_basename}_checkpoint_${checkpoint}_gen"

    echo "Starting server with model: $model_path"

    # Start server in background
    bash serve.sh "$model_path" &
    SERVER_PID=$!

    # Wait for server to be ready
    if ! check_server_ready; then
        echo "Failed to start server for checkpoint $checkpoint"
        kill $SERVER_PID 2>/dev/null
        continue
    fi

    echo "Running inference for checkpoint $checkpoint with eval setting: $eval_setting"

    # Run inference scripts
    python3 ./rm_inference/inference_genaibench_imggen.py --data-file "${GENAIBENCH_IMGGEN_DATA}" --evalsetting "${eval_setting}"
    sleep 30
    # python3 ./rm_inference/inference_genaibench_imgedit.py --data-file "${GENAIBENCH_IMGEDIT_DATA}" --evalsetting "${eval_setting}_edit"
    # python3 ./rm_inference/inference_mmrb2_imgedit.py --data-file "${MMRB2_IMGEDIT_DATA}" --evalsetting "${eval_setting}_edit"
    python3 ./rm_inference/inference_mmrb2_imggen.py --data-file "${MMRB2_IMGGEN_DATA}" --evalsetting "${eval_setting}"
    sleep 30
    # python3 ./rm_inference/inference_editreward_imgedit.py --data-file "${EDITREWARD_IMGEDIT_DATA}" --evalsetting "${eval_setting}_edit"

    # Kill server after inference
    echo "Stopping server for checkpoint $checkpoint"
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    pkill -f vllm
    sleep 30

    echo "Completed processing checkpoint: $checkpoint"
    echo "----------------------------------------"
done

echo "All checkpoints processed!"