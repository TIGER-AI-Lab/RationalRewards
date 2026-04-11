#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   DIFFUSION_ENV=/path/to/env \
#   REWARD_SERVER=http://127.0.0.1:6868 \
#   bash train_qwen_edit.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DIFFUSION_ENV="${DIFFUSION_ENV:-}"
if [[ -n "${DIFFUSION_ENV}" ]]; then
  # shellcheck disable=SC1090
  source activate "${DIFFUSION_ENV}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-52421}"
export REWARD_SERVER="${REWARD_SERVER:-http://127.0.0.1:6868}"
export WANDB_MODE="${WANDB_MODE:-offline}"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
CONFIG_NAME="${CONFIG_NAME:-config/qwen_edit_nft.py:qwen_mllm_reward}"

python3 -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --node_rank "${NODE_RANK}" \
  train_qwen_edit_rationalrewards.py --config "${CONFIG_NAME}"