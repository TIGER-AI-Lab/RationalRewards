#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${PROJECT_DIR}/examples/train_full/rationalrewards_full_sft.yaml"

LLAMAFACTORY_ENV="${LLAMAFACTORY_ENV:-}"
if [[ -n "${LLAMAFACTORY_ENV}" ]]; then
  # shellcheck disable=SC1090
  source activate "${LLAMAFACTORY_ENV}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-2048001}"

cd "${PROJECT_DIR}"
llamafactory-cli train "${CONFIG_PATH}"
