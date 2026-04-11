# Reward Model Evaluation

Evaluate pairwise preference prediction accuracy of a RationalRewards-compatible reward model.

## Prerequisites

This module requires a running vLLM server before inference starts.

- vLLM repository: [vllm-project/vllm](https://github.com/vllm-project/vllm)
- Install vLLM by following its official guide, then verify `vllm` is available in your environment.
- Use a Python environment with this module dependencies:

```bash
pip install pandas pyarrow pillow aiohttp requests tqdm
```

## What This Module Does

- Runs pairwise inference with `run_pairwise_inference.py`.
- Computes aggregate accuracy with `compute_pairwise_accuracy.py`.
- Supports both image-editing (`edit`) and text-to-image generation (`gen`) tasks.
- Provides one-command execution through `run_evaluation.sh`.

## Supported Dataset Types

Use `--dataset-type auto` (recommended), or set explicitly:

- `mmrb2_imgedit`
- `mmrb2_imggen`
- `genaibench_imgedit`
- `genaibench_imggen`
- `editreward_imgedit`

## Quick Start (Server First, Then Inference)

1. Start a vLLM reward-model server:

```bash
vllm serve /path/to/rationalrewards_checkpoint \
  --served-model-name Qwen3-VL-8B-Instruct \
  --port 6868
```

Recommended optional flags for multi-GPU setups:

```bash
--tensor-parallel-size 1 \
--data-parallel-size 8 \
--gpu-memory-utilization 0.8 \
--max-model-len 40960
```

1. In a new terminal, run inference + scoring:

```bash
DATA_FILE=/path/to/data.parquet_or_dir \
OUTPUT_DIR=evalresults/rr_eval_s4000 \
TASK_TYPE=edit \
SERVER_HOST=http://localhost \
SERVER_PORT=6868 \
bash run_evaluation.sh
```

Default server endpoint in this repository:

- `SERVER_HOST=http://localhost`
- `SERVER_PORT=6868`

## Runtime Configuration

Set these environment variables before running `run_evaluation.sh`:

- `DATA_FILE`: input parquet file or directory
- `OUTPUT_DIR`: directory for per-sample outputs and metrics
- `TASK_TYPE`: `edit` or `gen`
- `DATASET_TYPE`: `auto` (recommended) or explicit type
- `EVALSETTING`: evaluation profile name (default `rr_eval_s4000`)
- `SERVER_HOST` and `SERVER_PORT`: vLLM endpoint
- `CONCURRENCY`: number of async requests (default `32`)
- `MODE`: `all`, `text`, or `visual` for scoring
- `LABEL_SOURCE`: `chosen` or `ground_truth_fields`

## Manual Two-Step Usage

1. Pairwise inference:

```bash
python3 run_pairwise_inference.py \
  --data-file /path/to/data.parquet_or_dir \
  --task-type edit \
  --dataset-type auto \
  --evalsetting rr_eval_s4000 \
  --output-dir evalresults/rr_eval_s4000 \
  --server-host http://localhost \
  --server-port 6868 \
  --concurrency 32 \
  --resume
```

1. Accuracy scoring:

```bash
python3 compute_pairwise_accuracy.py \
  --result-dir evalresults/rr_eval_s4000 \
  --task-type edit \
  --mode all \
  --label-source chosen
```

## Outputs

- Inference writes one `*_pairwise.json` per sample pair.
- Accuracy script prints aggregate metrics and can optionally show failures.

## Notes for Release

- Modular implementation lives under `modular_rm_eval/` and is already used by default entry scripts.
- Legacy batch scripts are kept for checkpoint sweep workflows.
- Keep only this README as the canonical module documentation.
