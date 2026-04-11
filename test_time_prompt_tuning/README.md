# Test-Time Prompt Tuning

Improve diffusion outputs at inference time using RationalRewards-guided prompt refinement.

## Two-Stage Workflow

1. Prompt refinement with `prompt_tuning.py`.
2. Regeneration with refined prompts via `inference_with_refinedprompt.py`.

## Quick Start

```bash
GENERATED_IMAGE_DIR=/path/to/first_pass_outputs \
DATASET_PATH=/path/to/dataset.parquet_or_dir \
PRETRAINED_NAME_OR_PATH=/path/to/model_checkpoint \
OUTPUT_DIR=/path/to/refined_outputs \
bash run_test_time_tuning.sh
```

Common variables:

- `MODEL_FAMILY`: `qwen` or `flux`
- `DATANAME`: `pica`, `imgedit`, or `gedit`
- `SERVER_HOST` / `SERVER_PORT`: reward-model endpoint

## Manual Usage

1) Prompt refinement:

```bash
python3 prompt_tuning.py \
  --generated-image-dir /path/to/first_pass_outputs \
  --dataset-path /path/to/dataset.parquet_or_dir \
  --dataname pica \
  --server-host http://localhost \
  --server-port 6868 \
  --concurrency 32 \
  --output-dir /path/to/refined_prompts
```

1) Refined inference:

```bash
python3 inference_with_refinedprompt.py \
  --model-family qwen \
  --pretrained-name-or-path /path/to/model_checkpoint \
  --dataname pica \
  --dataset-path /path/to/dataset.parquet_or_dir \
  --refineprompt-path /path/to/refined_prompts \
  --output-dir /path/to/refined_outputs \
  --num-gpus 8
```

## Outputs

- Refinement stage: per-sample refinement JSON files.
- Inference stage: regenerated images in the requested output directory.

## Notes for Release

- The release path uses `diffusers` + multi-GPU execution, with dataset/model normalization separated in code.
- Keep this file as the canonical module README.
