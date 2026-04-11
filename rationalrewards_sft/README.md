# RationalRewards SFT

Train the RationalRewards reward model using the SFT setup built on LLaMA-Factory.

## Prerequisites

This module is built on top of LLaMA-Factory:

- Repository: [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- Installation guide: [LLaMA-Factory Installation](https://github.com/hiyouga/LLaMA-Factory#installation)

Before running this module, complete the upstream LLaMA-Factory environment setup and confirm:

- `llamafactory-cli` is available in your `PATH`
- CUDA/PyTorch stack is consistent with your GPU driver

## What This Module Does

- Uses `examples/train_full/rationalrewards_full_sft.yaml` as the training config.
- Provides one-command execution through `run_sft.sh`.
- Supports overriding key paths through environment variables.

## Key Files

- `run_sft.sh`: one-command SFT launcher.
- `examples/train_full/rationalrewards_full_sft.yaml`: main training configuration.
- `local_scripts/train_rationalrewards.sh`: local launcher used in prior experiments.
- `data/dataset_info.json`: dataset registry.

## Quick Start

```bash
MODEL_PATH=/path/to/base_model \
DATASET_DIR=/path/to/dataset_dir \
TOKENIZED_PATH=/path/to/tokenized_cache \
OUTPUT_DIR=/path/to/output_dir \
bash run_sft.sh
```

The launcher validates `llamafactory-cli` and exits early if the environment is not set up correctly.

## Config Checklist

Before running, confirm in `examples/train_full/rationalrewards_full_sft.yaml`:

- `model_name_or_path`
- `dataset` and `dataset_dir`
- `tokenized_path`
- `output_dir`
- batch size, gradient accumulation, learning rate, epochs

## Outputs

- Model checkpoints and logs are written to the configured `output_dir`.

## Notes for Release

- This folder intentionally keeps the training code focused on RationalRewards release usage.
- Keep this file as the canonical module README.
