# Diffusion RL Training

Train diffusion models with RationalRewards reward feedback.

## Prerequisites

This module follows the Edit-R1 training stack and conventions:

- Edit-R1 repository: [PKU-YuanGroup/Edit-R1](https://github.com/PKU-YuanGroup/Edit-R1)
- Environment setup guide: [Edit-R1 Environment Setup](https://github.com/PKU-YuanGroup/Edit-R1#-environment-set-up)
- Related RL base project: [NVlabs/DiffusionNFT](https://github.com/NVlabs/DiffusionNFT)

Before running this module, set up the environment by following Edit-R1 first, then return here for RationalRewards-specific commands.

## What This Module Does

- Starts a reward-model server through `start_server.sh`.
- Runs RL training for Flux-Kontext or Qwen-Edit targets.
- Provides one-command orchestration via `run_rl_training.sh`.

## Key Files

- `run_rl_training.sh`: one-command runner (server + training).
- `start_server.sh`: reward-model server launcher.
- `train_flux_kontext.sh` / `train_qwen_edit.sh`: training launchers.
- `config/flux_kontext_nft.py` / `config/qwen_edit_nft.py`: core configs.

## Quick Start

```bash
REWARD_MODEL_PATH=/path/to/rationalrewards_checkpoint \
TRAIN_TARGET=flux \
bash run_rl_training.sh
```

`TRAIN_TARGET` options:

- `flux`
- `qwen`

## Manual Run

Terminal A (reward server):

```bash
bash start_server.sh /path/to/rationalrewards_checkpoint
```

Terminal B (training):

```bash
bash train_flux_kontext.sh
# or
bash train_qwen_edit.sh
```

## Config Checklist

Before running, set:

- dataset path (`config.datapath`)
- base diffusion model path (`config.pretrained.model`)
- output path (`config.save_dir`)
- reward server endpoint (`REWARD_SERVER` in training scripts)
- training environment is installed per Edit-R1 guide

## Outputs

- Checkpoints and logs are written under `config.save_dir`.

## Notes for Release

- Replace all remaining `/path/to/...` placeholders with your local paths when running experiments.
- Keep this file as the canonical module README.
