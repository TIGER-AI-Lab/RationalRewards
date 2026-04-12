# RationalRewards: Reasoning Rewards Scale Visual Generation at Both Training and Test Time

RationalRewards is a reasoning-based reward model and toolkit for visual generation.  
It supports both:

- **train-time optimization** (RL with structured, interpretable reward feedback), and
- **test-time optimization** (Generate-Critique-Refine prompt revisions without parameter changes).

[Project Page](https://tiger-ai-lab.github.io/RationalRewards/) | [RationalRewards-T2I](https://huggingface.co/TIGER-Lab/RationalRewards-8B-T2I) | [RationalRewards-Edit](https://huggingface.co/TIGER-Lab/RationalRewards-8B-Edit) | [RationalRewards-SFT-Data](https://huggingface.co/datasets/TIGER-Lab/RationalRewards-SFTData) | [Diffusion RL Training Data](https://huggingface.co/datasets/TIGER-Lab/RationalRewards_DiffusionNFT_TrainData)

## 📣 News

- **[2026/04]** RationalRewards code release: SFT, reward-model evaluation, Diffusion RL with RationalRewards, and test-time prompt tuning.
- **[2026/04]** RationalRewards-8B reward models and public datasets are available on Hugging Face.
- **[Coming Soon]** Paper preprint and FlowFactory supports on other RL methods.

## 💥 Introduction

Most reward models for visual generation reduce rich human preferences into a single scalar score.  
RationalRewards instead generates **multi-dimensional critiques before scoring**, enabling:

- denser and more interpretable reward signals for RL, and
- post-hoc prompt refinement through a **Generate-Critique-Refine** loop.

To avoid expensive rationale annotation, we use **Preference-Anchored Rationalization (PARROT)** to recover high-quality rationales from preference data through anchored generation, consistency filtering, and distillation.

In our experiments, RationalRewards achieves state-of-the-art preference prediction among open-source reward models, improves diffusion RL training for both text-to-image and editing, and delivers strong test-time prompt tuning gains. Surprisingly, test-time prompt tuning can achieve gains that match or even surpass parameter tuning gains. 

**This reveals a critical finding:** the visual generators possess dormant rendering capabilities, which leaves substantial room for improvement via prompt-space optimization.


![RationalRewards teaser](assets/teaser.png)

*Train-time RL and test-time prompt tuning with RationalRewards across visual generation benchmarks.*


## ✨ Features

- **Reasoning reward model:** critique-then-score instead of scalar-only scoring.
- **Unified pipeline:** includes SFT, pairwise evaluation, RL training, and test-time prompt tuning.
- **Train-time + test-time optimization:** works in both parameter space and prompt space.

## 📋 Table of Contents

- [🛠 Environment Setup](#-environment-setup)
- [🚀 Quick Start](#-quick-start)
- [🔬 Evaluation and Benchmarks](#-evaluation-and-benchmarks)
- [📦 Public Datasets and Models](#-public-datasets-and-models)
- [🧪 Repository Structure](#-repository-structure)
- [🙏 Acknowledgements](#-acknowledgements)
- [📚 Citation](#-citation)

## 🛠 Environment Setup

We recommend Linux + CUDA GPUs with **separate Python environments** for each module.

This repository builds on the following upstream projects:

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (SFT training)
- [vLLM](https://github.com/vllm-project/vllm) (reward-model serving)
- [Edit-R1](https://github.com/PKU-YuanGroup/Edit-R1) (RL training environment conventions)
- [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) (diffusion RL design references)
- [diffusers](https://github.com/huggingface/diffusers) (inference/training ecosystem)

Install upstream dependencies first:

1. SFT setup: [LLaMA-Factory Installation](https://github.com/hiyouga/LLaMA-Factory#installation)
2. Reward serving setup: [vLLM Installation](https://github.com/vllm-project/vllm)
3. RL setup: [Edit-R1 Environment Setup](https://github.com/PKU-YuanGroup/Edit-R1#-environment-set-up)

Then follow module-level READMEs for exact command dependencies:

- `rationalrewards_sft/README.md`
- `reward_model_evaluation/README.md`
- `diffusion_rl_training/README.md`
- `test_time_prompt_tuning/README.md`

## 🚀 Quick Start

**Please refer to readme in the individual folders for details.** All scripts are environment-variable driven. Replace `/path/to/...` placeholders with your local paths.

### Step 1: Train RationalRewards with SFT

```bash
bash rationalrewards_sft/run_sft.sh
```

### Step 2: Start reward-model server (required before evaluation)

```bash
vllm serve /path/to/rationalrewards_checkpoint --port 6868
```

### Step 3: Run pairwise reward-model evaluation

```bash
bash reward_model_evaluation/run_evaluation.sh
```

### Step 4: Run diffusion RL training with RationalRewards feedback

```bash
bash diffusion_rl_training/run_rl_training.sh
```

### Step 5: Run test-time prompt tuning (Generate-Critique-Refine)

```bash
bash test_time_prompt_tuning/run_test_time_tuning.sh
```

## 🔬 Evaluation and Benchmarks

Released evaluation data covers benchmark families including:

- [GenAI-Bench](https://github.com/TIGER-AI-Lab/GenAI-Bench)
- [Multimodal RewardBench 2](https://github.com/facebookresearch/MMRB2)
- [EditReward-Bench](https://github.com/TIGER-AI-Lab/EditReward)

We uploaded the cleaned Preference evaluation data we use: [TIGER-Lab/RationalRewards-EvalData-GenAIBench-MMRB2-ERBench](https://huggingface.co/datasets/TIGER-Lab/RationalRewards-EvalData-GenAIBench-MMRB2-ERBench)

## 📦 Public Datasets and Models

- SFT training data for Reward Model Training: [TIGER-Lab/RationalRewards-SFTData](https://huggingface.co/datasets/TIGER-Lab/RationalRewards-SFTData)
- RL training data for diffusion RL training: [TIGER-Lab/RationalRewards_DiffusionNFT_TrainData](https://huggingface.co/datasets/TIGER-Lab/RationalRewards_DiffusionNFT_TrainData)
We collect the dataset by performing difficulty-aware stratified sampling.
- RationalRewards (T2I): [TIGER-Lab/RationalRewards-8B-T2I](https://huggingface.co/TIGER-Lab/RationalRewards-8B-T2I)
- RationalRewards (Edit): [TIGER-Lab/RationalRewards-8B-Edit](https://huggingface.co/TIGER-Lab/RationalRewards-8B-Edit)

## 🧪 Repository Structure

- `rationalrewards_sft/`: train RationalRewards via SFT.
- `reward_model_evaluation/`: run pairwise inference and aggregate accuracy.
- `diffusion_rl_training/`: RL train diffusion generators with RationalRewards reward feedback.
- `test_time_prompt_tuning/`: improve outputs at inference with critique-guided prompt refinement.
- `DATASETS_PLACEHOLDER.md`: dataset notes and schema references.

## 🙏 Acknowledgements

RationalRewards is built on top of open-source projects and communities, especially:

- [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT)
- [Edit-R1](https://github.com/PKU-YuanGroup/Edit-R1)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [diffusers](https://github.com/huggingface/diffusers)
- [vLLM](https://github.com/vllm-project/vllm)

## 📚 Citation

```bibtex
@article{rationalrewards2026,
  title   = {Think Before You Score: Reasoning Rewards Scale Visual
Generation at both Training and Test Time},
  author  = {Haozhe Wang, Cong Wei, Weiming Ren, Jiaming Liu, Fangzhen Lin, Wenhu Chen},
  journal = {arXiv preprint},
  year    = {2026}
}
```
