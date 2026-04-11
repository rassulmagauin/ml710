# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML710 course project: Parallelize LLaVA (Large Language and Vision Assistant) — a multimodal model from "Visual Instruction Tuning" (NeurIPS 2023). Team of 3 students, each implementing at least 1 non-trivial parallel/distributed strategy, plus a naive DDP baseline.

**Deadline: April 29, 2026. Deliverable: max 30 slides (PPT/PDF).**

## LLaVA Architecture

LLaVA connects a CLIP ViT-L/14-336 vision encoder to a Vicuna-7B LLM via a 2-layer MLP projection:
- **Vision encoder** (CLIP): frozen during all training
- **Projection** (MLP 2x GELU): small, trained in both stages
- **LLM** (Vicuna 7B): frozen in Stage 1, fine-tuned in Stage 2

Two-stage training:
1. **Stage 1 (Pre-train)**: Only projection layer trained on 558K image-caption pairs (fast, ~4h on 8×A100)
2. **Stage 2 (Fine-tune)**: Projection + LLM trained on 665K instruction data (slower, ~10h on 8×A100)

## Repository Structure

```
ml710/
├── LLaVA/                        # Official LLaVA repo (haotian-liu/LLaVA)
│   ├── llava/train/train.py      # Main training script (eager attention)
│   ├── llava/train/train_mem.py  # Training with flash_attention_2
│   ├── llava/train/llava_trainer.py  # Custom HuggingFace Trainer
│   ├── scripts/                  # Original training shell scripts
│   │   ├── v1_5/                 # LLaVA v1.5 scripts (our target)
│   │   ├── zero2.json            # DeepSpeed ZeRO Stage 2 config
│   │   └── zero3.json            # DeepSpeed ZeRO Stage 3 config
│   └── playground/data/          # Training data
│       ├── LLaVA-Pretrain/       # Stage 1: 558K image-caption pairs + images
│       └── llava_v1_5_mix665k.json  # Stage 2: 665K instruction tuning data
├── models/
│   ├── vicuna-7b-v1.5/           # Base LLM for training from scratch
│   └── llava-v1.5-7b/            # Pre-trained LLaVA checkpoint
├── scripts/                      # Our parallelism experiment scripts
├── configs/                      # DeepSpeed/training configs
│   └── zero0.json                # ZeRO Stage 0 (pure DDP, no sharding)
├── checkpoints/                  # Training output
├── logs/                         # SLURM job logs
└── 2026 Labs/                    # Course lab materials (reference)
```

## HPC Environment

- **GPU partition**: 4 nodes (gpu-01 to gpu-04), each with **8× NVIDIA RTX 5000 Ada (32GB)**, 128 CPUs, 765GB RAM
- **ws-ia partition**: ~116 nodes, each with 1× RTX 5000 Ada, 48 CPUs, 230GB RAM
- **Prefer `gpu` partition** for multi-GPU on single node (avoids cross-node communication overhead)
- CUDA 12.2 at `/usr/local/cuda-12.2`, driver 550.54.14
- Conda env: `llava` (Python 3.10, PyTorch 2.1.2, DeepSpeed 0.12.6, transformers 4.37.2)

## Key Commands

```bash
# Activate environment
eval "$(conda shell.bash hook)" && conda activate llava

# Set CUDA paths (needed for compilation on GPU nodes)
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH

# Single-GPU Stage 1 pretrain (baseline)
sbatch scripts/run_baseline_1gpu.sh

# DDP baseline (2 GPUs)
sbatch scripts/run_baseline_ddp.sh

# Interactive GPU session for debugging
salloc --partition=gpu --gpus=2 --cpus-per-task=16 --mem=128G --time=02:00:00

# Check job status
squeue -u rassul.magauin

# View training logs
tail -f logs/<logfile>.log
```

## Training Details

LLaVA uses HuggingFace Trainer + DeepSpeed under the hood. The entry point is `llava/train/train.py` which defines:
- `ModelArguments`: model path, vision tower, projector type
- `DataArguments`: data path, image folder, aspect ratio
- `TrainingArguments`: extends HF TrainingArguments with LoRA, quantization options

Training is launched via `deepspeed` launcher for multi-GPU, or plain `python` for single-GPU.

## Parallelism Strategies (Project Requirements)

Each student implements 1 non-trivial strategy. Baseline DDP doesn't count.
- **Pipeline Parallelism**: Split model stages (vision encoder → projection → LLM layers)
- **Tensor Parallelism**: Split transformer layers' attention/MLP across GPUs (Megatron-style)
- **FSDP / ZeRO**: Shard parameters, gradients, optimizer states across GPUs

## Experiment Guidelines

- Keep each training run **< 1 hour** (use subsets via `scripts/create_subset.py`)
- Report: throughput (samples/sec), statistical efficiency (loss at N samples), Goodput, GPU utilization, memory usage
- Compare all strategies against single-GPU and DDP baselines
