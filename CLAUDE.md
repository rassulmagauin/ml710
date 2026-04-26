
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML710 course project: Parallelize LLaVA (Large Language and Vision Assistant) — a multimodal model from "Visual Instruction Tuning" (NeurIPS 2023). Team of 3 students, each implementing at least 1 non-trivial parallel/distributed strategy, plus a naive DDP baseline.

**Deadline: April 29, 2026. Deliverable: max 30 slides (PPT/PDF).**

Current repo state:
- **Stage 1** scripts are kept as the original baseline / historical reference.
- **Stage 2** scripts are the main benchmark path: start from pre-trained `llava-v1.5-7b`, fine-tune with LoRA on a 10K instruction-tuning subset, and compare parallelization techniques on that workload.
- **LLaVA is now vendored** under `LLaVA/` (no longer a submodule), since the project ships modified pipeline-parallel training entry points (`LLaVA/llava/train/train_pipe*.py`).
- Implemented strategies: **DDP (baseline)**, **ZeRO-2**, **ZeRO-3**, **Pipeline (manual 2-stage)**. Results in `ZERO2_RESULTS.md` + `results_explanation.md`.
- The repo-level experiment wrappers (`scripts/run_logging.sh`, `scripts/summarize_run.py`) stay outside the upstream `LLaVA` code.

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
├── LLaVA/                              # Vendored LLaVA source (not a submodule)
│   ├── llava/train/
│   │   ├── train.py                    # Stock entry point (eager attention)
│   │   ├── train_mem.py                # Same, with flash_attention_2
│   │   ├── train_pipe.py               # Manual pipeline-parallel trainer (single node)
│   │   ├── train_pipe_dist.py          # 2-node distributed pipeline (Gloo)
│   │   ├── train_pipe_8gpu.py          # 8-GPU pipeline variant
│   │   └── llava_trainer.py            # Custom HuggingFace Trainer
│   └── playground/data/                # Training data (see SETUP.md)
│       ├── LLaVA-Pretrain/             # Stage 1 data: 558K image-caption pairs
│       ├── llava_instruct_150k.json    # Stage 2 single-source data (COCO 2014 only)
│       ├── llava_instruct_10k.json     # 10K subset for experiments (seed=42)
│       └── coco/train2014/             # COCO 2014 train images for Stage 2
├── models/
│   ├── vicuna-7b-v1.5/                 # Base LLM (used for Stage 1 baseline)
│   └── llava-v1.5-7b/                  # Pre-trained LLaVA (Stage 2 start point)
├── scripts/                            # Our parallelism experiment scripts
│   ├── download_stage2_data.sh         # Downloads Stage 2 data + 10K subset
│   ├── create_subset.py                # Random subset utility
│   ├── run_baseline_1gpu.sh            # Stage 1 single-GPU (historical)
│   ├── run_baseline_ddp.sh             # Stage 1 DDP (historical)
│   ├── run_stage2_1gpu.sh              # Stage 2 single-GPU LoRA baseline
│   ├── run_stage2_ddp.sh               # Stage 2 LoRA, DDP, 2 nodes x 1 GPU
│   ├── run_stage2_zero2_lora.sh        # Stage 2 LoRA, DeepSpeed ZeRO-2
│   ├── run_stage2_zero2_lora_util.sh   # ZeRO-2 utilization-tuned
│   ├── run_stage2_zero3_lora.sh        # Stage 2 LoRA, DeepSpeed ZeRO-3
│   ├── run_stage2_zero3_lora_util.sh   # ZeRO-3 with bounded async comm
│   ├── run_stage2_pipe_lora.sh         # Stage 2 LoRA, manual pipeline parallel
│   ├── run_stage2_pipe_lora_safe.sh    # Pipeline launcher with conservative defaults
│   ├── run_logging.sh                  # Shared logging wrapper
│   ├── summarize_run.py                # Per-run + cumulative CSV/plot summarizer
│   └── plot_statistical_efficiency.py  # Loss-vs-samples plotting
├── configs/                            # DeepSpeed configurations
│   ├── zero0.json                      # ZeRO-0 (pure DDP)
│   ├── zero2.json                      # ZeRO-2
│   ├── zero3.json                      # ZeRO-3
│   └── zero3_bounded_async.json        # ZeRO-3 with capped allgather/reduce buckets
├── checkpoints/                        # Training output
├── logs/runs/<user>/<run_id>/          # Per-run artifacts (train/gpu/ram/summary)
├── SETUP.md                            # Reproduction guide
├── SYSTEM_REQUIREMENTS.md              # Per-strategy hardware/software requirements
├── ZERO2_RESULTS.md                    # ZeRO-2/3 + Pipeline result tables
├── ZERO3_NOTES.md                      # ZeRO-3 launcher notes
├── results_explanation.md              # Narrative analysis of results
└── 2026 Labs/                          # Course lab materials (reference)
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

Current wrapper policy:
- Keep upstream `LLaVA` modular. Prefer repo-level shell/Python wrappers for experiment control, logging, and summarization.
- Use `scripts/run_logging.sh` and `scripts/summarize_run.py` from training scripts instead of editing upstream `llava/train/*.py` just to collect benchmark metrics.
- Current summaries include:
  - end-to-end runtime
  - trainer runtime / throughput
  - final and average logged loss
  - per-GPU memory/utilization snapshots
  - per-run artifacts under `logs/runs/<user>/<run_id>/`

## Parallelism Strategies (Implemented)

Baseline DDP (ZeRO-0) is the reference; the three implemented strategies are:

- **Pipeline Parallelism** — manual 2-stage split of the Vicuna decoder across 2 nodes (`LLaVA/llava/train/train_pipe_dist.py`). Best balanced split is layer 12; uneven splits (8 or 16) OOM the heavier rank.
- **DeepSpeed ZeRO-2** — sharded optimizer states + gradients. ~1.28× faster than DDP at matched effective batch.
- **DeepSpeed ZeRO-3** — sharded optimizer states + gradients + parameters. On this Ethernet-bound setup it is ~13× slower than DDP because of per-layer cross-node AllGather of bf16 weights.

Note: ZeRO-2 and ZeRO-3 are both formally classified as "ZeRO-DP" (data parallelism with sharded state). Whether they count as 2 distinct strategies for the per-student-strategy rule is being clarified with course staff.

See `ZERO2_RESULTS.md` for the result table and `results_explanation.md` for the narrative analysis.

## Experiment Guidelines

- Keep each training run **< 1 hour** (use subsets via `scripts/create_subset.py`)
- Report: throughput (samples/sec), statistical efficiency (loss at N samples), Goodput, GPU utilization, memory usage
- Compare all strategies against single-GPU and DDP baselines
