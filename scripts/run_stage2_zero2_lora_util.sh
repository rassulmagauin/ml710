#!/bin/bash
# =============================================================================
# Stage 2 ZeRO-2 LoRA tuned for higher GPU utilization on 2 x RTX 5000 Ada 32 GB
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/vicuna-7b-v1.5}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/checkpoints/llava-stage2-zero2-vicuna-lora-util}"

# Push more work into each step while keeping the same effective batch of 16.
export PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"

# Reduce input stalls and logging overhead.
export DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
export LOGGING_STEPS="${LOGGING_STEPS:-5}"

# Keep the current ZeRO-2 Vicuna optimization recipe unless explicitly overridden.
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-2e-5}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
export OPTIM="${OPTIM:-adamw_torch}"

exec bash "$SCRIPT_DIR/run_stage2_zero2_lora.sh"
