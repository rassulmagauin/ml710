#!/bin/bash
# =============================================================================
# Stage 2 ZeRO-3 LoRA tuned for higher GPU utilization on 2 x RTX 5000 Ada 32 GB
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/vicuna-7b-v1.5}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/checkpoints/llava-stage2-zero3-vicuna-lora-util}"
export ZERO_CONFIG="${ZERO_CONFIG:-$PROJECT_DIR/configs/zero3_bounded_async.json}"

# Increase per-step GPU work while preserving the current effective batch of 8.
export PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"

# Reduce dataloader idle time and trainer-side logging overhead.
export DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
export LOGGING_STEPS="${LOGGING_STEPS:-5}"

# Keep the more stable ZeRO-3 Vicuna optimization defaults unless overridden.
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-2e-5}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
export OPTIM="${OPTIM:-adamw_torch}"

exec bash "$SCRIPT_DIR/run_stage2_zero3_lora.sh"
