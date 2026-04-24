#!/bin/bash
# Conservative 2-node Stage 2 pipeline-parallel LoRA launcher.
#
# This keeps the same LoRA trainer as run_stage2_pipe_lora.sh, but uses the
# lower-memory configuration that successfully reached training steps on
# 2 x RTX 5000 Ada nodes.

set -eo pipefail

export PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-16}"
export NUM_MICRO_BATCHES="${NUM_MICRO_BATCHES:-1}"
export SPLIT_LAYER="${SPLIT_LAYER:-12}"
export DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/run_stage2_pipe_lora.sh"
