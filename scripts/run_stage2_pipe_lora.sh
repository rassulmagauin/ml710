#!/bin/bash
# =============================================================================
# Stage 2 Pipeline Parallelism LoRA fine-tuning for LLaVA — 2-node ws-ia
#
# One process per node (each with 1 GPU, ws-ia partition).
# Rank 0 (master): vision tower + mm_projector + embed_tokens + layers 0–15
# Rank 1 (worker): layers 16–31 + RMSNorm + lm_head + loss
#
# Run on BOTH nodes simultaneously:
#   Node A:  MASTER_NODE=ws-l4-XXX WORKER_NODE=ws-l4-YYY bash scripts/run_stage2_pipe_lora.sh
#   Node B:  MASTER_NODE=ws-l4-XXX WORKER_NODE=ws-l4-YYY bash scripts/run_stage2_pipe_lora.sh
# =============================================================================

set -eo pipefail

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate llava
fi

# ======================== USER CONFIGURATION ========================
master_node="${MASTER_NODE:-ws-l4-002}"
worker_node="${WORKER_NODE:-ws-l4-003}"
MASTER_PORT="${MASTER_PORT:-29501}"
# ============================ END CONFIG ============================

export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LLAVA_DIR="$PROJECT_DIR/LLaVA"
export LOCAL_MODEL_DIR="$PROJECT_DIR/models/llava-v1.5-7b"
DEFAULT_MODEL_SOURCE="liuhaotian/llava-v1.5-7b"
if [ -n "${MODEL_DIR:-}" ]; then
    export MODEL_DIR
elif [ -d "$LOCAL_MODEL_DIR" ]; then
    export MODEL_DIR="$LOCAL_MODEL_DIR"
else
    export MODEL_DIR="$DEFAULT_MODEL_SOURCE"
fi

export DATA_PATH="$LLAVA_DIR/playground/data/llava_instruct_10k.json"
export IMAGE_FOLDER="$LLAVA_DIR/playground/data/coco/train2014"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/checkpoints/llava-stage2-pipe-lora-dist}"
export LOG_DIR="$PROJECT_DIR/logs"

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_MICRO_BATCHES="${NUM_MICRO_BATCHES:-4}"
SPLIT_LAYER="${SPLIT_LAYER:-16}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-256}"
DIST_WORLD_SIZE=2
LOG_WORLD_SIZE=1

CURRENT_HOST=$(hostname)
if [ "$CURRENT_HOST" == "$master_node" ]; then
    NODE_RANK=0
elif [ "$CURRENT_HOST" == "$worker_node" ]; then
    NODE_RANK=1
else
    echo "ERROR: Current host '$CURRENT_HOST' does not match master_node='$master_node' or worker_node='$worker_node'."
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
source "$PROJECT_DIR/scripts/run_logging.sh"

export RUN_ID_SUFFIX="_rank${NODE_RANK}"
setup_run_logging "stage2_pipe_lora_dist" "Stage 2 Pipeline LoRA dist (rank $NODE_RANK)" "$PER_DEVICE_BATCH" "$GRAD_ACCUM" "$LOG_WORLD_SIZE"
trap cleanup_run_logging EXIT

echo "=========================================="
echo "Stage 2 Pipeline LoRA (2-node ws-ia)"
echo "=========================================="
echo "  Master     : $master_node"
echo "  Worker     : $worker_node"
echo "  Rank       : $NODE_RANK"
echo "  Split layer: $SPLIT_LAYER"
echo "  Model      : $MODEL_DIR"
echo "  Data       : $DATA_PATH"
echo "  Output     : $OUTPUT_DIR"
echo "  Log        : $TRAIN_LOG"
echo "  Effective batch: $((PER_DEVICE_BATCH * GRAD_ACCUM))"
nvidia-smi -L

start_run_logging

set +e
RANK=$NODE_RANK \
WORLD_SIZE=$DIST_WORLD_SIZE \
MASTER_ADDR=$master_node \
MASTER_PORT=$MASTER_PORT \
python "$LLAVA_DIR/llava/train/train_pipe_dist.py" \
    --model_name_or_path      "$MODEL_DIR" \
    --data_path               "$DATA_PATH" \
    --image_folder            "$IMAGE_FOLDER" \
    --output_dir              "$OUTPUT_DIR" \
    --num_train_epochs        "$NUM_TRAIN_EPOCHS" \
    --per_device_batch_size   "$PER_DEVICE_BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_micro_batches       "$NUM_MICRO_BATCHES" \
    --split_layer             "$SPLIT_LAYER" \
    --learning_rate           "$LEARNING_RATE" \
    --mm_projector_lr         "$MM_PROJECTOR_LR" \
    --warmup_ratio            "$WARMUP_RATIO" \
    --bf16 \
    --lora_r                  "$LORA_R" \
    --lora_alpha              "$LORA_ALPHA" \
    --dataloader_num_workers  "$DATALOADER_WORKERS" \
    --logging_steps           "$LOGGING_STEPS" \
    2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

finish_run_logging "$TRAIN_EXIT"

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "=== Stage 2 Pipeline LoRA FAILED on $CURRENT_HOST (rank $NODE_RANK, exit $TRAIN_EXIT) ==="
    exit "$TRAIN_EXIT"
fi

echo "=== Stage 2 Pipeline LoRA complete on $CURRENT_HOST ==="
