#!/bin/bash
# =============================================================================
# Stage 2 ZeRO-3 LoRA fine-tuning for LLaVA
# =============================================================================
# Usage:
#   1. Allocate one GPU node in two terminals.
#   2. Run this script on both nodes with matching env vars.
#   3. Override MASTER_NODE / WORKER_NODE if your allocation differs.
# =============================================================================

set -eo pipefail

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate llava
fi

master_node="${MASTER_NODE:-ws-l4-005}"
worker_node="${WORKER_NODE:-ws-l4-006}"
MASTER_PORT="${MASTER_PORT:-29500}"

export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LLAVA_DIR="$PROJECT_DIR/LLaVA"
export LOCAL_MODEL_DIR="$PROJECT_DIR/models/llava-v1.5-7b"
DEFAULT_MODEL_SOURCE="liuhaotian/llava-v1.5-7b"
if [ -d "$LOCAL_MODEL_DIR" ]; then
    export MODEL_DIR="${MODEL_DIR:-$LOCAL_MODEL_DIR}"
else
    export MODEL_DIR="${MODEL_DIR:-$DEFAULT_MODEL_SOURCE}"
fi
export DATA_PATH="${DATA_PATH:-$LLAVA_DIR/playground/data/llava_instruct_10k.json}"
export IMAGE_FOLDER="${IMAGE_FOLDER:-$LLAVA_DIR/playground/data/coco/train2014}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/checkpoints/llava-stage2-zero3-lora}"
export LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
export ZERO_CONFIG="${ZERO_CONFIG:-$PROJECT_DIR/configs/zero3.json}"

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
PER_DEVICE_EVAL_BATCH="${PER_DEVICE_EVAL_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
OPTIM="${OPTIM:-adamw_torch}"
SAVE_STEPS="${SAVE_STEPS:-24000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
WORLD_SIZE="${WORLD_SIZE:-2}"
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-256}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
source "$PROJECT_DIR/scripts/run_logging.sh"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

CURRENT_HOST="$(hostname)"
if [ "$CURRENT_HOST" = "$master_node" ]; then
    NODE_RANK=0
elif [ "$CURRENT_HOST" = "$worker_node" ]; then
    NODE_RANK=1
else
    echo "ERROR: Current host '$CURRENT_HOST' does not match MASTER_NODE='$master_node' or WORKER_NODE='$worker_node'."
    exit 1
fi

setup_run_logging "stage2_zero3_lora" "Stage 2 ZeRO-3 LoRA (rank $NODE_RANK)" "$PER_DEVICE_BATCH" "$GRAD_ACCUM" "$WORLD_SIZE"
trap cleanup_run_logging EXIT

echo "=========================================="
echo "Stage 2 ZeRO-3 LoRA fine-tuning"
echo "=========================================="
echo "  Master     : $master_node"
echo "  Worker     : $worker_node"
echo "  Rank       : $NODE_RANK"
echo "  Model      : $MODEL_DIR"
echo "  Data       : $DATA_PATH"
echo "  Images     : $IMAGE_FOLDER"
echo "  Output     : $OUTPUT_DIR"
echo "  Zero config: $ZERO_CONFIG"
echo "  Log        : $TRAIN_LOG"
echo "  Effective batch: $((PER_DEVICE_BATCH * GRAD_ACCUM * WORLD_SIZE))"
echo "  LR/Warmup: $LEARNING_RATE / $WARMUP_RATIO"
echo "  Optim/Scheduler: $OPTIM / $LR_SCHEDULER_TYPE"
echo "  Vision tower: frozen by Stage 2 LoRA defaults"
nvidia-smi -L

cd "$LLAVA_DIR"
TRAIN_SCRIPT=llava/train/train.py
python -c "import flash_attn" 2>/dev/null && TRAIN_SCRIPT=llava/train/train_mem.py
echo "Using: $TRAIN_SCRIPT"
start_run_logging

set +e
python -m torch.distributed.run \
    --nproc_per_node=1 \
    --nnodes="$WORLD_SIZE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$master_node" \
    --master_port="$MASTER_PORT" \
    "$TRAIN_SCRIPT" \
    --deepspeed "$ZERO_CONFIG" \
    --lora_enable True \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --mm_projector_lr "$MM_PROJECTOR_LR" \
    --model_name_or_path "$MODEL_DIR" \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay 0. \
    --warmup_ratio "$WARMUP_RATIO" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --optim "$OPTIM" \
    --logging_steps "$LOGGING_STEPS" \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers "$DATALOADER_WORKERS" \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

finish_run_logging "$TRAIN_EXIT"

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "=== Stage 2 ZeRO-3 LoRA failed on $CURRENT_HOST (rank $NODE_RANK, exit code: $TRAIN_EXIT) ==="
    exit "$TRAIN_EXIT"
fi

echo "=== Stage 2 ZeRO-3 LoRA complete on $CURRENT_HOST ==="
