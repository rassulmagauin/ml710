#!/bin/bash
# =============================================================================
# Stage 2 ZeRO-2 random-init "pretraining": 2-node distributed training
# Trains the LLM + mm_projector from random initialization while keeping
# the CLIP vision tower frozen/pretrained.
# =============================================================================

set -eo pipefail

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate llava
fi

# ======================== USER CONFIGURATION ========================
master_node=ws-l4-005
worker_node=ws-l4-006
MASTER_PORT=29500
# ============================ END CONFIG ============================

export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LLAVA_DIR="$PROJECT_DIR/LLaVA"
export LOCAL_MODEL_DIR="$PROJECT_DIR/models/llava-v1.5-7b"
DEFAULT_MODEL_SOURCE="liuhaotian/llava-v1.5-7b"
if [ -d "$LOCAL_MODEL_DIR" ]; then
    export MODEL_DIR="$LOCAL_MODEL_DIR"
else
    export MODEL_DIR="${MODEL_DIR:-$DEFAULT_MODEL_SOURCE}"
fi
export DATA_PATH="$LLAVA_DIR/playground/data/llava_instruct_10k.json"
export IMAGE_FOLDER="$LLAVA_DIR/playground/data/coco/train2014"
export OUTPUT_DIR="$PROJECT_DIR/checkpoints/llava-stage2-zero2-random-pretrain"
export LOG_DIR="$PROJECT_DIR/logs"
export ZERO_CONFIG="$PROJECT_DIR/configs/zero2.json"

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
WORLD_SIZE=2

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
source "$PROJECT_DIR/scripts/run_logging.sh"

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

CURRENT_HOST=$(hostname)
if [ "$CURRENT_HOST" == "$master_node" ]; then
    NODE_RANK=0
elif [ "$CURRENT_HOST" == "$worker_node" ]; then
    NODE_RANK=1
else
    echo "ERROR: Current host '$CURRENT_HOST' does not match master_node='$master_node' or worker_node='$worker_node'."
    exit 1
fi

setup_run_logging "stage2_zero2_random_pretrain" "Stage 2 ZeRO-2 Random Pretrain (rank $NODE_RANK)" "$PER_DEVICE_BATCH" "$GRAD_ACCUM" "$WORLD_SIZE"
trap cleanup_run_logging EXIT

echo "=========================================="
echo "Stage 2 ZeRO-2 random-init pretraining"
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
    --model_name_or_path "$MODEL_DIR" \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_projector_lr 1e-4 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --random_init_text_backbone True \
    --random_init_mm_projector True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

finish_run_logging "$TRAIN_EXIT"

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "=== Stage 2 ZeRO-2 random-init pretraining failed on $CURRENT_HOST (rank $NODE_RANK, exit code: $TRAIN_EXIT) ==="
    exit "$TRAIN_EXIT"
fi

echo "=== Stage 2 ZeRO-2 random-init pretraining complete on $CURRENT_HOST ==="
