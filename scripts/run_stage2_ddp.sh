#!/bin/bash
# =============================================================================
# DDP Stage 2: 2-node distributed fine-tuning for LLaVA (instruction tuning)
# =============================================================================
# Usage:
#   1. Open two terminals, SSH into HPC on both
#   2. Allocate a GPU on each (ws-ia partition, 1 GPU per node):
#        Terminal A: salloc -p ws-ia -N1 -n12 --mem=64G -w ws-l4-XXX
#        Terminal B: salloc -p ws-ia -N1 -n12 --mem=64G -w ws-l4-YYY
#   3. Update master_node and worker_node below with actual hostnames
#   4. Run on BOTH terminals:
#        conda activate llava
#        bash scripts/run_stage2_ddp.sh
# =============================================================================

set -eo pipefail

# ======================== USER CONFIGURATION ========================
# Set these to the hostnames of your two allocated nodes (from `hostname`)
master_node=ws-l4-003
worker_node=ws-l4-006

MASTER_PORT=29500
# ============================ END CONFIG ============================

# Paths (resolved relative to this script's location)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAVA_DIR=$PROJECT_DIR/LLaVA
MODEL_DIR=$PROJECT_DIR/models/llava-v1.5-7b
DATA_PATH=$LLAVA_DIR/playground/data/llava_instruct_10k.json
IMAGE_FOLDER=$LLAVA_DIR/playground/data/coco/train2014
OUTPUT_DIR=$PROJECT_DIR/checkpoints/llava-stage2-ddp-2gpu
LOG_DIR=$PROJECT_DIR/logs

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
source "$PROJECT_DIR/scripts/run_logging.sh"

# NCCL settings for cross-node communication over Ethernet
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Determine node rank from hostname
CURRENT_HOST=$(hostname)
if [ "$CURRENT_HOST" == "$master_node" ]; then
    NODE_RANK=0
    echo "=========================================="
    echo "Running as MASTER node (rank 0)"
    echo "=========================================="
elif [ "$CURRENT_HOST" == "$worker_node" ]; then
    NODE_RANK=1
    echo "=========================================="
    echo "Running as WORKER node (rank 1)"
    echo "=========================================="
else
    echo "ERROR: Current host '$CURRENT_HOST' does not match"
    echo "       master_node='$master_node' or worker_node='$worker_node'"
    echo "Please update the master_node and worker_node variables in this script."
    exit 1
fi

echo "  Master: $master_node | Worker: $worker_node | Rank: $NODE_RANK"
echo "  Model  : $MODEL_DIR"
echo "  Data   : $DATA_PATH"
echo "  Images : $IMAGE_FOLDER"
nvidia-smi -L

setup_run_logging "stage2_ddp_lora" "Stage 2 DDP LoRA (rank $NODE_RANK)" 16 1 2
trap cleanup_run_logging EXIT
echo "  Output : $OUTPUT_DIR"
echo "  Log    : $TRAIN_LOG"

cd $LLAVA_DIR

TRAIN_SCRIPT=llava/train/train.py
python -c "import flash_attn" 2>/dev/null && TRAIN_SCRIPT=llava/train/train_mem.py
echo "Using: $TRAIN_SCRIPT"
start_run_logging

echo "=== Starting DDP Stage 2 fine-tuning (2 nodes x 1 GPU) ==="

set +e
torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$master_node \
    --master_port=$MASTER_PORT \
    $TRAIN_SCRIPT \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --model_name_or_path $MODEL_DIR \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
finish_run_logging "$TRAIN_EXIT"

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "=== DDP Stage 2 training failed on $CURRENT_HOST (rank $NODE_RANK, exit code: $TRAIN_EXIT) ==="
    exit "$TRAIN_EXIT"
fi

echo "=== DDP Stage 2 training complete on $CURRENT_HOST ==="
