#!/bin/bash
# =============================================================================
# DDP Baseline: 2-node distributed training for LLaVA Stage 1
# =============================================================================
# Usage:
#   1. Open two terminals, SSH into HPC on both
#   2. Allocate a GPU on each:
#        Terminal A: salloc -p ws-ia -N1 -n12 --mem=64G -w ws-l4-XXX
#        Terminal B: salloc -p ws-ia -N1 -n12 --mem=64G -w ws-l4-YYY
#   3. Update master_node and worker_node below with actual hostnames
#   4. Run on BOTH terminals:
#        conda activate llava
#        bash scripts/run_baseline_ddp.sh
# =============================================================================

set -e

# ======================== USER CONFIGURATION ========================
# Modify these to match your allocated nodes (from `hostname` on each)
master_node=ws-l4-003
worker_node=ws-l4-006

MASTER_PORT=29500
# ============================ END CONFIG ============================

# Paths
PROJECT_DIR=/home/rassul.magauin/ml710
LLAVA_DIR=$PROJECT_DIR/LLaVA
MODEL_DIR=$PROJECT_DIR/models/vicuna-7b-v1.5
DATA_PATH=$LLAVA_DIR/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_10k.json
IMAGE_FOLDER=$LLAVA_DIR/playground/data/LLaVA-Pretrain
OUTPUT_DIR=$PROJECT_DIR/checkpoints/llava-v1.5-7b-pretrain-ddp-2gpu

# NCCL settings for cross-node communication
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
nvidia-smi -L

cd $LLAVA_DIR

# Use eager attention (no flash-attn)
TRAIN_SCRIPT=llava/train/train.py
python -c "import flash_attn" 2>/dev/null && TRAIN_SCRIPT=llava/train/train_mem.py
echo "Using: $TRAIN_SCRIPT"

echo "=== Starting DDP baseline (2 nodes x 1 GPU, Stage 1) ==="

torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$master_node \
    --master_port=$MASTER_PORT \
    $TRAIN_SCRIPT \
    --deepspeed $PROJECT_DIR/configs/zero0.json \
    --model_name_or_path $MODEL_DIR \
    --version plain \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
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
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

echo "=== DDP baseline training complete on $CURRENT_HOST ==="
