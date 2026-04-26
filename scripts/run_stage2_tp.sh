#!/bin/bash
# =============================================================================
# Stage 2 Tensor Parallel training for LLaVA (Rassul / Omar — third strategy)
# =============================================================================
# Splits each LlamaDecoderLayer's attention + MLP linear sublayers across two
# nodes via torch.distributed.tensor.parallel. Each forward/backward triggers
# ~6 cross-node AllReduces per layer per pass — 384 cross-node collectives per
# step on a 32-layer Vicuna-7B.
#
# Trains the MLP projection only (Stage 1-style). LoRA is intentionally
# disabled — LoRA + TP-sharded base linears is fragile in PyTorch 2.1.2.
#
# Usage:
#   1. Allocate one GPU node in two terminals (ws-ia partition).
#   2. Set MASTER_NODE / WORKER_NODE to the two hostnames.
#   3. Run this script on both nodes within ~30 s of each other.
# =============================================================================

set -eo pipefail

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate llava
fi

master_node="${MASTER_NODE:-ws-l4-002}"
worker_node="${WORKER_NODE:-ws-l4-009}"
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
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/checkpoints/llava-stage2-tp}"
export LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WORLD_SIZE="${WORLD_SIZE:-2}"
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
    echo "ERROR: '$CURRENT_HOST' doesn't match MASTER_NODE='$master_node' or WORKER_NODE='$worker_node'."
    exit 1
fi

setup_run_logging "stage2_tp" "Stage 2 Tensor Parallel (rank $NODE_RANK)" "$PER_DEVICE_BATCH" "$GRAD_ACCUM" "$WORLD_SIZE"
trap cleanup_run_logging EXIT

echo "=========================================="
echo "Stage 2 Tensor Parallel (projection-only)"
echo "=========================================="
echo "  Master  : $master_node"
echo "  Worker  : $worker_node"
echo "  Rank    : $NODE_RANK"
echo "  Model   : $MODEL_DIR"
echo "  Data    : $DATA_PATH"
echo "  Output  : $OUTPUT_DIR"
echo "  Effective batch: $((PER_DEVICE_BATCH * GRAD_ACCUM * WORLD_SIZE))"
echo "  LR/Warmup: $LEARNING_RATE / $WARMUP_RATIO"
nvidia-smi -L

cd "$LLAVA_DIR"
start_run_logging

set +e
python -m torch.distributed.run \
    --nproc_per_node=1 \
    --nnodes="$WORLD_SIZE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$master_node" \
    --master_port="$MASTER_PORT" \
    llava/train/train_tp_dist.py \
    --model_name_or_path "$MODEL_DIR" \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --logging_steps "$LOGGING_STEPS" \
    --model_max_length 2048 \
    --gradient_checkpointing \
    --lazy_preprocess \
    --bf16 \
    2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

finish_run_logging "$TRAIN_EXIT"

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "=== Stage 2 TP failed on $CURRENT_HOST (rank $NODE_RANK, exit $TRAIN_EXIT) ==="
    exit "$TRAIN_EXIT"
fi

echo "=== Stage 2 TP complete on $CURRENT_HOST ==="
