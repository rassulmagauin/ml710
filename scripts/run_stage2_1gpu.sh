#!/bin/bash
#SBATCH --job-name=llava-stage2-1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/stage2_1gpu_%j.log

# Activate environment
eval "$(conda shell.bash hook)"
conda activate llava
set -eo pipefail
# Set paths (resolved relative to this script's location)
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LLAVA_DIR=$PROJECT_DIR/LLaVA
export MODEL_DIR=$PROJECT_DIR/models/llava-v1.5-7b
export DATA_PATH=$LLAVA_DIR/playground/data/llava_instruct_10k.json
export IMAGE_FOLDER=$LLAVA_DIR/playground/data/coco/train2014
export OUTPUT_DIR=$PROJECT_DIR/checkpoints/llava-stage2-1gpu
export LOG_DIR=$PROJECT_DIR/logs

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
# Shared logging wrapper: defines per-run log paths and exports metadata
# used by the monitor/summarizer helpers.
source "$PROJECT_DIR/scripts/run_logging.sh"
setup_run_logging "stage2_1gpu_lora" "Stage 2 1-GPU LoRA" 1 1 1
# Ensure background GPU/RAM monitors are cleaned up if the script exits early.
trap cleanup_run_logging EXIT

# Print GPU info
nvidia-smi
echo "=== Starting single-GPU Stage 2 fine-tuning ==="
echo "    Model  : $MODEL_DIR"
echo "    Data   : $DATA_PATH"
echo "    Images : $IMAGE_FOLDER"
echo "    Output : $OUTPUT_DIR"
echo "    Log    : $TRAIN_LOG"

cd $LLAVA_DIR
# Use flash_attn-backed train_mem.py if available, else eager attention train.py.
TRAIN_SCRIPT=llava/train/train.py
python -c "import flash_attn" 2>/dev/null && TRAIN_SCRIPT=llava/train/train_mem.py
echo "Using: $TRAIN_SCRIPT"
# Start external GPU/RAM sampling before launching training.
start_run_logging

# Stage 2: LoRA fine-tune — low-rank adapters on the LLM, full MLP projection update.
# CLIP vision encoder stays frozen throughout.
set +e
python "$TRAIN_SCRIPT" \
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
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
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
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
# Stop monitors, compute summary metrics, and append this run to the CSV table.
finish_run_logging "$TRAIN_EXIT"

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "=== Stage 2 single-GPU training failed (exit code: $TRAIN_EXIT) ==="
    exit "$TRAIN_EXIT"
fi

echo "=== Stage 2 single-GPU training complete ==="
