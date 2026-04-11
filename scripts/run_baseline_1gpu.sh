#!/bin/bash
#SBATCH --job-name=llava-baseline-1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/rassul.magauin/ml710/logs/baseline_1gpu_%j.log

# Activate environment
eval "$(conda shell.bash hook)"
conda activate llava

# Set paths
export PROJECT_DIR=/home/rassul.magauin/ml710
export LLAVA_DIR=$PROJECT_DIR/LLaVA
export MODEL_DIR=$PROJECT_DIR/models/vicuna-7b-v1.5
export DATA_PATH=$LLAVA_DIR/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_10k.json
export IMAGE_FOLDER=$LLAVA_DIR/playground/data/LLaVA-Pretrain
export OUTPUT_DIR=$PROJECT_DIR/checkpoints/llava-v1.5-7b-pretrain-baseline-1gpu

# Print GPU info
nvidia-smi
echo "=== Starting single-GPU baseline (Stage 1: pretrain projection only) ==="

cd $LLAVA_DIR

# Stage 1: Pre-training (only trains the mm_projector)
# This is fast because only the projection layer W is trained,
# vision encoder and LLM are frozen.
# Use train.py (eager attn) if flash-attn not installed, train_mem.py if it is
TRAIN_SCRIPT=llava/train/train.py
python -c "import flash_attn" 2>/dev/null && TRAIN_SCRIPT=llava/train/train_mem.py
echo "Using: $TRAIN_SCRIPT"

python $TRAIN_SCRIPT \
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

echo "=== Baseline training complete ==="
