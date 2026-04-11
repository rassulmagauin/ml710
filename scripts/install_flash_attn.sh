#!/bin/bash
#SBATCH --job-name=install-flash-attn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rassul.magauin/ml710/logs/install_flash_attn_%j.log

eval "$(conda shell.bash hook)"
conda activate llava

nvidia-smi
echo "CUDA version:"
nvcc --version 2>/dev/null || echo "nvcc not found, using pip wheel"

pip install flash-attn --no-build-isolation 2>&1
echo "=== flash-attn installation complete ==="
python -c "import flash_attn; print(f'flash-attn version: {flash_attn.__version__}')"
