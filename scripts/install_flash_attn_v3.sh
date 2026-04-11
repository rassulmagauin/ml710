#!/bin/bash
#SBATCH --job-name=install-flash-attn
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rassul.magauin/ml710/logs/install_flash_attn_v3_%j.log

eval "$(conda shell.bash hook)"
conda activate llava

export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

nvidia-smi
nvcc --version

# Fix missing pkg_resources (v82+ removed it)
pip install "setuptools<70" 2>&1

# Now install flash-attn
pip install flash-attn --no-build-isolation 2>&1

echo "=== Result ==="
python -c "import flash_attn; print(f'flash-attn version: {flash_attn.__version__}')" 2>&1
