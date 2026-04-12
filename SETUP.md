# Setup & Reproduction Guide

Step-by-step instructions to reproduce the LLaVA baselines on the MBZUAI HPC.

---

## 1. SSH into the HPC

```bash
ssh YOUR_USERNAME@login-student-lab.mbzu.ae
```

You land on the **login node** — no GPUs here. Use `salloc` or `sbatch` to run anything heavy.

---

## 2. Clone the project repo

```bash
cd ~
git clone --recurse-submodules git@github.com:rassulmagauin/ml710.git
cd ml710
```

The `--recurse-submodules` flag pulls the LLaVA source into `LLaVA/`. If you forgot:

```bash
git submodule update --init --recursive
```

---

## 3. Create the conda environment

If you don't have conda yet, install [Miniconda](https://docs.anaconda.com/miniconda/install/) first.

```bash
# Create env
conda create -n llava python=3.10 -y
conda activate llava

# Install LLaVA dependencies
cd LLaVA
pip install --upgrade pip
pip install -e .

# Pin specific versions (LLaVA's defaults break newer setuptools/peft/accelerate)
pip install "setuptools<70"
pip install "peft==0.6.0" "accelerate==0.25.0"

cd ..
```

**Why these pins?**
- `setuptools<70` — newer versions removed `pkg_resources`, breaking many packages
- `peft==0.6.0` + `accelerate==0.25.0` — newer peft requires newer accelerate, but LLaVA pins `accelerate==0.21.0` which is missing `clear_device_cache`. This combination works.

**Verify**:
```bash
python -c "import torch, deepspeed, transformers, peft, accelerate; print(torch.__version__, deepspeed.__version__, transformers.__version__)"
# Expected: 2.1.2+cu121 0.12.6 4.37.2
```

**Skip flash-attn**: It takes 1+ hour to compile on this HPC and doesn't change correctness. The training scripts auto-detect and fall back to eager attention. If you really want it, see `scripts/install_flash_attn.sh`.

---

## 4. Download model weights

We use **vicuna-7b-v1.5** as the base LLM. You also need the CLIP vision encoder (downloaded automatically by HuggingFace at first run).

```bash
mkdir -p models && cd models

# Vicuna 7B base (~14GB)
git lfs install
git clone https://huggingface.co/lmsys/vicuna-7b-v1.5

# Optional: pre-trained LLaVA checkpoint (for evaluation/inference, ~14GB)
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b

cd ..
```

> **Note**: Use `git lfs install` first or you'll get tiny placeholder files. Verify with `du -sh models/vicuna-7b-v1.5` (should be ~13GB).

---

## 5. Download Stage 1 pre-training data

The Stage 1 dataset is `LLaVA-Pretrain` (558K image-caption pairs, ~24GB after extraction).

```bash
cd LLaVA/playground/data
git lfs install
git clone https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain

cd LLaVA-Pretrain
unzip images.zip   # extracts ~558K jpg files into folders like 00000/, 00001/, ...

cd ~/ml710
```

You should now have:
```
LLaVA/playground/data/LLaVA-Pretrain/
├── blip_laion_cc_sbu_558k.json     # captions
├── images.zip
└── 00000/  00001/  00002/  ...     # extracted images
```

---

## 6. Create a 10K data subset

Full dataset training takes ~43 hours per run, way too long. Each experiment must finish in < 1 hour, so we use a 10K sample subset:

```bash
python scripts/create_subset.py \
    --input  LLaVA/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --output LLaVA/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_10k.json \
    --n 10000 \
    --seed 42
```

> Use `seed=42` so everyone trains on the **same** subset and results are comparable.

---

## 7. Run the single-GPU baseline

This uses `sbatch` (SLURM batch job) — fire-and-forget, survives network disconnects.

```bash
# Edit paths if your username differs from rassul.magauin
sed -i "s|/home/rassul.magauin|$HOME|g" scripts/run_baseline_1gpu.sh

mkdir -p logs checkpoints
sbatch scripts/run_baseline_1gpu.sh
```

### Batch job parameters used (single-GPU)

```bash
#SBATCH --job-name=llava-baseline-1gpu
#SBATCH --partition=gpu          # 8x RTX 5000 Ada nodes (1 GPU per user max)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G                # 64GB host RAM (Vicuna-7B needs ~14GB GPU + room for activations)
#SBATCH --time=01:30:00          # 1.5h wall-clock budget
#SBATCH --output=logs/baseline_1gpu_%j.log
```

### Training hyperparameters

| Arg | Value | Note |
|---|---|---|
| `--per_device_train_batch_size` | 16 | Per-GPU batch |
| `--gradient_accumulation_steps` | 1 | No accumulation |
| `--learning_rate` | 1e-3 | Stage 1 standard LLaVA LR |
| `--lr_scheduler_type` | cosine | |
| `--warmup_ratio` | 0.03 | |
| `--num_train_epochs` | 1 | |
| `--bf16` | True | Mixed precision |
| `--tf32` | True | TF32 matmul on Ampere/Ada |
| `--gradient_checkpointing` | True | Lower memory, slower compute |
| `--model_max_length` | 2048 | |
| `--tune_mm_mlp_adapter` | True | Stage 1 = train projection only |
| `--mm_projector_type` | mlp2x_gelu | 2-layer MLP with GELU |
| `--mm_vision_select_layer` | -2 | Use second-to-last CLIP layer |
| `--vision_tower` | openai/clip-vit-large-patch14-336 | Auto-downloaded |

### Monitor progress

```bash
squeue -u $USER                            # job state
tail -f logs/baseline_1gpu_<JOBID>.log     # live log
```

Expected: ~46 minutes wall time, 625 training steps.

---

## 8. Run the DDP baseline (2 nodes x 1 GPU)

The HPC QOS limits the `gpu` partition to **1 GPU per user**. To do multi-GPU we use 2 separate `salloc` sessions on the `ws-ia` partition (each ws-ia node has 1 GPU).

### Step 1 — Allocate two nodes (in two separate terminals)

```bash
# Terminal A — master node:
salloc -p ws-ia -N1 -n12 -w ws-l4-005 --mem=64G
hostname    # note this, e.g. ws-l4-005

# Terminal B — worker node:
salloc -p ws-ia -N1 -n12 -w ws-l4-006 --mem=64G
hostname    # note this, e.g. ws-l4-006
```

> **Important**: Pick `ws-l4-XXX` nodes (avoid `ws-l4-001` per the lab notes). Avoid auto-allocation; the QOS only allows one auto job, so you must use `-w` to explicitly target different nodes.

### Step 2 — Edit the script with your node hostnames

```bash
# In scripts/run_baseline_ddp.sh, set:
master_node=ws-l4-005   # your terminal A
worker_node=ws-l4-006   # your terminal B
```

### Step 3 — Run the same script on **both** terminals

```bash
# Terminal A:
eval "$(conda shell.bash hook)" && conda activate llava
bash scripts/run_baseline_ddp.sh

# Terminal B (same command):
eval "$(conda shell.bash hook)" && conda activate llava
bash scripts/run_baseline_ddp.sh
```

> Start both within ~30 seconds of each other or NCCL will time out waiting for the second node.

### How the launch works

We use `torchrun` (not `deepspeed` launcher — DeepSpeed 0.12.6's launcher doesn't support `--no_ssh`/`--node_rank`):

```bash
torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \      # 0 on master, 1 on worker
    --master_addr=$master_node \
    --master_port=29500 \
    llava/train/train.py \
    --deepspeed configs/zero0.json \
    ... (same training args as 1-GPU baseline)
```

`zero0.json` is pure DDP (ZeRO Stage 0 = no parameter/gradient/optimizer sharding). Each node runs 1 process bound to its single GPU; gradients are all-reduced over the network via NCCL.

### NCCL environment variables (set in script)

```bash
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^lo,docker   # use any interface except loopback/docker
export NCCL_IB_DISABLE=1               # no InfiniBand on ws-ia
export NCCL_P2P_DISABLE=1              # cross-node, no P2P
```

Expected: ~25 minutes wall time, 313 training steps (half of 1-GPU because data is split across 2 GPUs).

---

## 9. Results so far

| Metric | 1 GPU | 2 GPU DDP | Speedup |
|---|---|---|---|
| Runtime | 46.5 min | 24.6 min | **1.89x** |
| Throughput | 3.585 samples/sec | 6.773 samples/sec | **1.89x** |
| Steps | 625 | 313 | — |
| Effective batch | 16 | 32 | — |
| Avg train loss | 2.7509 | 3.1735 | — |

See [README.md](README.md) for full discussion.

---

## Common gotchas

1. **`flash-attn` build hangs / fails**: Skip it. The scripts auto-fall-back to eager attention. Building from source needs `setuptools<70` and a GPU node, takes 1+ hour.

2. **`ImportError: cannot import name 'clear_device_cache'`**: peft/accelerate version mismatch. Run `pip install "peft==0.6.0" "accelerate==0.25.0"`.

3. **`QOSMaxGRESPerUser`** when using `sbatch --gpus=2`: The `gpu` partition limits 1 GPU/user. Use the 2-node `ws-ia` approach instead.

4. **NCCL connection timeout**: Both ranks must call `torchrun` within ~30s of each other. If you saw the error, kill both and re-run.

5. **`Node count specification invalid` on `ws-ia`**: That partition has `MaxNodes=1`. You can't sbatch a 2-node job there — must use 2 separate `salloc` sessions.

6. **Image path errors**: `IMAGE_FOLDER` must point to `LLaVA-Pretrain/` (parent of the `00000/` `00001/` ... folders), **not** `LLaVA-Pretrain/images/`. The JSON contains relative paths like `00453/004539375.jpg`.

7. **Tiny model files (~1KB)**: You forgot `git lfs install` before cloning. `cd models/vicuna-7b-v1.5 && git lfs pull` to fix.

---

## Where to look

| What | Where |
|---|---|
| Training entry point | `LLaVA/llava/train/train.py` |
| Custom HF Trainer | `LLaVA/llava/train/llava_trainer.py` |
| Model definition | `LLaVA/llava/model/llava_arch.py` |
| Original LLaVA scripts | `LLaVA/scripts/v1_5/` |
| Our launch scripts | `scripts/` |
| DeepSpeed configs | `configs/`, `LLaVA/scripts/zero{2,3}.json` |
| Logs | `logs/` |
| Checkpoints | `checkpoints/` |
