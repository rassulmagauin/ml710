# Setup & Reproduction Guide

Step-by-step instructions to reproduce the LLaVA Stage 2 baselines on the MBZUAI HPC.

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

We need two models: **vicuna-7b-v1.5** for Stage 1 and **llava-v1.5-7b** as the starting point for Stage 2. The CLIP vision encoder is downloaded automatically by HuggingFace at first run.

```bash
mkdir -p models && cd models

git lfs install

# Vicuna 7B base — used for Stage 1 baseline (~14GB)
git clone https://huggingface.co/lmsys/vicuna-7b-v1.5

# Pre-trained LLaVA — starting point for Stage 2 fine-tuning (~14GB)
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b

cd ..
```

> **Note**: Use `git lfs install` first or you'll get tiny placeholder files. Verify with `du -sh models/vicuna-7b-v1.5` (should be ~13GB).

---

## 5. Download Stage 2 instruction-tuning data

The benchmark target is Stage 2 fine-tuning (trains both the MLP projection and the Vicuna-7B LLM). We use **LLaVA-Instruct-150K** as the annotation source — it's single-source (all images from COCO 2014 train), so you only need one image zip instead of the full 665K mix.

```bash
bash scripts/download_stage2_data.sh
```

This script does three things automatically:

1. Downloads `llava_instruct_150k.json` (~1.1 GB) from HuggingFace
2. Downloads and unzips COCO 2014 train images (~13 GB) into `LLaVA/playground/data/coco/train2014/`
3. Creates a 10K subset `llava_instruct_10k.json` (seed=42) for fast experiments

Expected layout after the script completes:
```
LLaVA/playground/data/
├── llava_instruct_150k.json
├── llava_instruct_10k.json
└── coco/
    └── train2014/
        ├── COCO_train2014_000000000009.jpg
        └── ...
```

> The script is idempotent — re-running it skips already-downloaded files.

---

## 6. Run the Stage 2 single-GPU baseline

This is the **main benchmark baseline** for comparing parallelism strategies. It fine-tunes both the Vicuna-7B LLM and the MLP projection on 10K instruction samples starting from `models/llava-v1.5-7b`.

```bash
mkdir -p logs checkpoints
sbatch scripts/run_stage2_1gpu.sh
```

### Key training configuration

| Arg | Value | Note |
|---|---|---|
| `--model_name_or_path` | `models/llava-v1.5-7b` | Pre-trained LLaVA checkpoint |
| `--version` | `v1` | Conversation template (Vicuna-style) |
| `--image_aspect_ratio` | `pad` | Pad to square, matches v1.5 official setup |
| `--per_device_train_batch_size` | 16 | Per-GPU batch |
| `--learning_rate` | `2e-5` | Stage 2 standard LR |
| `--num_train_epochs` | 1 | |
| `--bf16` | True | Mixed precision |
| `--gradient_checkpointing` | True | Lower memory, slower compute |
| `--model_max_length` | 2048 | |

### Monitor progress

```bash
squeue -u $USER
tail -f logs/stage2_1gpu_<JOBID>.log
```

### Logging wrapper used by the training scripts

The current repo-level training scripts do **not** modify upstream `LLaVA` to collect benchmarking metrics. Instead, they use shared wrappers:

- `scripts/run_logging.sh`
- `scripts/summarize_run.py`

These wrappers are called from inside the repo-level launch scripts (for example `run_stage2_1gpu.sh` and `run_baseline_1gpu.sh`) and automatically record:

- end-to-end runtime
- trainer-reported runtime and throughput
- final and average logged loss
- GPU memory / utilization snapshots
- host RAM snapshots

Each run is saved under a dedicated folder:

```text
logs/runs/<user>/<run_id>/
```

with files such as:

```text
train.log
gpu.csv
ram.log
summary.txt
summary.json
```

and a cumulative per-method CSV:

```text
logs/runs/<user>/<method>_summary.csv
```

If you add a new training script, reuse the same wrapper pattern:

```bash
source "$PROJECT_DIR/scripts/run_logging.sh"
setup_run_logging "<method_name>" "<run_label>" <per_device_batch> <grad_accum> <world_size>
trap cleanup_run_logging EXIT
start_run_logging

set +e
python ... 2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

finish_run_logging "$TRAIN_EXIT"
```

---

## 7. Run the Stage 2 DDP baseline (2 nodes x 1 GPU)

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
# In scripts/run_stage2_ddp.sh, set:
master_node=ws-l4-005   # your terminal A
worker_node=ws-l4-006   # your terminal B
```

### Step 3 — Run the same script on **both** terminals

```bash
eval "$(conda shell.bash hook)" && conda activate llava
bash scripts/run_stage2_ddp.sh
```

> Start both within ~30 seconds of each other or NCCL will time out waiting for the second node.

Uses plain PyTorch DDP via `torchrun` — no DeepSpeed config needed. HF Trainer handles DDP natively when launched with multiple processes.

### How the launch works

```bash
torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \      # 0 on master, 1 on worker
    --master_addr=$master_node \
    --master_port=29500 \
    llava/train/train.py \
    ... (same training args as single-GPU baseline)
```

Each node runs 1 process bound to its single GPU; gradients are all-reduced over the network via NCCL.

### NCCL environment variables (set in script)

```bash
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^lo,docker   # use any interface except loopback/docker
export NCCL_IB_DISABLE=1               # no InfiniBand on ws-ia
export NCCL_P2P_DISABLE=1              # cross-node, no P2P
```

---

## 8. Results so far

| Metric | 1 GPU | 2 GPU DDP | Speedup |
|---|---|---|---|
| Runtime | 46.5 min | 24.6 min | **1.89x** |
| Throughput | 3.585 samples/sec | 6.773 samples/sec | **1.89x** |
| Steps | 625 | 313 | — |
| Effective batch | 16 | 32 | — |
| Avg train loss | 2.7509 | 3.1735 | — |

> These numbers are from Stage 1 runs. Stage 2 results TBD.

---

## Common gotchas

1. **`flash-attn` build hangs / fails**: Skip it. The scripts auto-fall-back to eager attention. Building from source needs `setuptools<70` and a GPU node, takes 1+ hour.

2. **`ImportError: cannot import name 'clear_device_cache'`**: peft/accelerate version mismatch. Run `pip install "peft==0.6.0" "accelerate==0.25.0"`.

3. **`QOSMaxGRESPerUser`** when using `sbatch --gpus=2`: The `gpu` partition limits 1 GPU/user. Use the 2-node `ws-ia` approach instead.

4. **NCCL connection timeout**: Both ranks must call `torchrun` within ~30s of each other. If you saw the error, kill both and re-run.

5. **`Node count specification invalid` on `ws-ia`**: That partition has `MaxNodes=1`. You can't sbatch a 2-node job there — must use 2 separate `salloc` sessions.

6. **Image path errors**: `IMAGE_FOLDER` must point to `coco/train2014/` directly (the folder containing `COCO_train2014_*.jpg` files).

7. **Tiny model files (~1KB)**: You forgot `git lfs install` before cloning. `cd models/llava-v1.5-7b && git lfs pull` to fix.

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

---

---

# Appendix: Stage 1 Baseline (Historical Reference)

Stage 1 trains **only the MLP projection** on image-caption pairs, with the vision encoder and LLM both frozen. It's kept here for reference but is not the primary benchmark — Stage 2 is.

---

## A. Download Stage 1 pre-training data

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

## B. Create a 10K data subset

```bash
python scripts/create_subset.py \
    --input  LLaVA/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --output LLaVA/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_10k.json \
    --n 10000 \
    --seed 42
```

> Use `seed=42` so everyone trains on the **same** subset and results are comparable.

---

## C. Run the Stage 1 single-GPU baseline

```bash
mkdir -p logs checkpoints
sbatch scripts/run_baseline_1gpu.sh
```

Expected: ~46 minutes wall time, 625 training steps.

---

## D. Run the Stage 1 DDP baseline (2 nodes x 1 GPU)

Same two-terminal `salloc` approach as Stage 2 DDP (see step 7 above). Use `scripts/run_baseline_ddp.sh` instead.

```bash
# In scripts/run_baseline_ddp.sh, set:
master_node=ws-l4-005
worker_node=ws-l4-006

# Then on BOTH terminals:
eval "$(conda shell.bash hook)" && conda activate llava
bash scripts/run_baseline_ddp.sh
```

Expected: ~25 minutes wall time, 313 training steps.
