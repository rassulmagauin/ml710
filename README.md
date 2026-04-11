# ML710: Parallelizing LLaVA Training

Course project for ML710 at MBZUAI. We parallelize [LLaVA](https://github.com/haotian-liu/LLaVA) (Large Language and Vision Assistant) — a multimodal model from "Visual Instruction Tuning" (NeurIPS 2023) — using multiple distributed training strategies.

## LLaVA Architecture

LLaVA connects a frozen **CLIP ViT-L/14-336** vision encoder to a **Vicuna-7B** LLM via a 2-layer MLP projection:

```
Image --> [CLIP ViT-L/14-336] --> [MLP 2x GELU Projection] --> [Vicuna 7B LLM] --> Text
              (frozen)               (trainable)                (trainable*)
```

Two-stage training:
1. **Stage 1 (Pre-train)**: Only projection layer trained on 558K image-caption pairs
2. **Stage 2 (Fine-tune)**: Projection + LLM trained on 665K instruction-following data

*In Stage 1, only the projection is trained; LLM is frozen.*

## Repository Structure

```
ml710/
├── LLaVA/                    # Official LLaVA repo (haotian-liu/LLaVA)
│   └── llava/train/          # Training scripts (train.py, llava_trainer.py)
├── scripts/                  # Our experiment launch scripts
│   ├── run_baseline_1gpu.sh  # Single-GPU baseline (sbatch)
│   ├── run_baseline_ddp.sh   # 2-GPU DDP baseline (torchrun, 2-node)
│   └── create_subset.py      # Utility to create data subsets
├── configs/                  # DeepSpeed configurations
│   └── zero0.json            # ZeRO Stage 0 (pure DDP, no sharding)
├── CLAUDE.md                 # Claude Code project context
└── README.md
```

## HPC Environment

- **Cluster**: MBZUAI HPC (`login-student-lab.mbzu.ae`)
- **GPU partition**: 4 nodes, 8x NVIDIA RTX 5000 Ada (32GB) each — limited to 1 GPU/user
- **ws-ia partition**: 116 nodes, 1x RTX 5000 Ada each — used for multi-node DDP
- **Software**: Python 3.10, PyTorch 2.1.2, DeepSpeed 0.12.6, Transformers 4.37.2

Multi-GPU training uses 2 separate `salloc` sessions on `ws-ia` nodes with `torchrun --nnodes=2`, since the HPC QOS limits each job to 1 GPU.

## Baseline Results

All experiments use a **10K sample subset** (from 558K total) of Stage 1 pre-training data, to keep each run under 1 hour.

### Single-GPU vs DDP (2-GPU) Baseline

| Metric | 1 GPU | 2 GPU DDP | Speedup |
|---|---|---|---|
| Runtime | 2789.8s (46.5 min) | 1476.3s (24.6 min) | **1.89x** |
| Throughput | 3.585 samples/sec | 6.773 samples/sec | **1.89x** |
| Training steps | 625 | 313 | — |
| Batch size (per GPU) | 16 | 16 | — |
| Effective batch size | 16 | 32 | — |
| Avg train loss | 2.7509 | 3.1735 | — |
| Final loss | 2.64 | 2.83 | — |

**Key observations:**
- **1.89x speedup** with 2 GPUs — close to ideal 2x linear scaling
- ~11% overhead from cross-node NCCL communication (ws-ia nodes connected over network, not NVLink)
- DDP avg loss is higher because each GPU sees only half the data (fewer gradient updates), but final losses are comparable
- Both runs used eager attention (no flash-attn), bf16, gradient checkpointing

### Setup Details

- **Model**: Vicuna-7B-v1.5 base (not pre-trained LLaVA checkpoint)
- **Training**: Stage 1 only (projection pre-training), 1 epoch, lr=1e-3, cosine schedule
- **Data**: 10K random subset (seed=42) of `blip_laion_cc_sbu_558k.json`
- **DeepSpeed**: ZeRO Stage 0 for DDP (pure data parallelism, no parameter/gradient/optimizer sharding)

## Parallelism Strategies (TODO)

Each team member implements one non-trivial strategy beyond the DDP baseline:

1. **Pipeline Parallelism** — Split model stages (vision encoder / projection / LLM layers) across GPUs
2. **Tensor Parallelism** — Split attention/MLP within transformer layers across GPUs (Megatron-style)
3. **FSDP / ZeRO Stage 2-3** — Shard parameters, gradients, and optimizer states across GPUs

## How to Run

### Prerequisites

```bash
ssh YOUR_USERNAME@login-student-lab.mbzu.ae
eval "$(conda shell.bash hook)" && conda activate llava
```

### Single-GPU Baseline

```bash
sbatch scripts/run_baseline_1gpu.sh
```

### DDP Baseline (2 nodes)

```bash
# Terminal A:
salloc -p ws-ia -N1 -n12 -w ws-l4-XXX --mem=64G
conda activate llava && bash scripts/run_baseline_ddp.sh

# Terminal B:
salloc -p ws-ia -N1 -n12 -w ws-l4-YYY --mem=64G
conda activate llava && bash scripts/run_baseline_ddp.sh
```

Update `master_node` and `worker_node` in `scripts/run_baseline_ddp.sh` with actual allocated hostnames before running.

### Create Data Subset

```bash
python scripts/create_subset.py --input LLaVA/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --output LLaVA/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_10k.json --n 10000
```

## Team

ML710 Spring 2026, MBZUAI

## References

- Liu et al., "Visual Instruction Tuning", NeurIPS 2023
- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
