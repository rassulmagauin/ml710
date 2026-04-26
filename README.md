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
├── LLaVA/                            # Official LLaVA source, vendored (not a submodule)
│   └── llava/train/
│       ├── train.py                  # Stock LLaVA Stage 1/2 training entry point
│       ├── train_mem.py              # Same, with flash_attention_2
│       ├── train_pipe.py             # Manual pipeline-parallel trainer (single node)
│       ├── train_pipe_dist.py        # 2-node distributed pipeline (Gloo)
│       └── train_pipe_8gpu.py        # 8-GPU pipeline variant
├── scripts/                          # Our experiment launch scripts
│   ├── run_baseline_1gpu.sh          # Stage 1 single-GPU (historical reference)
│   ├── run_baseline_ddp.sh           # Stage 1 DDP, 2 nodes x 1 GPU
│   ├── run_stage2_1gpu.sh            # Stage 2 LoRA, single-GPU baseline
│   ├── run_stage2_ddp.sh             # Stage 2 LoRA, DDP, 2 nodes x 1 GPU
│   ├── run_stage2_zero2_lora.sh      # Stage 2 LoRA, DeepSpeed ZeRO-2
│   ├── run_stage2_zero2_lora_util.sh # ZeRO-2 with utilization-tuned defaults
│   ├── run_stage2_zero3_lora.sh      # Stage 2 LoRA, DeepSpeed ZeRO-3, 2 nodes
│   ├── run_stage2_zero3_lora_util.sh # ZeRO-3 with bounded async comm
│   ├── run_stage2_pipe_lora.sh       # Stage 2 LoRA, manual pipeline parallel
│   ├── run_stage2_pipe_lora_safe.sh  # Pipeline launcher with conservative defaults
│   ├── download_stage2_data.sh       # Downloads Stage 2 data + creates 10K subset
│   ├── create_subset.py              # Random subset utility
│   ├── run_logging.sh                # Shared run-logging wrapper
│   ├── summarize_run.py              # Per-run + cumulative CSV/plot summarizer
│   └── plot_statistical_efficiency.py
├── configs/                          # DeepSpeed configurations
│   ├── zero0.json                    # ZeRO Stage 0 (pure DDP, no sharding)
│   ├── zero2.json                    # ZeRO Stage 2 (optimizer + grad sharding)
│   ├── zero3.json                    # ZeRO Stage 3 (full sharding)
│   └── zero3_bounded_async.json      # ZeRO-3 with capped allgather/reduce buckets
├── logs/runs/<user>/<run_id>/        # Per-run artifacts (train.log, gpu.csv, summary.*)
├── checkpoints/                      # Training output
├── SETUP.md                          # Reproduction instructions
├── SYSTEM_REQUIREMENTS.md            # Per-strategy hardware/software requirements
├── ZERO2_RESULTS.md                  # ZeRO-2 / ZeRO-3 / Pipeline result tables
├── ZERO3_NOTES.md                    # ZeRO-3 launcher notes
├── results_explanation.md            # Narrative explanation of why each result happened
├── CLAUDE.md                         # Project context for Claude Code
└── README.md
```

## HPC Environment

- **Cluster**: MBZUAI HPC (`login-student-lab.mbzu.ae`)
- **GPU partition**: 4 nodes, 8x NVIDIA RTX 5000 Ada (32GB) each — limited to 1 GPU/user via `gpu-1` QOS
- **ws-ia partition**: 116 nodes, 1x RTX 5000 Ada each — used for multi-node distributed training (2 separate `salloc` sessions)
- **Software**: Python 3.10, PyTorch 2.1.2, DeepSpeed 0.12.6, Transformers 4.37.2, peft 0.6.0, accelerate 0.25.0

Multi-GPU runs use 2 separate `salloc` sessions on `ws-ia` nodes connected over ~1 Gbps Ethernet (no NVLink, no InfiniBand). This network-bound interconnect strongly shapes the results below.

## Stage 2 Results (LoRA fine-tune from `llava-v1.5-7b`, 10K instruction subset)

All Stage 2 runs use the same recipe: Vicuna-7B + CLIP ViT-L/14-336, LoRA (r=128, α=256), bf16, gradient checkpointing, cosine schedule, 1 epoch. Each strategy was launched on the `ws-ia` partition with 2 nodes × 1 GPU (or 1 GPU for the single-GPU baselines).

| Strategy | Effective batch | Wall time | Throughput | Peak GPU mem | GPU util | Notes |
|---|---:|---:|---:|---:|---:|---|
| Single-GPU LoRA | 1 | 96 min | 1.74 samples/s | 26.3 GiB | 90.9% | Baseline (no parallelism) |
| **DDP** (2×1 GPU) | 32 | 4491 s (74.9 min) | 2.22 samples/s | 31.4 GiB | 70.7% | Naïve baseline |
| **ZeRO-2** (best) | 32 | 3503 s (58.4 min) | 3.01 samples/s | 28.3 GiB | 88.5% | **1.28× over DDP** |
| **ZeRO-3** | 8 | timed out at 292/1250 steps | ~0.17 samples/s extrapolated | 30.3 GiB | 99.3% | Projected ~16.5 h, ~13× slower than DDP |
| **FSDP** (PyTorch native) | 8 | stopped at 28/1250 steps | 0.095 samples/s | 32.3 GiB | 97.5% | Projected ~27.9 h, ~22× slower than DDP; ~1.7× slower than ZeRO-3 |
| **Pipeline** (split=12) | 16 | 4325 s (72.1 min) | 0.85 samples/s | 32.1 GiB | 33.9% / 56.1% | Completed; not faster than DDP |

See **`ZERO2_RESULTS.md`** for the full per-run table and **`results_explanation.md`** for the analysis. Highlights:

- **ZeRO-2 only modestly beats DDP** because LoRA already shrinks optimizer/gradient state — there's almost nothing to shard.
- **Pipeline only completed with split=12 + microbatch=1**; uneven splits OOM'd the heavier rank.
- **ZeRO-3 is ~13× slower than DDP on this setup** because per-layer AllGather/ReduceScatter of full bf16 weights bottlenecks on cross-node 1 Gbps Ethernet.
- **PyTorch FSDP is even slower than DeepSpeed ZeRO-3** (~1.7×) on the same hardware — same algorithm class, but ZeRO-3's bucketed async communication overlaps gather with compute more aggressively than FSDP's defaults. Implementation matters when interconnect is the bottleneck.

## Parallelism Strategies (Implemented)

The course project asks each team member to implement at least one non-trivial parallelism strategy (DDP doesn't count). The repo currently has:

1. **Pipeline Parallelism** — manual 2-stage split of the Vicuna decoder across 2 nodes. Vision tower + projection + decoder layers `0..k` on rank 0, layers `k..31` + LM head on rank 1. Implemented in `LLaVA/llava/train/train_pipe_dist.py` (Gloo backend over Ethernet). Best balanced split is `k=12`.
2. **DeepSpeed ZeRO-2** — sharded optimizer states + gradients across 2 ranks. Pure data parallelism with memory-optimized state.
3. **DeepSpeed ZeRO-3** — sharded optimizer states + gradients + parameters. Same data-parallel semantics as ZeRO-2 but with per-layer AllGather of weights at compute time.

Whether ZeRO-2 and ZeRO-3 satisfy the "3 distinct strategies" rule (vs. counting as two configurations of the same algorithm) is being clarified with the course staff — both are formally classified as "ZeRO-DP" in the DeepSpeed paper.

## How to Run

> **For full reproduction instructions (env setup, model/data download, common gotchas), see [SETUP.md](SETUP.md).**
>
> **For per-strategy hardware/software requirements, see [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md).**

### Prerequisites

```bash
ssh YOUR_USERNAME@login-student-lab.mbzu.ae
eval "$(conda shell.bash hook)" && conda activate llava
```

### Stage 2 single-GPU baseline

```bash
bash scripts/download_stage2_data.sh   # one-time, ~14 GB
sbatch scripts/run_stage2_1gpu.sh
```

### Stage 2 multi-node (2× ws-ia, 1 GPU each)

All multi-node launchers follow the same pattern: allocate two nodes in two terminals, then run the same script on both.

```bash
# Terminal A:
salloc -p ws-ia -N1 -n12 -w ws-l4-XXX --mem=64G
conda activate llava && bash scripts/<launcher>.sh

# Terminal B:
salloc -p ws-ia -N1 -n12 -w ws-l4-YYY --mem=64G
conda activate llava && bash scripts/<launcher>.sh
```

Where `<launcher>` is one of:

| Strategy | Launcher |
|---|---|
| DDP (Stage 1) | `run_baseline_ddp.sh` |
| DDP (Stage 2 LoRA) | `run_stage2_ddp.sh` |
| ZeRO-2 | `run_stage2_zero2_lora.sh` |
| ZeRO-2 (utilization-tuned) | `run_stage2_zero2_lora_util.sh` |
| ZeRO-3 | `run_stage2_zero3_lora.sh` |
| ZeRO-3 (bounded async comm) | `run_stage2_zero3_lora_util.sh` |
| Pipeline parallel | `run_stage2_pipe_lora.sh` |

Each launcher requires editing `master_node` and `worker_node` (or `MASTER_NODE` / `WORKER_NODE` env vars) to match the hostnames returned by `salloc`.

### Logging and run artifacts

All training launchers source `scripts/run_logging.sh`, which records:

- end-to-end runtime
- HF Trainer-reported runtime, throughput, samples/sec
- per-step logged loss
- GPU memory and utilization snapshots (`nvidia-smi`, every 5 s)
- host RAM snapshots (`free -h`, every 5 s)

Run artifacts are stored at:

```
logs/runs/<user>/<run_id>/
    train.log
    gpu.csv
    ram.log
    summary.txt
    summary.json
    plots/
```

Plus a per-method cumulative CSV at `logs/runs/<user>/<method>_summary.csv`.

To add the same logging to a new training script:

```bash
source "$PROJECT_DIR/scripts/run_logging.sh"
setup_run_logging "<method_name>" "<run_label>" <per_device_batch> <grad_accum> <world_size>
trap cleanup_run_logging EXIT
start_run_logging
...
finish_run_logging "$TRAIN_EXIT"
```

## Stage 1 Baseline (Historical Reference)

The original baseline trained only the MLP projection (Stage 1 pretrain) on a 10K subset of `blip_laion_cc_sbu_558k`. Kept here as a controlled DDP scaling reference, not the primary benchmark.

| Metric | 1 GPU | 2-GPU DDP | Speedup |
|---|---:|---:|---:|
| Runtime | 2789.8 s | 1476.3 s | **1.89×** |
| Throughput | 3.585 samples/s | 6.773 samples/s | **1.89×** |
| Effective batch | 16 | 32 | — |
| Avg train loss | 2.7509 | 3.1735 | — |

The Stage 2 numbers are smaller in nominal speedup because the MLP-only Stage 1 has almost no gradient communication, while Stage 2 LoRA AllReduces adapter gradients across nodes every step.

## Team

ML710 Spring 2026, MBZUAI. Deadline: 2026-04-29. Deliverable: ≤ 30 slides.

## References

- Liu et al., "Visual Instruction Tuning", NeurIPS 2023
- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", SC 2020
