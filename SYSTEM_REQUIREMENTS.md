# System Requirements for LLaVA Parallelization

This document consolidates the project requirements from `README.md` and `SETUP.md`, then turns them into concrete system requirements for each training strategy.

## Techniques We Need to Run

The project requires:

1. Single-GPU baseline (reference only)
2. Naive DDP baseline (required comparison, but does not count as a non-trivial strategy)
3. Pipeline parallelism
4. Tensor parallelism
5. FSDP or DeepSpeed ZeRO sharding

The non-trivial strategies explicitly called out by the project docs are pipeline parallelism, tensor parallelism, and FSDP/ZeRO.

## Shared Requirements Across All Runs

### Hardware

- Login node is not sufficient for training. GPU jobs must run under `salloc` or `sbatch`.
- Working GPU class from the docs: NVIDIA RTX 5000 Ada, 32 GB VRAM per GPU.
- Minimum host RAM already used successfully by the baseline scripts: 64 GB.
- Recommended CPU allocation already used by the baseline scripts: 16 CPU cores per GPU.
- Disk requirements:
  - Vicuna-7B v1.5 base model: about 14 GB
  - Optional LLaVA checkpoint: about 14 GB
  - Stage 1 pretrain data after extraction: about 24 GB
  - Extra space required for checkpoints and logs

### Software

- OS: Linux on MBZUAI HPC
- Python: 3.10
- Conda env: `llava`
- PyTorch: `2.1.2+cu121`
- DeepSpeed: `0.12.6`
- Transformers: `4.37.2`
- LLaVA installed editable from `LLaVA/`
- `flash-attn` is optional and currently not required

### Data and Models

- Base LLM: `models/vicuna-7b-v1.5`
- Vision tower: `openai/clip-vit-large-patch14-336`
- Stage 1 data root: `LLaVA/playground/data/LLaVA-Pretrain/`
- Shared 10K subset for comparable experiments:
  `LLaVA/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_10k.json`

### Runtime Constraints

- Each experiment should stay under 1 hour.
- All comparisons should report throughput, loss/statistical efficiency, goodput, GPU utilization, and memory usage.
- Stage 1 is the safer target for all strategy bring-up because only the projector is trainable.

## Cluster Constraints That Affect Design

- Verified behavior on this account:
  - `sbatch` with `--gpus-per-node=2` on the `gpu` partition is rejected by QoS policy
  - a 1-GPU `sbatch` on the same partition is accepted
  - the current baseline path is therefore the 2-node `ws-ia` setup with 1 GPU per node
- Effective policy for this project:
  - single-node multi-GPU on `gpu` should be treated as unavailable unless the account QoS changes
  - multi-GPU experiments must currently assume 2 or more nodes, each contributing 1 GPU

Given that constraint, the practical requirement for each strategy is:

- DDP baseline: feasible today with 2 separate `ws-ia` nodes
- ZeRO/FSDP: feasible today across 2 or more nodes
- Pipeline parallelism: technically possible, but much less attractive on the current 1-GPU-per-node layout
- Tensor parallelism: technically possible, but strongly constrained by cross-node communication overhead

## Per-Strategy System Requirements

### 1. Single-GPU Baseline

Purpose: performance and convergence reference.

- Minimum GPUs: 1
- Topology: single node
- GPU memory target: 32 GB VRAM is sufficient for the current Stage 1 baseline
- Host RAM: 64 GB
- CPU: 16 cores
- Launcher: `sbatch scripts/run_baseline_1gpu.sh`
- Existing repo support:
  - `scripts/run_baseline_1gpu.sh`
  - `LLaVA/llava/train/train.py`
- Notes:
  - Uses eager attention by default
  - Uses bf16, tf32, and gradient checkpointing

### 2. Naive DDP Baseline

Purpose: required baseline for all non-trivial comparisons.

- Minimum GPUs: 2
- Topology: 2 nodes x 1 GPU on `ws-ia`
- Host RAM: 64 GB per node
- CPU: 12-16 CPU cores per node
- Launcher: `bash scripts/run_baseline_ddp.sh` on both allocated nodes
- Existing repo support:
  - `scripts/run_baseline_ddp.sh`
  - `configs/zero0.json`
- Required runtime configuration:
  - `torchrun --nnodes=2 --nproc_per_node=1`
  - consistent `master_addr` and `master_port`
  - NCCL socket configuration for cross-node communication
- Risks:
  - cross-node communication overhead
  - synchronized startup requirement
  - one-GPU-per-node layout is the confirmed cluster constraint for multi-GPU work on this account
  - this layout is a poor fit for pipeline or tensor parallel extensions

### 3. Pipeline Parallelism

Purpose: split model stages across GPUs, likely:
vision encoder -> projector -> early LLM layers -> late LLM layers.

- Minimum GPUs: 2
- Recommended GPUs: 2-4
- Topology requirement:
  - current cluster reality is cross-node, 1 GPU per node
  - single-node multi-GPU would be better, but is not currently available under the verified QoS limits
  - cross-node pipeline adds latency and pipeline-bubble risk
- GPU memory goal:
  - enough to place separate stages on separate ranks
  - 32 GB VRAM per GPU is likely sufficient for Stage 1 and useful for Stage 2 if the split is balanced
- Host RAM: 64-128 GB
- CPU: 16 cores total minimum; 8+ cores per active rank is safer
- Software requirements:
  - PyTorch distributed
  - either native `torch.distributed.pipeline.sync.Pipe` style support or a custom stage-partitioned training loop
  - changes in `LLaVA/llava/model/` and training entry points to isolate stage boundaries
- Repo status:
  - no existing pipeline implementation found
  - must add new launch script(s) and model partition code
  - should be treated as higher-risk under the current 2-node setup
- Implementation requirements:
  - define stage boundaries explicitly
  - handle micro-batching to avoid pipeline bubbles
  - keep activation transfers on fast interconnects when possible
- Measurement requirements:
  - bubble overhead
  - stage imbalance
  - per-stage memory and utilization

### 4. Tensor Parallelism

Purpose: split attention and MLP computations inside transformer layers across GPUs.

- Minimum GPUs: 2
- Recommended GPUs: 2-4
- Topology requirement:
  - current cluster reality is cross-node, 1 GPU per node
  - tensor parallelism is much less attractive in this environment because it needs frequent fine-grained collectives
  - high-bandwidth GPU-to-GPU communication matters much more here than for DDP
- GPU memory goal:
  - 32 GB VRAM per GPU is enough for experimentation, especially for Stage 1
- Host RAM: 64-128 GB
- CPU: 16 cores total minimum
- Software requirements:
  - distributed collectives for all-reduce/all-gather in every transformer block
  - likely Megatron-style layer rewrites or integration with an existing tensor-parallel library
  - deterministic rank-local partitioning of linear layers and attention heads
- Repo status:
  - no existing tensor parallel code found
  - must add new model wrappers, distributed launch scripts, and validation logic
  - should be treated as the highest-risk strategy on the current cluster policy
- Implementation requirements:
  - shard QKV and MLP projections consistently
  - synchronize outputs across tensor-parallel ranks
  - ensure checkpoint save/load semantics across sharded weights
- Measurement requirements:
  - communication overhead per layer
  - memory savings versus DDP
  - throughput sensitivity to interconnect quality

### 5. FSDP / DeepSpeed ZeRO

Purpose: shard parameters, gradients, and optimizer states instead of fully replicating them.

- Minimum GPUs: 2
- Recommended GPUs: 2-4
- Topology:
  - can run on multi-node setups
  - benefits from good network performance, but is less topology-sensitive than tensor parallelism
- GPU memory goal:
  - 32 GB VRAM per GPU is sufficient for Stage 1 and likely enough to study Stage 2 scaling benefits
- Host RAM: 64 GB minimum, 128 GB safer for larger checkpoints and optimizer partitioning
- CPU: 16 cores total minimum
- Software requirements:
  - for ZeRO: DeepSpeed 0.12.6 with `zero2.json` and `zero3.json`
  - for FSDP: PyTorch FSDP integration plus new wrapping policy code
- Existing repo support:
  - `configs/zero0.json`
  - `LLaVA/scripts/zero2.json`
  - `LLaVA/scripts/zero3.json`
  - existing training code already imports DeepSpeed hooks
- Repo status:
  - ZeRO configs exist and are the lowest-effort non-trivial strategy to implement next
  - FSDP-specific code does not exist yet
  - this is the best-fit non-trivial parallelization family for the confirmed 2-node setup
- Implementation requirements:
  - decide whether the project deliverable uses DeepSpeed ZeRO or native FSDP
  - create dedicated launch scripts for ZeRO-2 and/or ZeRO-3
  - validate checkpoint save/load and training stability
- Measurement requirements:
  - peak GPU memory
  - step time increase from sharding communication
  - maximum feasible batch size and sequence length

## Recommended Execution Order

Given the repo state today, the practical order is:

1. Keep the existing 1-GPU baseline as the reference
2. Keep the current 2-GPU DDP baseline as the comparison floor
3. Implement ZeRO-2 first, then ZeRO-3 if needed
4. Implement pipeline parallelism only if the team decides to accept the extra cross-node complexity
5. Implement tensor parallelism last, because it is both the deepest model surgery and the worst fit for the current cluster policy

## Immediate Gaps

- There is no project-local implementation yet for pipeline parallelism.
- There is no project-local implementation yet for tensor parallelism.
- There is no project-local launch script yet for ZeRO-2 or ZeRO-3 experiments.
- The account is currently constrained to a 2-node, 1-GPU-per-node setup for multi-GPU experiments.

That cluster policy is the main infrastructure blocker for efficient pipeline and tensor parallel work.
