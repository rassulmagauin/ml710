# ZeRO-3 Stage 2 Launcher

This repo now includes a dedicated Stage 2 ZeRO-3 LoRA launcher:

- [scripts/run_stage2_zero3_lora.sh](/l/users/youssef.ghallab/ml710/scripts/run_stage2_zero3_lora.sh)
- [configs/zero3.json](/l/users/youssef.ghallab/ml710/configs/zero3.json)

Defaults were chosen to be conservative for `2 x RTX 5000 Ada 32 GB`:

- `PER_DEVICE_BATCH=1`
- `GRAD_ACCUM=4`
- effective batch `8`
- `LEARNING_RATE=5e-5`
- `WARMUP_RATIO=0.1`
- `OPTIM=adamw_torch`
- `LR_SCHEDULER_TYPE=cosine`
- `PER_DEVICE_EVAL_BATCH=1`
- `DATALOADER_WORKERS=0`

Usage on two allocated nodes:

```bash
MASTER_NODE=ws-l4-005 WORKER_NODE=ws-l4-006 bash scripts/run_stage2_zero3_lora.sh
```

Run the same command on both nodes after activating the `llava` environment. The launcher reuses the shared run logging, so summaries, throughput, goodput, statistical efficiency, and plots are written under `logs/runs/...` like the existing DDP and ZeRO-2 scripts.

Assumptions and risks:

- The LLaVA training code already contains ZeRO-3-aware save helpers, so this launcher relies on that existing behavior.
- The config keeps the vision tower frozen indirectly by matching the current Stage 2 LoRA training path rather than introducing new unfreeze flags.
- ZeRO-3 compatibility can still depend on the installed DeepSpeed version. If checkpoint saving fails, the most likely cause is a version mismatch around `stage3_gather_16bit_weights_on_model_save`.
