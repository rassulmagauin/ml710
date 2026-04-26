#!/usr/bin/env python3
"""
train_tp_dist.py — 2-rank cross-node Tensor Parallelism for LLaVA.

Each rank holds the full LLaVA model, but every LlamaDecoderLayer's
attention and MLP linear sublayers are sharded across the two ranks
via torch.distributed.tensor.parallel.parallelize_module:

    self_attn.q_proj:  ColwiseParallel   (output split 4096 → 2048)
    self_attn.k_proj:  ColwiseParallel
    self_attn.v_proj:  ColwiseParallel
    self_attn.o_proj:  RowwiseParallel   (input split, AllReduce after)
    mlp.gate_proj:     ColwiseParallel   (output split 11008 → 5504)
    mlp.up_proj:       ColwiseParallel
    mlp.down_proj:     RowwiseParallel   (input split, AllReduce after)

Per forward pass: ~6 cross-node AllReduces × 32 layers ≈ 192 collectives.
Per step (fwd + bwd): ~384 cross-node collectives.

We train only the MLP projection (Stage 1-style). The full Vicuna body is
frozen but still has activations flowing through its TP-sharded layers,
which is what we're measuring here. LoRA is intentionally disabled — LoRA
adapters interact badly with column/row-sharded base linear layers in
PyTorch 2.1.2's TP API.

Launch (mirrors run_stage2_zero3_lora.sh):
    Node A: MASTER_NODE=<A> WORKER_NODE=<B> bash scripts/run_stage2_tp.sh
    Node B: MASTER_NODE=<A> WORKER_NODE=<B> bash scripts/run_stage2_tp.sh
"""

import argparse
import json
import logging
import math
import os
import pathlib
import sys
import time
from typing import Dict, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
# PyTorch 2.1.2 keeps device_mesh inside torch.distributed._tensor; later
# versions promote it to torch.distributed.device_mesh.
try:
    from torch.distributed.device_mesh import init_device_mesh
except ImportError:
    from torch.distributed._tensor import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.utils.data import DataLoader

_LLAVA_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LLAVA_ROOT))

from llava.model import LlavaLlamaForCausalLM  # noqa: E402
from llava.train.train import (  # noqa: E402
    DataArguments,
    ModelArguments,
    make_supervised_data_module,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Args
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--image_folder", required=True)
    p.add_argument("--vision_tower", default="openai/clip-vit-large-patch14-336")
    p.add_argument("--mm_projector_type", default="mlp2x_gelu")
    p.add_argument("--mm_vision_select_layer", type=int, default=-2)
    p.add_argument("--mm_use_im_start_end", action="store_true")
    p.add_argument("--mm_use_im_patch_token", action="store_true")
    p.add_argument("--image_aspect_ratio", default="pad")
    p.add_argument("--version", default="v1")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--model_max_length", type=int, default=2048)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--lazy_preprocess", action="store_true", default=True)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# TP plan and head-count fix
# ══════════════════════════════════════════════════════════════════════════════

TP_PLAN = {
    "self_attn.q_proj":   ColwiseParallel(),
    "self_attn.k_proj":   ColwiseParallel(),
    "self_attn.v_proj":   ColwiseParallel(),
    "self_attn.o_proj":   RowwiseParallel(),
    "mlp.gate_proj":      ColwiseParallel(),
    "mlp.up_proj":        ColwiseParallel(),
    "mlp.down_proj":      RowwiseParallel(),
}


def apply_tensor_parallel(model: nn.Module, mesh, tp_size: int) -> None:
    """Shard each LlamaDecoderLayer's attention + MLP across the device mesh.

    PyTorch 2.1.2's ColwiseParallel splits along the output dim, which means
    after sharding self_attn.q_proj the output is hidden_size // tp_size on
    each rank. LlamaAttention.forward then reshapes by `num_heads`, so we
    must update `num_heads` and `num_key_value_heads` per layer to match the
    new local shard. Same for hidden_size (used by attention).
    """
    decoder = model.get_model()  # LlamaModel inside LlavaLlamaForCausalLM
    for layer in decoder.layers:
        parallelize_module(layer, mesh, TP_PLAN)
        # Patch attention bookkeeping so reshape ops use the local shard sizes.
        attn = layer.self_attn
        attn.num_heads = attn.num_heads // tp_size
        attn.num_key_value_heads = attn.num_key_value_heads // tp_size
        attn.hidden_size = attn.hidden_size // tp_size


# ══════════════════════════════════════════════════════════════════════════════
# Distributed setup
# ══════════════════════════════════════════════════════════════════════════════

def setup_distributed() -> tuple:
    """Initialize NCCL distributed group. Returns (rank, world_size, local_rank, mesh)."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    mesh = init_device_mesh("cuda", (world_size,))
    return rank, world_size, local_rank, mesh


# ══════════════════════════════════════════════════════════════════════════════
# Model construction (mirrors LLaVA's train.py for the projection-only path)
# ══════════════════════════════════════════════════════════════════════════════

def build_model(args, dtype: torch.dtype) -> tuple:
    """Load LLaVA, freeze the LLM, train only the MLP projection."""
    log.info(f"[rank {dist.get_rank()}] loading model from {args.model_name_or_path}")
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    # Initialize vision modules (projection + vision tower hookup)
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        version=args.version,
        vision_tower=args.vision_tower,
        mm_projector_type=args.mm_projector_type,
        mm_vision_select_layer=args.mm_vision_select_layer,
        mm_use_im_start_end=args.mm_use_im_start_end,
        mm_use_im_patch_token=args.mm_use_im_patch_token,
    )
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=dtype, device="cuda")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # Freeze everything, then unfreeze the MLP projection only.
    model.requires_grad_(False)
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # Bring conversation template in from llava's setup
    from llava import conversation as conversation_lib  # noqa: WPS433
    if args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def trainable_params(model: nn.Module) -> Iterable[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    rank, world_size, local_rank, mesh = setup_distributed()
    is_master = rank == 0
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    if is_master:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- model ---------------------------------------------------------------
    model, tokenizer = build_model(args, dtype)
    model.cuda()

    # --- TP -------------------------------------------------------------------
    log.info(f"[rank {rank}] applying TP across {world_size} ranks")
    apply_tensor_parallel(model, mesh, world_size)
    log.info(f"[rank {rank}] TP applied; trainable params: "
             f"{sum(p.numel() for p in trainable_params(model)):,}")

    # --- data -----------------------------------------------------------------
    data_args = DataArguments(
        data_path=args.data_path,
        lazy_preprocess=args.lazy_preprocess,
        is_multimodal=True,
        image_folder=args.image_folder,
        image_aspect_ratio=args.image_aspect_ratio,
    )
    # `image_processor` lives on the vision tower
    data_args.image_processor = model.get_vision_tower().image_processor
    data_args.mm_use_im_start_end = args.mm_use_im_start_end

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    train_dataset = data_module["train_dataset"]
    collator = data_module["data_collator"]

    sampler = torch.utils.data.SequentialSampler(train_dataset)  # deterministic order
    loader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.per_device_train_batch_size,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    # --- optimizer ------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        trainable_params(model),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # --- training loop --------------------------------------------------------
    n_samples = len(train_dataset)
    steps_per_epoch = math.ceil(n_samples / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
    total_steps = int(steps_per_epoch * args.num_train_epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    if is_master:
        log.info(f"n_samples={n_samples}  per_device_batch={args.per_device_train_batch_size}  "
                 f"accum={args.gradient_accumulation_steps}  total_steps={total_steps}  "
                 f"warmup={warmup_steps}")

    model.train()
    global_step = 0
    epoch = 0.0
    accum_loss = 0.0
    accum_count = 0
    t_start = time.time()

    optimizer.zero_grad(set_to_none=True)
    for ep in range(int(math.ceil(args.num_train_epochs))):
        for batch_idx, batch in enumerate(loader):
            # Move batch to GPU
            batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            # Cast image tensor to bf16 to match model
            if "images" in batch and torch.is_tensor(batch["images"]):
                batch["images"] = batch["images"].to(dtype=dtype)

            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            accum_loss += loss.item() * args.gradient_accumulation_steps
            accum_count += 1

            if accum_count >= args.gradient_accumulation_steps:
                lr = cosine_lr(global_step, total_steps, warmup_steps, args.learning_rate)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                epoch = global_step / steps_per_epoch
                if is_master and global_step % args.logging_steps == 0:
                    avg = accum_loss / accum_count
                    elapsed = time.time() - t_start
                    rate = global_step / max(1e-6, elapsed)
                    eta = (total_steps - global_step) / max(1e-6, rate)
                    # HF Trainer-style log line — picked up by summarize_run.py
                    print(json.dumps(
                        {"loss": round(avg, 4),
                         "learning_rate": lr,
                         "epoch": round(epoch, 4)},
                    ).replace('"', "'"), flush=True)
                    # tqdm-style progress so summarize_run.py can extract step counts
                    print(f"  {int(100*global_step/total_steps):3d}%|"
                          f"{'#'*int(20*global_step/total_steps):<20}| "
                          f"{global_step}/{total_steps} "
                          f"[{int(elapsed//60):02d}:{int(elapsed%60):02d}<"
                          f"{int(eta//60):02d}:{int(eta%60):02d}, "
                          f"{rate:.2f}it/s]",
                          flush=True)
                accum_loss = 0.0
                accum_count = 0

                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break

    # --- save the projection (only trainable part) ---------------------------
    dist.barrier()
    if is_master:
        save_path = pathlib.Path(args.output_dir) / "mm_projector.bin"
        proj_state = {k: v.detach().cpu()
                      for k, v in model.get_model().mm_projector.state_dict().items()}
        torch.save(proj_state, save_path)
        log.info(f"saved projection to {save_path}")
        log.info(f"total training time: {(time.time() - t_start)/60:.1f} min")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
