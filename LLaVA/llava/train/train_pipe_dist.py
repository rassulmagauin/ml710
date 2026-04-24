#!/usr/bin/env python3
"""
train_pipe_dist.py — 2-stage cross-node pipeline parallelism for LLaVA.

One process per node, each with 1 GPU:
  Rank 0 (node A, cuda:0): vision tower + mm_projector + embed_tokens + layers[0:split_layer]
  Rank 1 (node B, cuda:0): layers[split_layer:32] + RMSNorm + lm_head + CE loss

Schedule: GPipe (all forward, then all backward).
  - Forward : rank 0 sends hidden states → rank 1 (one chunk at a time)
  - Backward: rank 1 sends ∂loss/∂h    → rank 0 (one chunk at a time)

Communication per micro-batch (seq_len=2048, hidden=4096, bf16):
  hidden states : ~16 MB  (forward + backward = ~32 MB per chunk)
  attn_mask 2D  : ~4 KB   (recompute 4D on rank 1 — avoids sending ~8 MB)
  labels / pos  : negligible

Launch (same pattern as run_stage2_zero2_lora.sh):
  Node A:  MASTER_NODE=<A> WORKER_NODE=<B> bash scripts/run_stage2_pipe_lora.sh
  Node B:  MASTER_NODE=<A> WORKER_NODE=<B> bash scripts/run_stage2_pipe_lora.sh
"""

import argparse
import logging
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, SequentialSampler
from transformers.models.llama.modeling_llama import _prepare_4d_causal_attention_mask

_LLAVA_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LLAVA_ROOT))

from llava.model import LlavaLlamaForCausalLM          # noqa: E402
from llava.train.train import (                        # noqa: E402
    DataArguments, ModelArguments, make_supervised_data_module,
)
from llava import conversation as conversation_lib     # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

IGNORE_INDEX = -100
RANK0_DST = 0
RANK1_DST = 1


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Stages
# ══════════════════════════════════════════════════════════════════════════════

class PipeStage0(nn.Module):
    """Rank 0 — vision encoding, multimodal fusion, first split_layer decoder layers."""

    def __init__(self, llava_model: LlavaLlamaForCausalLM, split_layer: int):
        super().__init__()
        inner = llava_model.get_model()
        self.vision_tower  = inner.get_vision_tower()
        self.mm_projector  = inner.mm_projector
        self.embed_tokens  = inner.embed_tokens
        self.layers        = nn.ModuleList(inner.layers[:split_layer])
        self._llava_causal = llava_model
        self._llama_model  = inner

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        images: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = next(self.embed_tokens.parameters()).device

        (_, position_ids, attention_mask, _, inputs_embeds, labels) = (
            self._llava_causal.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, labels, images, None
            )
        )
        bsz, seq_len, _ = inputs_embeds.shape

        # Build 4-D causal mask locally (never sent across the wire)
        attn_4d = _prepare_4d_causal_attention_mask(
            attention_mask, (bsz, seq_len), inputs_embeds, 0
        )
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=dev).unsqueeze(0).expand(bsz, -1)

        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, attention_mask=attn_4d, position_ids=position_ids)[0]

        # Return: hidden states (grad needed), 2D mask (cheap to send), pos_ids, labels
        return h, attention_mask, position_ids, labels


class PipeStage1(nn.Module):
    """Rank 1 — remaining decoder layers, RMSNorm, lm_head, CE loss."""

    def __init__(self, llava_model: LlavaLlamaForCausalLM, split_layer: int):
        super().__init__()
        inner = llava_model.get_model()
        self.layers    = nn.ModuleList(inner.layers[split_layer:])
        self.norm      = inner.norm
        self.lm_head   = llava_model.lm_head
        self._llama_model = inner

    def forward(
        self,
        hidden_states: torch.Tensor,    # requires_grad=True (leaf on rank 1)
        attention_mask_2d: torch.Tensor,
        position_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        dev = hidden_states.device
        bsz, seq_len, _ = hidden_states.shape

        # Reconstruct 4-D causal mask on this side — avoids sending ~8 MB over the wire
        attn_4d = _prepare_4d_causal_attention_mask(
            attention_mask_2d, (bsz, seq_len), hidden_states, 0
        )

        h = hidden_states
        for layer in self.layers:
            h = layer(h, attention_mask=attn_4d, position_ids=position_ids)[0]

        h = self.norm(h)
        logits = self.lm_head(h)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(dev)
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )


# ══════════════════════════════════════════════════════════════════════════════
# P2P communication helpers
# ══════════════════════════════════════════════════════════════════════════════

def _send_tensor(t: torch.Tensor, dst: int) -> None:
    """Send ndim → shape → data (blocking). Each send is a fixed-size tensor."""
    wire = t.detach().contiguous().cpu()
    ndim = torch.tensor([t.ndim], dtype=torch.long)
    dist.send(ndim, dst=dst)
    shape = torch.tensor(list(t.shape), dtype=torch.long)   # exactly t.ndim elements
    dist.send(shape, dst=dst)
    dist.send(wire, dst=dst)


def _recv_tensor(src: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Receive ndim → shape → data (blocking)."""
    ndim_buf = torch.zeros(1, dtype=torch.long)
    dist.recv(ndim_buf, src=src)
    shape_buf = torch.zeros(int(ndim_buf.item()), dtype=torch.long)
    dist.recv(shape_buf, src=src)
    buf = torch.empty(tuple(shape_buf.tolist()), dtype=dtype)
    dist.recv(buf, src=src)
    return buf.to(device, non_blocking=True)


def send_activations(h: torch.Tensor, attn_2d: torch.Tensor,
                     pos_ids: torch.Tensor, labels: torch.Tensor) -> None:
    """Rank 0 → Rank 1: forward activations."""
    _send_tensor(h.detach(),  dst=RANK1_DST)
    _send_tensor(attn_2d,     dst=RANK1_DST)
    _send_tensor(pos_ids,     dst=RANK1_DST)
    _send_tensor(labels,      dst=RANK1_DST)


def recv_activations(dtype: torch.dtype, device: torch.device) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank 1 receives forward activations from rank 0."""
    h       = _recv_tensor(RANK0_DST, dtype,       device)
    attn_2d = _recv_tensor(RANK0_DST, torch.bool,  device)
    pos_ids = _recv_tensor(RANK0_DST, torch.long,  device)
    labels  = _recv_tensor(RANK0_DST, torch.long,  device)
    return h, attn_2d, pos_ids, labels


def send_grad(grad: torch.Tensor) -> None:
    dist.send(grad.detach().contiguous().cpu(), dst=RANK0_DST)


def recv_grad(h: torch.Tensor) -> torch.Tensor:
    buf = torch.empty(h.shape, dtype=h.dtype)
    dist.recv(buf, src=RANK1_DST)
    return buf.to(h.device, non_blocking=True)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline step (one gradient-accumulation step)
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_step_rank0(stage: PipeStage0, chunks: List[Dict],
                        grad_scale: float, dtype: torch.dtype) -> None:
    """GPipe forward-all then backward-all on rank 0.

    grad_scale (= 1/grad_accum) is applied by rank 1 before sending gradients,
    so we must NOT multiply again here — just pass grad_h straight to backward().
    """
    saved = []

    # Broadcast actual chunk count so rank 1 knows how many recvs to do
    n = torch.tensor([len(chunks)], dtype=torch.long)
    dist.broadcast(n, src=0)

    for chunk in chunks:
        h, attn_2d, pos_ids, labels = stage(
            chunk["input_ids"], chunk["attention_mask"],
            chunk["labels"],    chunk["images"],
        )
        send_activations(h, attn_2d, pos_ids, labels)
        saved.append(h)

    for h in saved:
        grad_h = recv_grad(h)
        h.backward(grad_h)   # grad_scale already embedded by rank 1


def pipeline_step_rank1(stage: PipeStage1, grad_scale: float,
                        dtype: torch.dtype, device: torch.device) -> float:
    """GPipe forward-all then backward-all on rank 1. Returns mean loss."""
    # Receive actual chunk count from rank 0
    n_t = torch.zeros(1, dtype=torch.long)
    dist.broadcast(n_t, src=0)
    n_chunks = int(n_t.item())

    chunk_inputs = []
    chunk_losses = []

    for _ in range(n_chunks):
        h_data, attn_2d, pos_ids, labels = recv_activations(dtype, device)
        h    = h_data.requires_grad_(True)
        loss = stage(h, attn_2d, pos_ids, labels)
        chunk_inputs.append(h)
        chunk_losses.append(loss)

    mean_loss = torch.stack(chunk_losses).mean()
    # Apply grad_scale here once; rank 0 will use the result directly
    (mean_loss * grad_scale).backward()

    for h in chunk_inputs:
        send_grad(h.grad)

    return mean_loss.item()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def split_batch(batch: Dict, num_chunks: int) -> List[Dict]:
    bsz  = batch["input_ids"].size(0)
    sz   = max(1, (bsz + num_chunks - 1) // num_chunks)
    imgs = batch.get("images")
    return [
        {
            "input_ids":      batch["input_ids"][s : s + sz],
            "attention_mask": batch["attention_mask"][s : s + sz],
            "labels":         batch["labels"][s : s + sz],
            "images":         imgs[s : s + sz] if imgs is not None else None,
        }
        for s in range(0, bsz, sz)
    ]


def build_optimizer(stage, lr, mm_projector_lr, weight_decay):
    has_proj = hasattr(stage, "mm_projector")
    proj_ids = {id(p) for p in stage.mm_projector.parameters()} if has_proj else set()
    groups = []
    if has_proj:
        groups.append({
            "params": [p for p in stage.mm_projector.parameters() if p.requires_grad],
            "lr": mm_projector_lr,
        })
    groups.append({
        "params": [p for p in stage.parameters()
                   if id(p) not in proj_ids and p.requires_grad],
        "lr": lr,
    })
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    rank   = dist.get_rank()
    device = torch.device("cuda:0")
    dtype  = torch.bfloat16 if args.bf16 else torch.float32

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info(f"[rank {rank}] Loading LLaVA model …")
    base_model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    base_model.config.use_cache = False

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        version="v1",
        vision_tower="openai/clip-vit-large-patch14-336",
        mm_projector_type="mlp2x_gelu",
        mm_vision_select_layer=-2,
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
    )
    base_model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)

    vision_tower = base_model.get_model().get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(dtype=dtype)
    for p in vision_tower.parameters():
        p.requires_grad_(False)

    # ── LoRA ──────────────────────────────────────────────────────────────────
    log.info(f"[rank {rank}] Applying LoRA …")
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
    )
    peft_model  = get_peft_model(base_model, lora_cfg)
    llava_model = peft_model.base_model.model

    if rank == 0:
        peft_model.print_trainable_parameters()

    # ── Build stage for this rank ──────────────────────────────────────────────
    if rank == 0:
        stage = PipeStage0(llava_model, args.split_layer).to(dtype).to(device)
        log.info(f"[rank 0] Stage 0: decoder layers 0–{args.split_layer - 1}")
    else:
        stage = PipeStage1(llava_model, args.split_layer).to(dtype).to(device)
        log.info(f"[rank 1] Stage 1: decoder layers {args.split_layer}–31")

    # Discard unused modules to free memory
    if rank == 0:
        del llava_model.model.layers[args.split_layer:]
    else:
        del llava_model.model.layers[:args.split_layer]
        del llava_model.model.embed_tokens
        del llava_model.get_model().vision_tower
        del llava_model.get_model().mm_projector
    torch.cuda.empty_cache()

    # ── Data (rank 0 only) ────────────────────────────────────────────────────
    if rank == 0:
        data_args = DataArguments(
            data_path=args.data_path,
            image_folder=args.image_folder,
            lazy_preprocess=True,
            image_aspect_ratio="pad",
        )
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        peft_model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
        peft_model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        peft_model.config.image_aspect_ratio = data_args.image_aspect_ratio

        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        loader = DataLoader(
            data_module["train_dataset"],
            batch_size=args.per_device_batch_size,
            sampler=SequentialSampler(data_module["train_dataset"]),
            num_workers=args.dataloader_num_workers,
            collate_fn=data_module["data_collator"],
            pin_memory=True,
        )
        n_batches = len(loader)

    # Broadcast n_batches so rank 1 knows how many steps to run
    nb_t = torch.tensor([n_batches if rank == 0 else 0], dtype=torch.long)
    dist.broadcast(nb_t, src=0)
    n_batches = nb_t.item()

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    total_steps  = (n_batches * args.num_train_epochs) // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer    = build_optimizer(stage, args.learning_rate, args.mm_projector_lr, args.weight_decay)
    scheduler    = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    grad_scale   = 1.0 / args.gradient_accumulation_steps

    log.info(f"[rank {rank}] {n_batches} batches/epoch | {total_steps} total steps")

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    stage.train()
    global_step  = 0
    running_loss = 0.0
    t0           = time.time()
    train_start  = time.time()

    log.info(
        f"[rank {rank}] Training start | epochs={args.num_train_epochs} | "
        f"steps={total_steps} | effective_batch="
        f"{args.per_device_batch_size * args.gradient_accumulation_steps}"
    )

    for epoch in range(args.num_train_epochs):
        if rank == 0:
            data_iter = iter(loader)

        for step in range(n_batches):
            if rank == 0:
                batch = next(data_iter)
                iids  = batch["input_ids"].to(device)
                amask = batch["attention_mask"].to(device)
                lbls  = batch["labels"].to(device)
                imgs  = batch.get("images")
                if imgs is not None:
                    imgs = (imgs.to(device, dtype=dtype)
                            if isinstance(imgs, torch.Tensor)
                            else [i.to(device, dtype=dtype) for i in imgs])
                full   = dict(input_ids=iids, attention_mask=amask, labels=lbls, images=imgs)
                chunks = split_batch(full, args.num_micro_batches)

            if rank == 0:
                pipeline_step_rank0(stage, chunks, grad_scale, dtype)
                step_loss = None
            else:
                step_loss = pipeline_step_rank1(stage, grad_scale, dtype, device)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(stage.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 1 and global_step % args.logging_steps == 0:
                    running_loss += step_loss
                    avg = running_loss / args.logging_steps
                    log.info(
                        f"epoch {epoch+1} | step {global_step}/{total_steps} | "
                        f"loss {avg:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                        f"{time.time() - t0:.0f}s elapsed"
                    )
                    running_loss = 0.0
                    t0 = time.time()
                elif rank == 1:
                    running_loss += step_loss

    total_train_s = time.time() - train_start
    log.info(
        f"[rank {rank}] Training finished | steps={global_step}/{total_steps} | "
        f"train_wall_s={total_train_s:.1f} | train_wall_min={total_train_s / 60:.2f}"
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    dist.barrier()
    stage.cpu()
    if rank == 0:
        peft_model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        log.info(f"[rank 0] Saved adapter to {args.output_dir}")
    dist.barrier()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path",          required=True)
    p.add_argument("--data_path",                   required=True)
    p.add_argument("--image_folder",                required=True)
    p.add_argument("--output_dir",                  default="checkpoints/llava-stage2-pipe-lora-dist")
    p.add_argument("--num_train_epochs",            type=int,   default=1)
    p.add_argument("--per_device_batch_size",       type=int,   default=4)
    p.add_argument("--gradient_accumulation_steps", type=int,   default=4)
    p.add_argument("--num_micro_batches",           type=int,   default=4)
    p.add_argument("--split_layer",                type=int,   default=16)
    p.add_argument("--learning_rate",              type=float, default=2e-4)
    p.add_argument("--mm_projector_lr",            type=float, default=2e-5)
    p.add_argument("--weight_decay",               type=float, default=0.0)
    p.add_argument("--warmup_ratio",               type=float, default=0.03)
    p.add_argument("--bf16",                       action="store_true", default=True)
    p.add_argument("--model_max_length",           type=int,   default=2048)
    p.add_argument("--lora_r",                     type=int,   default=128)
    p.add_argument("--lora_alpha",                 type=int,   default=256)
    p.add_argument("--lora_dropout",               type=float, default=0.05)
    p.add_argument("--dataloader_num_workers",     type=int,   default=0)
    p.add_argument("--logging_steps",              type=int,   default=1)
    return p.parse_args()


if __name__ == "__main__":
    dist.init_process_group(backend="gloo")   # gloo works over plain Ethernet
    args = parse_args()
    try:
        train(args)
    finally:
        dist.destroy_process_group()
