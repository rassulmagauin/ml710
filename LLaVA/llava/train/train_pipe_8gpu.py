#!/usr/bin/env python3
"""
train_pipe_8gpu.py — 8-stage GPipe pipeline parallelism for LLaVA Stage 2 LoRA.

All 8 GPUs are on the same node (gpu partition), so inter-stage transfers use
fast PCIe / NVLink — no network overhead.

Stage layout (32 Llama decoder layers split evenly, 4 per GPU):
  GPU 0 : vision tower + mm_projector + embed_tokens + layers  0– 3
  GPU 1 : layers  4– 7
  GPU 2 : layers  8–11
  GPU 3 : layers 12–15
  GPU 4 : layers 16–19
  GPU 5 : layers 20–23
  GPU 6 : layers 24–27
  GPU 7 : layers 28–31 + RMSNorm + lm_head + CE loss

GPipe schedule: batch → K micro-batches → forward all → backward all.
Pipeline bubble = (N-1)/K, so we default to K=16 for ~44% bubble with N=8.
With K=32 the bubble drops to ~22% at the cost of more activation memory.

Single process, no torch.distributed needed.
"""

import argparse
import logging
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

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
NUM_LLAMA_LAYERS = 32


# ══════════════════════════════════════════════════════════════════════════════
# Stage modules
# ══════════════════════════════════════════════════════════════════════════════

class FirstStage(nn.Module):
    """GPU 0: vision encoding, multimodal fusion, first layers_per_stage decoder layers."""

    def __init__(self, llava_model: LlavaLlamaForCausalLM, layer_start: int, layer_end: int):
        super().__init__()
        inner = llava_model.get_model()
        self.vision_tower  = inner.get_vision_tower()
        self.mm_projector  = inner.mm_projector
        self.embed_tokens  = inner.embed_tokens
        self.layers        = nn.ModuleList(inner.layers[layer_start:layer_end])
        self._llava_causal = llava_model
        self._llama_model  = inner

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        images: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        dev = next(self.embed_tokens.parameters()).device

        (_, position_ids, attention_mask, _, inputs_embeds, labels) = (
            self._llava_causal.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, labels, images, None
            )
        )
        bsz, seq_len, _ = inputs_embeds.shape
        attn_4d = self._llama_model._prepare_decoder_attention_mask(
            attention_mask, (bsz, seq_len), inputs_embeds, 0
        )
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=dev).unsqueeze(0).expand(bsz, -1)

        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, attention_mask=attn_4d, position_ids=position_ids)[0]

        return h, attn_4d, position_ids, labels


class MidStage(nn.Module):
    """Intermediate GPU: a slice of decoder layers, passes tensors straight through."""

    def __init__(self, llava_model: LlavaLlamaForCausalLM, layer_start: int, layer_end: int):
        super().__init__()
        self.layers = nn.ModuleList(llava_model.get_model().layers[layer_start:layer_end])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_4d: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        h = hidden_states
        for layer in self.layers:
            h = layer(h, attention_mask=attn_4d, position_ids=position_ids)[0]
        return h, attn_4d, position_ids, labels


class LastStage(nn.Module):
    """Final GPU: last decoder layers + RMSNorm + lm_head + CE loss."""

    def __init__(self, llava_model: LlavaLlamaForCausalLM, layer_start: int, layer_end: int):
        super().__init__()
        inner = llava_model.get_model()
        self.layers   = nn.ModuleList(inner.layers[layer_start:layer_end])
        self.norm     = inner.norm
        self.lm_head  = llava_model.lm_head

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_4d: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        h = hidden_states
        for layer in self.layers:
            h = layer(h, attention_mask=attn_4d, position_ids=position_ids)[0]

        logits = self.lm_head(self.norm(h))
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(h.device)
        return nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline model
# ══════════════════════════════════════════════════════════════════════════════

class LlavaPipeline8GPU(nn.Module):
    """
    8-stage pipeline. Holds one stage per GPU as an nn.ModuleList so that
    save/load and parameter iteration work naturally.
    """

    def __init__(self, llava_model: LlavaLlamaForCausalLM, num_gpus: int = 8):
        super().__init__()
        assert NUM_LLAMA_LAYERS % num_gpus == 0, \
            f"num_gpus ({num_gpus}) must divide {NUM_LLAMA_LAYERS} evenly"
        self.num_gpus = num_gpus
        lps = NUM_LLAMA_LAYERS // num_gpus   # layers per stage

        stages: List[nn.Module] = []
        for i in range(num_gpus):
            start, end = i * lps, (i + 1) * lps
            if i == 0:
                stage = FirstStage(llava_model, start, end)
            elif i == num_gpus - 1:
                stage = LastStage(llava_model, start, end)
            else:
                stage = MidStage(llava_model, start, end)

            dtype = next(llava_model.parameters()).dtype
            stage = stage.to(dtype).to(torch.device(f"cuda:{i}"))
            stages.append(stage)
            log.info(f"  GPU {i}: layers {start}–{end - 1}"
                     + (" + vision/embed" if i == 0 else "")
                     + (" + norm/lm_head" if i == num_gpus - 1 else ""))

        self.stages = nn.ModuleList(stages)

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        images: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Single micro-batch forward. Returns scalar loss on GPU N-1."""
        dev0 = torch.device("cuda:0")

        # Stage 0
        h, attn_4d, pos_ids, lbls = self.stages[0](
            input_ids.to(dev0),
            attention_mask.to(dev0),
            labels.to(dev0),
            images.to(dev0) if isinstance(images, torch.Tensor) else
            [img.to(dev0) for img in images] if images is not None else None,
        )

        # Intermediate + last stages
        for i, stage in enumerate(self.stages[1:], start=1):
            dev = torch.device(f"cuda:{i}")
            # .to(dev) is tracked by autograd → gradients flow back automatically
            h      = h.to(dev)
            attn_4d = attn_4d.to(dev) if attn_4d is not None else None
            pos_ids = pos_ids.to(dev) if pos_ids is not None else None
            lbls    = lbls.to(dev)

            if i == self.num_gpus - 1:
                return stage(h, attn_4d, pos_ids, lbls)   # scalar loss
            else:
                h, attn_4d, pos_ids, lbls = stage(h, attn_4d, pos_ids, lbls)

        raise RuntimeError("unreachable")

    def forward(
        self,
        batch: Dict,
        num_micro_batches: int,
        grad_accum_steps: int,
    ) -> torch.Tensor:
        """GPipe: forward all micro-batches, return mean loss (ready for .backward())."""
        chunks = split_batch(batch, num_micro_batches)
        scale  = len(chunks) * grad_accum_steps

        losses = [self.forward_chunk(
            c["input_ids"], c["attention_mask"], c["labels"], c["images"]
        ) / scale for c in chunks]

        return sum(losses)


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


def build_optimizer(pipeline: LlavaPipeline8GPU, lr: float,
                    mm_projector_lr: float, weight_decay: float):
    stage0      = pipeline.stages[0]
    proj_ids    = {id(p) for p in stage0.mm_projector.parameters()}

    groups = [
        {   # mm_projector gets its own (lower) LR
            "params": [p for p in stage0.mm_projector.parameters() if p.requires_grad],
            "lr": mm_projector_lr,
        },
        {   # everything else trainable across all stages
            "params": [
                p for p in pipeline.parameters()
                if id(p) not in proj_ids and p.requires_grad
            ],
            "lr": lr,
        },
    ]
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    assert torch.cuda.device_count() >= args.num_gpus, (
        f"Need {args.num_gpus} GPUs, found {torch.cuda.device_count()}"
    )
    dtype = torch.bfloat16 if args.bf16 else torch.float32

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
    log.info("Loading LLaVA model on CPU …")
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
        p.requires_grad_(False)   # always frozen in Stage 2

    # ── LoRA ──────────────────────────────────────────────────────────────────
    log.info("Applying LoRA …")
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
    )
    peft_model  = get_peft_model(base_model, lora_cfg)
    peft_model.print_trainable_parameters()
    llava_model = peft_model.base_model.model

    # ── Build 8-stage pipeline (distributes layers to GPUs 0–7) ───────────────
    log.info(f"Building {args.num_gpus}-stage pipeline …")
    pipeline = LlavaPipeline8GPU(llava_model, num_gpus=args.num_gpus)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_args = DataArguments(
        data_path=args.data_path,
        image_folder=args.image_folder,
        lazy_preprocess=True,
        image_aspect_ratio="pad",
    )
    data_args.image_processor   = vision_tower.image_processor
    data_args.is_multimodal      = True
    peft_model.config.mm_use_im_start_end   = model_args.mm_use_im_start_end
    peft_model.config.mm_use_im_patch_token  = model_args.mm_use_im_patch_token
    peft_model.config.image_aspect_ratio     = data_args.image_aspect_ratio

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    loader = DataLoader(
        data_module["train_dataset"],
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_module["data_collator"],
        pin_memory=True,
    )

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    total_steps  = (len(loader) * args.num_train_epochs) // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer    = build_optimizer(pipeline, args.learning_rate, args.mm_projector_lr, args.weight_decay)
    scheduler    = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log.info(
        f"Training: {len(data_module['train_dataset'])} samples | {total_steps} steps | "
        f"batch={args.per_device_batch_size} × {args.num_micro_batches} micro-batches | "
        f"grad_accum={args.gradient_accumulation_steps} | "
        f"effective_batch={args.per_device_batch_size * args.gradient_accumulation_steps}"
    )
    log.info(f"Pipeline bubble fraction ≈ {(args.num_gpus - 1) / args.num_micro_batches:.1%}")

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pipeline.train()
    global_step  = 0
    running_loss = 0.0
    t0           = time.time()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(loader):
            full = {
                "input_ids":      batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels":         batch["labels"],
                "images":         batch.get("images"),
            }
            if isinstance(full["images"], torch.Tensor):
                full["images"] = full["images"].to(dtype=dtype)

            loss = pipeline(full, args.num_micro_batches, args.gradient_accumulation_steps)
            loss.backward()
            running_loss += loss.item() * args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(pipeline.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg = running_loss / args.gradient_accumulation_steps
                    log.info(
                        f"epoch {epoch + 1} | step {global_step}/{total_steps} | "
                        f"loss {avg:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                        f"{time.time() - t0:.0f}s"
                    )
                    running_loss = 0.0
                    t0 = time.time()

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info(f"Saving LoRA adapter to {args.output_dir} …")
    for stage in pipeline.stages:
        stage.cpu()
    peft_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log.info("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLaVA 8-GPU pipeline-parallel LoRA fine-tuning")
    p.add_argument("--model_name_or_path",          required=True)
    p.add_argument("--data_path",                   required=True)
    p.add_argument("--image_folder",                required=True)
    p.add_argument("--output_dir",                  default="checkpoints/llava-stage2-pipe-8gpu-lora")
    p.add_argument("--num_gpus",                    type=int,   default=8)
    p.add_argument("--num_train_epochs",            type=int,   default=1)
    p.add_argument("--per_device_batch_size",       type=int,   default=16,
                   help="Total pipeline batch (split into num_micro_batches)")
    p.add_argument("--gradient_accumulation_steps", type=int,   default=1)
    p.add_argument("--num_micro_batches",           type=int,   default=16,
                   help="GPipe chunks; bubble = (num_gpus-1)/num_micro_batches")
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
    train(parse_args())
