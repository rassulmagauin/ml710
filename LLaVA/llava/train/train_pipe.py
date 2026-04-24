#!/usr/bin/env python3
"""
train_pipe.py — 2-stage GPipe pipeline parallelism for LLaVA Stage 2 LoRA fine-tuning.

Stage 0 (GPU 0): vision tower + mm_projector + embed_tokens + decoder layers [0 : split_layer]
Stage 1 (GPU 1): decoder layers [split_layer : 32] + RMSNorm + lm_head + loss

Schedule: synchronous GPipe — all micro-batch forwards, then loss.backward() lets
autograd trace back through the GPU0→GPU1 tensor transfer automatically.
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

# ── Path setup ────────────────────────────────────────────────────────────────
_LLAVA_ROOT = pathlib.Path(__file__).resolve().parents[2]  # .../LLaVA/
sys.path.insert(0, str(_LLAVA_ROOT))

from llava.model import LlavaLlamaForCausalLM                            # noqa: E402
from llava.train.train import DataArguments, ModelArguments              # noqa: E402
from llava.train.train import make_supervised_data_module                # noqa: E402
from llava import conversation as conversation_lib                        # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

IGNORE_INDEX = -100
GPU0 = torch.device("cuda:0")
GPU1 = torch.device("cuda:1")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Stages
# ══════════════════════════════════════════════════════════════════════════════

class PipeStage0(nn.Module):
    """GPU 0 — vision encoding, multimodal fusion, first split_layer decoder layers."""

    def __init__(self, llava_model: LlavaLlamaForCausalLM, split_layer: int):
        super().__init__()
        inner = llava_model.get_model()       # LlavaLlamaModel
        self.vision_tower = inner.get_vision_tower()
        self.mm_projector = inner.mm_projector
        self.embed_tokens = inner.embed_tokens
        self.layers = nn.ModuleList(inner.layers[:split_layer])
        # Keep references for prepare_inputs_labels_for_multimodal and mask prep
        self._llava_causal = llava_model      # LlavaLlamaForCausalLM
        self._llama_model = inner             # LlavaLlamaModel (inherits LlamaModel)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        images: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        # 1. Multimodal fusion: embeds vision tokens into the sequence
        #    Returns inputs_embeds (already embedded), updated mask/pos_ids/labels
        (_, position_ids, attention_mask, _, inputs_embeds, labels) = (
            self._llava_causal.prepare_inputs_labels_for_multimodal(
                input_ids, None, attention_mask, None, labels, images, None
            )
        )

        bsz, seq_len, _ = inputs_embeds.shape

        # 2. Build 4-D causal mask expected by LlamaDecoderLayer
        attn_4d = self._llama_model._prepare_decoder_attention_mask(
            attention_mask, (bsz, seq_len), inputs_embeds, 0
        )

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=GPU0).unsqueeze(0).expand(bsz, -1)

        # 3. Decoder layers 0 … split_layer-1
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, attention_mask=attn_4d, position_ids=position_ids)[0]

        return h, attn_4d, position_ids, labels


class PipeStage1(nn.Module):
    """GPU 1 — remaining decoder layers, RMSNorm, lm_head, cross-entropy loss."""

    def __init__(self, llava_model: LlavaLlamaForCausalLM, split_layer: int):
        super().__init__()
        inner = llava_model.get_model()
        self.layers = nn.ModuleList(inner.layers[split_layer:])
        self.norm = inner.norm
        self.lm_head = llava_model.lm_head

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        h = hidden_states
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask, position_ids=position_ids)[0]

        h = self.norm(h)
        logits = self.lm_head(h)        # (B, L, V)

        # Next-token CE loss; ignore padding / image tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(GPU1)
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )
        return loss


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def split_batch(batch: Dict, num_chunks: int) -> List[Dict]:
    """Divide a batch into at most num_chunks micro-batches."""
    bsz = batch["input_ids"].size(0)
    chunk_size = max(1, (bsz + num_chunks - 1) // num_chunks)
    chunks = []
    images = batch.get("images")
    for start in range(0, bsz, chunk_size):
        end = min(start + chunk_size, bsz)
        chunks.append({
            "input_ids":      batch["input_ids"][start:end],
            "attention_mask": batch["attention_mask"][start:end],
            "labels":         batch["labels"][start:end],
            "images":         images[start:end] if images is not None else None,
        })
    return chunks


def build_optimizer(stage0, stage1, lr, mm_projector_lr, weight_decay):
    projector_ids = {id(p) for p in stage0.mm_projector.parameters()}
    param_groups = [
        {
            "params": [
                p for p in stage0.parameters()
                if id(p) not in projector_ids and p.requires_grad
            ],
            "lr": lr,
        },
        {
            "params": [p for p in stage0.mm_projector.parameters() if p.requires_grad],
            "lr": mm_projector_lr,
        },
        {
            "params": [p for p in stage1.parameters() if p.requires_grad],
            "lr": lr,
        },
    ]
    param_groups = [g for g in param_groups if g["params"]]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
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
    log.info("Loading LLaVA model …")
    base_model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="eager",   # flash-attn not compatible with manual pipeline
    )
    base_model.config.use_cache = False

    # Initialise vision modules (sets config attrs; loads CLIP if not cached)
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
        p.requires_grad_(False)   # vision tower is always frozen in Stage 2

    # ── LoRA ──────────────────────────────────────────────────────────────────
    log.info("Applying LoRA …")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    peft_model.print_trainable_parameters()

    # Unwrap to access submodules directly
    llava_model = peft_model.base_model.model   # LlavaLlamaForCausalLM (with LoRA injected)

    # ── Pipeline stages ───────────────────────────────────────────────────────
    log.info(f"Splitting at layer {args.split_layer}: "
             f"layers 0–{args.split_layer-1} on GPU 0, "
             f"layers {args.split_layer}–31 on GPU 1")
    stage0 = PipeStage0(llava_model, args.split_layer).to(dtype).to(GPU0)
    stage1 = PipeStage1(llava_model, args.split_layer).to(dtype).to(GPU1)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_args = DataArguments(
        data_path=args.data_path,
        image_folder=args.image_folder,
        lazy_preprocess=True,
        image_aspect_ratio="pad",
    )
    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True
    peft_model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    peft_model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    peft_model.config.image_aspect_ratio = data_args.image_aspect_ratio

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
    total_steps = (len(loader) * args.num_train_epochs) // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = build_optimizer(stage0, stage1, args.learning_rate, args.mm_projector_lr, args.weight_decay)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info(f"Training: {len(data_module['train_dataset'])} samples | "
             f"{total_steps} steps | {args.num_micro_batches} micro-batches/step")

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    stage0.train()
    stage1.train()

    global_step = 0
    running_loss = 0.0
    t0 = time.time()

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(loader):
            # ── Move batch to GPU 0 ───────────────────────────────────────────
            input_ids      = batch["input_ids"].to(GPU0)
            attention_mask = batch["attention_mask"].to(GPU0)
            labels         = batch["labels"].to(GPU0)
            images         = batch.get("images")
            if images is not None:
                if isinstance(images, torch.Tensor):
                    images = images.to(GPU0, dtype=dtype)
                else:
                    images = [img.to(GPU0, dtype=dtype) for img in images]

            # ── GPipe: all-forward over micro-batches ─────────────────────────
            full = dict(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels, images=images)
            chunks = split_batch(full, args.num_micro_batches)
            n = len(chunks)

            step_loss = torch.zeros(1, device=GPU1)
            for chunk in chunks:
                # Stage 0 forward on GPU 0
                h, attn_4d, pos_ids, chunk_labels = stage0(
                    chunk["input_ids"],
                    chunk["attention_mask"],
                    chunk["labels"],
                    chunk["images"],
                )
                # Stage 1 forward on GPU 1
                # PyTorch autograd tracks the .to() transfer → grads flow back to GPU 0
                loss = stage1(
                    h.to(GPU1),
                    attn_4d.to(GPU1) if attn_4d is not None else None,
                    pos_ids.to(GPU1) if pos_ids is not None else None,
                    chunk_labels,
                )
                step_loss = step_loss + loss / (n * args.gradient_accumulation_steps)

            # Backward through both stages
            step_loss.backward()
            running_loss += step_loss.item() * args.gradient_accumulation_steps

            # ── Optimizer step ────────────────────────────────────────────────
            if (step + 1) % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(
                    list(stage0.parameters()) + list(stage1.parameters()), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    elapsed = time.time() - t0
                    avg_loss = running_loss / args.gradient_accumulation_steps
                    lr_now = scheduler.get_last_lr()[0]
                    log.info(
                        f"epoch {epoch+1} | step {global_step}/{total_steps} | "
                        f"loss {avg_loss:.4f} | lr {lr_now:.2e} | {elapsed:.0f}s elapsed"
                    )
                    running_loss = 0.0
                    t0 = time.time()

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info(f"Saving LoRA adapter to {args.output_dir} …")
    stage0.cpu()
    stage1.cpu()
    peft_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log.info("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLaVA pipeline-parallel LoRA fine-tuning")
    p.add_argument("--model_name_or_path",         required=True)
    p.add_argument("--data_path",                  required=True)
    p.add_argument("--image_folder",               required=True)
    p.add_argument("--output_dir",                 default="checkpoints/llava-stage2-pipe-lora")
    p.add_argument("--num_train_epochs",           type=int,   default=1)
    p.add_argument("--per_device_batch_size",      type=int,   default=4,
                   help="Total pipeline batch size (split into num_micro_batches chunks)")
    p.add_argument("--gradient_accumulation_steps",type=int,   default=4)
    p.add_argument("--num_micro_batches",          type=int,   default=4,
                   help="GPipe chunks; per_device_batch_size must be >= num_micro_batches")
    p.add_argument("--split_layer",               type=int,   default=16,
                   help="First layer index assigned to Stage 1 (GPU 1)")
    p.add_argument("--learning_rate",             type=float, default=2e-4)
    p.add_argument("--mm_projector_lr",           type=float, default=2e-5)
    p.add_argument("--weight_decay",              type=float, default=0.0)
    p.add_argument("--warmup_ratio",              type=float, default=0.03)
    p.add_argument("--bf16",                      action="store_true", default=True)
    p.add_argument("--model_max_length",          type=int,   default=2048)
    p.add_argument("--lora_r",                    type=int,   default=128)
    p.add_argument("--lora_alpha",                type=int,   default=256)
    p.add_argument("--lora_dropout",              type=float, default=0.05)
    p.add_argument("--dataloader_num_workers",    type=int,   default=0)
    p.add_argument("--logging_steps",             type=int,   default=1)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
