"""
Build the ML710 LLaVA parallelism presentation (PPTX).

Section attribution:
  Pipeline parallelism            -> Youssef Ghallab
  ZeRO-1 + ZeRO-2 + ZeRO-3        -> Omar Ahmed
  FSDP + Tensor Parallel          -> Rassul Magauin

Run:
    python scripts/build_presentation.py
Output:
    presentation/ML710_LLaVA_Parallelism.pptx
"""

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR


PROJECT = Path(__file__).resolve().parents[1]
PLOTS = PROJECT / "plots" / "comparison"
RUNS = PROJECT / "logs" / "runs"
OUT_DIR = PROJECT / "presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "ML710_LLaVA_Parallelism.pptx"

# 16:9 (matches example deck)
prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)

BLANK = prs.slide_layouts[6]

GREY_TXT = RGBColor(0x55, 0x55, 0x55)
DARK_TXT = RGBColor(0x10, 0x10, 0x10)


def add_title(slide, text, top=Inches(0.4), left=Inches(0.6),
              width=Inches(12.1), height=Inches(0.9), size=34, bold=True,
              align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = DARK_TXT
    return box


def add_bullets(slide, items, left=Inches(0.6), top=Inches(1.5),
                width=Inches(6.5), height=Inches(5.6), base_size=20,
                centered_vertically=True):
    """Items: list[str] or list[(text, level)] tuples."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    if centered_vertically:
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        # Use a real bullet at level 0, en-dash at level 1
        bullet = "•  " if level == 0 else "–  "
        prefix = "    " * level + bullet
        run = p.add_run()
        run.text = prefix + text
        run.font.size = Pt(base_size - 3 * level)
        run.font.color.rgb = DARK_TXT if level == 0 else GREY_TXT
        p.space_after = Pt(8 if level == 0 else 4)
    return box


def add_image(slide, path, left, top, width=None, height=None):
    if not Path(path).exists():
        return None
    if width and height:
        return slide.shapes.add_picture(str(path), left, top, width=width, height=height)
    if width:
        return slide.shapes.add_picture(str(path), left, top, width=width)
    if height:
        return slide.shapes.add_picture(str(path), left, top, height=height)
    return slide.shapes.add_picture(str(path), left, top)


def title_slide(title, subtitle, footer=None):
    s = prs.slides.add_slide(BLANK)
    add_title(s, title, top=Inches(2.5), height=Inches(1.5),
              size=56, bold=True, align=PP_ALIGN.CENTER)
    sub = s.shapes.add_textbox(Inches(0.5), Inches(4.4), Inches(12.3), Inches(2.2))
    tf = sub.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = subtitle
    r.font.size = Pt(24)
    r.font.color.rgb = GREY_TXT
    if footer:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r = p2.add_run()
        r.text = footer
        r.font.size = Pt(18)
        r.font.color.rgb = GREY_TXT
    return s


def section_slide(text):
    s = prs.slides.add_slide(BLANK)
    add_title(s, text, top=Inches(3.1), height=Inches(1.4),
              size=54, bold=False, align=PP_ALIGN.CENTER)
    return s


def split_slide(title, bullets, image_path, image_left=Inches(7.4),
                image_top=Inches(1.6), image_width=Inches(5.5),
                bullet_width=Inches(6.4), base_size=20):
    """Bullets on the left, image on the right."""
    s = prs.slides.add_slide(BLANK)
    add_title(s, title)
    add_bullets(s, bullets, width=bullet_width, base_size=base_size)
    if image_path and Path(image_path).exists():
        add_image(s, image_path, image_left, image_top, width=image_width)
    return s


def text_only(title, bullets, base_size=22, centered_vertically=True):
    s = prs.slides.add_slide(BLANK)
    add_title(s, title)
    add_bullets(s, bullets, width=Inches(12.1), base_size=base_size,
                centered_vertically=centered_vertically)
    return s


def image_focus(title, image_path, takeaway, image_top=Inches(1.9),
                image_width=Inches(8.5), image_left=None):
    """One central plot with a single takeaway line under the title."""
    s = prs.slides.add_slide(BLANK)
    add_title(s, title)
    # Subtitle / takeaway — allow up to 2 wrapped lines
    if takeaway:
        sub = s.shapes.add_textbox(Inches(0.6), Inches(1.15),
                                   Inches(12.1), Inches(0.75))
        tf = sub.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = takeaway
        r.font.size = Pt(16)
        r.font.italic = True
        r.font.color.rgb = GREY_TXT
    # Center the plot
    if image_left is None:
        image_left = Inches((13.33 - image_width.inches) / 2)
    add_image(s, image_path, image_left, image_top, width=image_width)
    return s


# ============================================================
# Build the deck
# ============================================================

# 1. Title
title_slide(
    "Parallelizing LLaVA",
    "ML710 course project · MBZUAI",
    "Rassul Magauin · Omar Ahmed · Youssef Ghallab\nSpring 2026",
)

# 2. Hardware & team
text_only(
    "Setup",
    [
        "MBZUAI HPC, ws-ia partition: 2 nodes × 1 RTX 5000 Ada (32 GB), Ethernet ~1 Gbps.",
        ("No NVLink, no InfiniBand. Cross-node comm goes over a slow link.", 1),
        ("The gpu partition has a 1-GPU-per-user QOS, so multi-GPU = two salloc sessions + torchrun.", 1),
        "Software: PyTorch 2.1.2, DeepSpeed 0.12.6, Transformers 4.37.2, peft 0.6.0.",
        "Team of three, each owning at least one non-trivial parallelism strategy:",
        ("Pipeline parallelism — Youssef Ghallab", 1),
        ("DeepSpeed ZeRO-1, ZeRO-2, and ZeRO-3 — Omar Ahmed", 1),
        ("PyTorch FSDP and Tensor Parallelism — Rassul Magauin", 1),
    ],
    base_size=22,
)

# 3. ML task
text_only(
    "The model: LLaVA",
    [
        'Liu et al., "Visual Instruction Tuning" (NeurIPS 2023). Multimodal: image + text → text.',
        "About 7B parameters total, in three pieces:",
        ("CLIP ViT-L/14-336 vision encoder — frozen.", 1),
        ("2-layer MLP projection — small, trainable.", 1),
        ("Vicuna-7B language model — frozen in Stage 1, fine-tuned in Stage 2.", 1),
        "Two-stage training:",
        ("Stage 1 trains only the projection (~80 MB) on 558K image-caption pairs.", 1),
        ("Stage 2 fine-tunes the projection plus the LLM on 665K instructions.", 1),
        "We use Stage 2 because it has meaningful gradient communication. Stage 1's AllReduce is too small to matter.",
    ],
    base_size=22,
)

# 4. Stage 2 setup
text_only(
    "Stage 2 — what we trained",
    [
        "Starting checkpoint: liuhaotian/llava-v1.5-7b. The CLIP and MLP were already trained in Stage 1.",
        "Dataset: 10K random samples from llava_instruct_150k.json with COCO 2014 train images. One epoch keeps each run under an hour.",
        "Training target: LoRA adapters (r=128, α=256) on the LLM, plus the full MLP projection.",
        ("≈4.6% of total parameters — about 340M of 7.4B.", 1),
        "AdamW, bf16, gradient checkpointing, cosine schedule.",
        "Effective batch sizes vary so each method actually fits in memory:",
        ("DDP, ZeRO-1, ZeRO-2: 32 (per_device=16, world=2)", 1),
        ("Pipeline: 16 (per_device=1, accum=16, world=2)", 1),
        ("ZeRO-3, FSDP, TP: 8 (per_device=1, accum=4, world=2)", 1),
        ("TP trains the projection only — no LoRA, since LoRA on TP-sharded base linears is fragile in PyTorch 2.1.", 1),
    ],
    base_size=22,
)

# 5. Strategies overview
text_only(
    "What we compared",
    [
        "Single-GPU LoRA — reference for what no parallelism looks like.",
        "DDP (DeepSpeed ZeRO-0) — naïve baseline; the model is replicated, gradients are AllReduce'd.",
        "Pipeline parallelism — the model is split across the two nodes layer-wise. Owned by Ghallab.",
        "ZeRO-1 — DDP with sharded optimizer states. Owned by Omar.",
        "ZeRO-2 — same plus sharded gradients. Owned by Omar.",
        "ZeRO-3 — same plus sharded parameters. Owned by Omar.",
        "PyTorch FSDP — PyTorch-native equivalent of ZeRO-3. Owned by Rassul.",
        "Tensor Parallel — splits each layer's matmuls across the two nodes. Owned by Rassul.",
        "Every multi-node run uses the same two ws-ia nodes, same Ethernet link.",
    ],
    base_size=22,
)

# 6. Single-GPU baseline
text_only(
    "Baseline 1: single GPU",
    [
        "One RTX 5000 Ada, batch size 1, 10000 steps for one epoch over the 10K subset.",
        "Wall time: 96 minutes.",
        "Throughput: 1.74 samples/sec.",
        "Peak GPU memory: 25.7 GiB out of 32. Fits comfortably.",
        "Average GPU utilization: 11%.",
        ("CPU-bound on data preprocessing — dataloader_num_workers=0, single-threaded image loading.", 1),
        ("The GPU spends most of its time waiting for the next batch.", 1),
        "Final loss: 1.32 (avg 1.10 across the run).",
        "Reference for what 'no parallelism' looks like before we add anything.",
    ],
    base_size=22,
)

# 7. DDP
split_slide(
    "Baseline 2: DDP across two nodes",
    [
        "PyTorch DistributedDataParallel via torchrun --nnodes=2.",
        ("DeepSpeed ZeRO-0 — no sharding, each rank holds the whole model.", 1),
        "Per-device batch 16, effective batch 32, 313 steps for one epoch.",
        "Wall time: 74.8 min — about 1.3× faster than single GPU.",
        "Throughput: 2.22 samples/sec.",
        "Peak GPU memory: 30.7 GiB.",
        "GPU utilization: 70.7%.",
        ("Visible AllReduce stalls when both nodes wait on cross-node Ethernet.", 1),
        "Final loss: 1.04. This is the reference everything else is compared against.",
    ],
    image_path=RUNS / "youssef.ghallab/stage2_ddp_lora_20260420_000617/plots/loss_curve.png",
    image_top=Inches(1.8), image_width=Inches(5.4),
)

# 8. Pipeline section
section_slide("Pipeline Parallelism")

# 10. Pipeline implementation
text_only(
    "How we built it (Ghallab)",
    [
        "Manual two-stage pipeline. We didn't use torchgpipe or DeepSpeed-Pipeline.",
        ("Custom training loop: LLaVA/llava/train/train_pipe_dist.py (~520 lines).", 1),
        ("Gloo backend for cross-node send/recv. NCCL had ordering issues with our mixed-dtype activations.", 1),
        "The model is cut at decoder layer k. There are 32 decoder layers in Vicuna-7B.",
        ("Rank 0 holds the vision tower (frozen), the MLP projection, and decoder layers 0..k-1.", 1),
        ("Rank 1 holds layers k..31, the final norm, the LM head, and computes loss.", 1),
        ("Activations flow forward, gradients flow back, by hand.", 1),
        "Gradient accumulation 16, microbatch 1 → effective batch 16. LoRA is added on each rank's local layers.",
    ],
    base_size=22,
)

# 11. Pipeline split balancing
text_only(
    "Finding a split that fits",
    [
        "Picking the layer index k matters more than expected.",
        "k = 16 (50/50 by layer count): rank 0 OOMs in the first forward.",
        ("Vision tower + image features sit on rank 0. With layers 0..15 it hits 31.4 GiB — over the 32 GiB card.", 1),
        "k = 8 (push work to rank 1): rank 1 OOMs.",
        ("LM head + final norm + layers 8..31 hit 31.0 GiB.", 1),
        "k = 12 with microbatch 1 + grad accum 16: both ranks fit.",
        ("Rank 1 ≈ 31.4 GiB, rank 0 ≈ 28 GiB. Run completes 625/625 steps.", 1),
        "Lesson: layer count is a poor proxy for memory cost. The vision tower and LM head are heavier than they look.",
    ],
    base_size=22,
)

# 12. Pipeline results
split_slide(
    "Pipeline — results",
    [
        "Wall time: 72.1 min. Almost identical to DDP.",
        "Throughput: 2.31 samples/sec — 1.04× DDP. Barely faster.",
        "Effective batch 16, half of DDP, so fewer steps per epoch.",
        "Peak GPU memory: 31.4 GiB on the heavier rank.",
        "GPU utilization is uneven: rank 0 at 34%, rank 1 at 56%.",
        ("Whichever rank finishes its layers first sits idle until the other catches up.", 1),
        ("Gloo is much slower than NCCL, so the comm gap is wider.", 1),
        "Final loss: 1.57 — higher than DDP because of the smaller effective batch.",
        "Pipeline made the model fit. It did not make it faster.",
    ],
    image_path=RUNS / "youssef.ghallab/stage2_pipe_lora_dist_20260424_153422_rank0/plots/gpu_utilization_curve.png",
    image_top=Inches(2.0), image_width=Inches(5.4),
    base_size=18,
)

# 13. ZeRO-2 section
section_slide("DeepSpeed ZeRO-1 / ZeRO-2")

# 14. ZeRO-2 explanation
text_only(
    "ZeRO-1 vs ZeRO-2 (Omar)",
    [
        "Both are data parallelism with sharded state. The difference is what gets sharded.",
        "ZeRO-1:",
        ("Optimizer states sharded. Gradients still AllReduce'd (like DDP), parameters replicated.", 1),
        ("After the optimizer step, AllGather to put updated parameters back on every rank.", 1),
        "ZeRO-2 adds gradient sharding via ReduceScatter:",
        ("Gradients land on their owner-rank shards directly, no AllReduce.", 1),
        ("Each rank takes one optimizer step on its shard, then AllGather params back.", 1),
        ("Comm volume per step is roughly 1.5× the model size — same as ZeRO-1, different schedule.", 1),
        "On a LoRA workload, both win mostly from DeepSpeed's bucketed async overlap, not from the sharding itself.",
        ("ZeRO-1 throughput is between DDP and ZeRO-2; ZeRO-2 is fastest because ReduceScatter overlaps better with backward.", 1),
    ],
    base_size=20,
)

# 15. ZeRO-1 / ZeRO-2 results
split_slide(
    "ZeRO-1 / ZeRO-2 — results",
    [
        "Both at effective batch 32, full epoch (313 steps).",
        "ZeRO-1: 65 min, 2.56 sps, 28.8 GiB, 81% util, loss 1.14 — 1.15× DDP.",
        "ZeRO-2: 58 min, 2.85 sps, 27.6 GiB, 89% util, loss 1.11 — 1.28× DDP.",
        ("The only method that beats plain DDP outright.", 1),
        "Why ZeRO-2 edges out ZeRO-1: ReduceScatter overlaps with the backward pass, AllReduce doesn't as cleanly.",
        "Why the speedup over DDP isn't 2×: LoRA's optimizer state is tiny — most of the win is overlap, not sharding.",
    ],
    image_path=RUNS / "youssef.ghallab/stage2_zero2_lora_20260424_134311/plots/gpu_utilization_curve.png",
    image_top=Inches(2.0), image_width=Inches(5.4),
    base_size=18,
)

# 16. ZeRO-3 section
section_slide("DeepSpeed ZeRO-3")

# 17. ZeRO-3 explanation
text_only(
    "ZeRO-3: shard everything (Omar)",
    [
        "Same data-parallel semantics as ZeRO-2. Each rank still sees a different microbatch.",
        "The difference: parameters are also sharded across the two ranks.",
        "Per-step comm now happens at every layer, twice:",
        ("Forward pass: AllGather parameters for layer N → compute → drop them.", 1),
        ("Backward pass: AllGather them again → compute gradients → drop again.", 1),
        ("ReduceScatter the gradients at the end of backward, then run the optimizer on each shard.", 1),
        "On 2 ranks, ~7 GB of parameters live on each. But every layer briefly re-materializes its full ~430 MB on every rank.",
        "Back-of-envelope per microbatch:",
        ("AllGather other rank's 7 GB shard, fwd + bwd → ~14 GB cross-node per microbatch.", 1),
        ("× 4 microbatches per optimizer step + ReduceScatter at end ≈ ~63 GB on the wire per step.", 1),
        ("Over 1 Gbps Ethernet that's minutes of pure comm — DeepSpeed bucketed-async overlap (50 MB) is the only thing keeping it tractable.", 1),
        "We used configs/zero3_bounded_async.json to cap allgather/reduce buckets.",
    ],
    base_size=18,
)

# 18. ZeRO-3 results
text_only(
    "ZeRO-3 — results",
    [
        "Hit the SLURM time limit before finishing. Reached 292 of 1250 steps in 3 h 52 min.",
        ("Each step took about 48 seconds. Projected full run: 16.5 hours.", 1),
        "Throughput: 0.17 samples/sec — about 13× slower than DDP.",
        "Peak GPU memory: 22 GiB — actual sharding showing up.",
        ("But memory was never the bottleneck on this card. Communication was.", 1),
        "GPU utilization: 99%. This is misleading.",
        ("DeepSpeed pipelines comm with compute, so 'utilization' includes time spent waiting on AllGather.", 1),
        ("High util doesn't mean fast — it means the engine kept the GPU busy on something.", 1),
        "Last logged loss: 1.31. Convergence was fine — the run was just too slow to finish.",
        "Diagnosis: per-layer AllGather over 1 Gbps Ethernet is the wall.",
    ],
    base_size=20,
)

# 19. FSDP section
section_slide("PyTorch FSDP")

# 20. FSDP explanation
text_only(
    "FSDP: PyTorch's take (Rassul)",
    [
        "PyTorch-native equivalent of ZeRO-3. Same algorithm, different implementation.",
        "Design choices that differ from DeepSpeed:",
        ("Each transformer block is wrapped as one FSDP unit (a contiguous FlatParameter).", 1),
        ("AllGather happens at the unit boundary, not per-tensor.", 1),
        ("Mixed precision is set explicitly via a policy (we used bf16).", 1),
        "Our config: full_shard auto_wrap on LlamaDecoderLayer, use_orig_params=True (required for LoRA), cpu_ram_efficient_loading=true.",
        "Two LLaVA-specific bugs we had to patch to get this running:",
        ("self.vision_tower[0] crashes when loading from a vendored checkpoint where vision_tower isn't a list.", 1),
        ("LoRA adapters are created in fp32 after the base model is cast to bf16. FSDP requires uniform dtype, so we cast LoRA to bf16 too.", 1),
    ],
    base_size=20,
)

# 21. FSDP results
split_slide(
    "FSDP — results",
    [
        "Ran to step 292 — matched ZeRO-3's stopping point — at 84 s/step.",
        "Wall time: 6.8 h. Throughput: 0.095 sps — ~22× slower than DDP, ~1.8× slower than ZeRO-3.",
        "Peak GPU memory: 31.6 GiB (right at the card's limit).",
        "GPU utilization: 97.5%. Final loss: 1.34 — matches ZeRO-3 within noise (same algorithm).",
        "Why slower than ZeRO-3 on the same hardware?",
        ("DeepSpeed's bounded async (50 MB buckets) overlaps comm with compute aggressively.", 1),
        ("FSDP's defaults wait for each block's AllGather to finish before computing.", 1),
        ("Same algorithm, different async behavior — the difference is real on a slow link.", 1),
    ],
    image_path=RUNS / "rassul.magauin/stage2_fsdp_lora_20260426_143948/plots/gpu_utilization_curve.png",
    image_top=Inches(2.0), image_width=Inches(5.4),
    base_size=18,
)

# Tensor Parallel section
section_slide("Tensor Parallelism")

# TP explanation
text_only(
    "Tensor Parallel: split the matmuls (Rassul)",
    [
        "Megatron-style intra-layer model parallelism. Each LlamaDecoderLayer's linear sublayers are sharded column-wise or row-wise across the two ranks.",
        "Sharding plan, applied per layer via torch.distributed.tensor.parallel.parallelize_module:",
        ("q_proj, k_proj, v_proj, gate_proj, up_proj — Colwise (output dim split).", 1),
        ("o_proj, down_proj — Rowwise (input dim split, AllReduce on output).", 1),
        ("After parallelize, num_heads / num_key_value_heads / hidden_size are halved per rank.", 1),
        "Per-step communication: ~6 cross-node AllReduces per layer × 32 layers × 2 (fwd+bwd) ≈ 384 collectives per microbatch, × 4 grad-accum.",
        "We trained the MLP projection only — LoRA on TP-sharded base linears is fragile in PyTorch 2.1.2 (DTensor + LoRA's lora_A/B reshape don't compose cleanly).",
        "Custom 2-rank torchrun launcher: LLaVA/llava/train/train_tp_dist.py + scripts/run_stage2_tp.sh.",
    ],
    base_size=18,
)

# TP results
split_slide(
    "TP — results",
    [
        "Ran a full epoch — 1250 steps × effective batch 8 — at 11 s/step.",
        "Wall time: 3.85 h. Throughput: 0.72 sps — 0.32× DDP, but 4× faster than ZeRO-3 and 8× faster than FSDP.",
        ("TP ships ~2 GB of activations per microbatch vs ZeRO-3's ~14 GB of weights — order of magnitude less.", 1),
        "Peak GPU memory: 30.6 GiB. GPU util: 77.7%. Final loss: 1.03 — matches DDP at full epoch.",
        "Takeaway: TP is the only model-parallel method actually usable on this link.",
        ("Still 3× slower than DDP — per-layer collectives accumulate Ethernet latency.", 1),
        ("Designed for NVLink, where AllReduce is essentially free.", 1),
    ],
    image_path=RUNS / "rassul.magauin/stage2_tp_20260427_021303/plots/gpu_utilization_curve.png",
    image_top=Inches(2.0), image_width=Inches(5.4),
    base_size=18,
)

# Comparison section
section_slide("Putting it all together")

# Throughput
image_focus(
    "Throughput by method",
    image_path=PLOTS / "1_throughput.png",
    takeaway="ZeRO-2 wins, ZeRO-1 close behind. ZeRO-3 / FSDP collapse on Ethernet; TP in the middle.",
)

# Memory
image_focus(
    "Peak GPU memory",
    image_path=PLOTS / "4_gpu_memory.png",
    takeaway="ZeRO-3 shards the most. FSDP and TP saturate the card. ZeRO-2 hits a sweet spot.",
)

# Goodput per second
image_focus(
    "Goodput — loss reduction per second",
    image_path=PLOTS / "9a_goodput_per_sec.png",
    takeaway="Full-epoch methods cluster on top; cross-layer-comm methods (ZeRO-3 / FSDP / TP) sit below, ordered by throughput.",
)

# Loss curves
image_focus(
    "Loss vs step (smoothed)",
    image_path=PLOTS / "6_loss_vs_step.png",
    takeaway="All methods follow the same loss-vs-step curve at matched batch. TP's curve extends further because its smaller batch (8 vs 32) needs 1250 steps for a full epoch.",
    image_width=Inches(9.0),
)

# 29. Findings
text_only(
    "What we learned",
    [
        "ZeRO-2 wins for LoRA on slow interconnects (1.28× DDP). ZeRO-1 is close behind at 1.15×.",
        "Pipeline parallelism made the model fit but not faster — Gloo cross-node transfers cost more than DDP's single AllReduce.",
        "ZeRO-3 and FSDP fall apart over 1 Gbps Ethernet — per-layer AllGather of bf16 weights is intractable.",
        ("Implementation matters: DeepSpeed ZeRO-3 is ~1.7× faster than PyTorch FSDP, thanks to better overlap.", 1),
        "Tensor Parallel is the only model-parallel method actually usable here — 3× slower than DDP, but ships ~7× less data than ZeRO-3.",
        "The heavy methods (ZeRO-3, FSDP, TP) all need NVLink or InfiniBand. Wrong tool for Ethernet.",
        "A 7B model with LoRA fits comfortably on one 32 GiB card — pick a data-parallel variant.",
    ],
    base_size=22,
)

# 30. Thanks
title_slide(
    "Thank you",
    "Questions?",
    "github.com/rassulmagauin/ml710 · ZERO2_RESULTS.md · results_explanation.md",
)


prs.save(OUT)
print(f"Wrote {OUT}")
print(f"Slides: {len(prs.slides)}")
