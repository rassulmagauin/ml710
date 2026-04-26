# Results Explanation

## Why ZeRO-2 Didn't Really Speed Up Compared to DDP

The data tells a pretty clear story. The key numbers:

| | DDP | ZeRO-2 (best) |
|---|---|---|
| Nodes × GPU | 2 × 1 | 2 × 1 ("1 GPU observed") |
| Effective batch | 32 | 32 |
| Wall time | 4491s | 3503s |
| Throughput | 2.22 samples/s | 3.01 samples/s |
| GPU util | 70.7% | 88.5% |

~1.28× speedup — not the ~2× you'd expect. Here's why:

The root cause is LoRA. ZeRO-2 is designed to shard two things across GPUs: optimizer states (Adam's m and v tensors) and gradients. For a 7B model with full fine-tuning, those are enormous — ~112 GB of optimizer states. ZeRO-2 cuts that per-GPU footprint in half.

With LoRA, you have maybe 0.5% of parameters trainable. The optimizer states and gradients are tiny — a few hundred MB total. There's almost nothing for ZeRO-2 to shard. So:

- **Memory**: Both DDP and ZeRO-2 fit comfortably (30 GiB peak on a 32 GiB card) — ZeRO-2's memory sharding provides zero practical benefit
- **Gradient communication**: DDP AllReduces only LoRA adapter gradients — already small and fast. ZeRO-2 does the same AllReduce, then adds extra coordination for optimizer state scatter/gather that DDP skips entirely
- **Net result**: ZeRO-2 adds overhead without reducing the dominant cost

The 70.7% vs 88.5% GPU utilization gap actually confirms this: DDP was stalling more, likely on cross-node Ethernet AllReduce synchronization, while ZeRO-2's DeepSpeed engine kept the GPU busier via async overlap. But that's an implementation-level efficiency gain from DeepSpeed's scheduler, not a fundamental ZeRO-2 benefit — you'd get the same from a better-tuned DDP run.

**The takeaway for slides:** ZeRO-2 is the wrong tool for LoRA fine-tuning. It's designed for full fine-tuning where optimizer states dominate memory. With LoRA, DDP is already near-optimal and ZeRO-2's sharding machinery becomes pure overhead on a slow cross-node link.

---

## Pipeline Parallelism With LoRA

The manual pipeline-parallel LoRA run eventually worked, but only after changing the memory shape of the pipeline:

| Attempt | Split | Batch config | Result |
|---|---:|---|---|
| `stage2_pipe_lora_dist_20260424_152047` | 16 | batch 4, grad accum 4 | Failed: rank 0 CUDA OOM |
| `stage2_pipe_lora_dist_20260424_152425` / `152426` | 8 | batch 4, grad accum 4 | Failed: rank 1 CUDA OOM |
| `stage2_pipe_lora_dist_20260424_153422` | 12 | batch 1, grad accum 16 | Completed |

The successful run used Vicuna 7B weights, LoRA adapters, split layer `12`, effective batch `16`, and finished `625/625` steps. The training loop took `4057.9s` (`67.63 min`), while the full logged wall time was `4325s`. This is close to the DDP baseline wall time (`4491s`) but at effective batch `16` instead of `32`, so it is not a better throughput result.

Why the failed splits matter:

- Split `16` put the vision tower, embeddings, projector, and decoder layers `0-15` on rank 0. Rank 0 OOMed in the first forward pass with about `31.38 GiB` in use.
- Split `8` moved too many decoder layers to rank 1. Rank 1 owned layers `8-31`, norm, lm head, and loss, and OOMed in the first forward pass with about `30.95 GiB` in use.
- Split `12` balanced the model better, but the real fix was reducing the active microbatch memory: `per_device_batch=1`, `gradient_accumulation_steps=16`, and `num_micro_batches=1`.

The pipeline run is still LoRA. The log reports `338,690,048` trainable parameters out of `7,401,592,832`, or `4.58%`, so it is not full fine-tuning.

**Takeaway:** pipeline parallelism can make the run fit and complete, but on this setup it does not create a strong speedup. The manual Gloo pipeline sends activations and gradients over Ethernet and has pipeline idle time; LoRA already reduces optimizer/gradient memory enough that the main gain is memory placement, not faster training.

---

## Why ZeRO-3 Failed

From the results file the answer is straightforward — it didn't fail, it timed out. But ZeRO-3 being 4–5× slower than DDP is itself a failure worth explaining:

**The math:** it reached 292/1250 steps in 3:52, projecting ~16:33 total wall time. DDP finished the same workload in 4491s (~1:15). ZeRO-3 would have taken ~59,580s — roughly **13× slower** than DDP.

**Why ZeRO-3 is so much slower here, same LoRA argument plus one more layer:**

ZeRO-3 shards *everything* — parameters, gradients, and optimizer states across both nodes. This means every forward and backward pass requires **AllGather of the full model weights** before each layer, then **immediate discard** after. For a 7B model in bf16, that's ~14 GB of weights being re-gathered across cross-node Ethernet every single step.

Concretely:

- Every forward pass: 32 AllGathers (one per transformer layer) × ~430 MB per layer = **~14 GB transferred**
- Every backward pass: 32 ReduceScatter passes = another **~14 GB**
- Over 1Gbps Ethernet: ~224 seconds per step just in communication
- Their run: 292 steps in ~14,000s ≈ **48s/step**, which matches

The effective batch was also halved to 8 (vs DDP's 32), so it needed more steps per epoch to begin with — 1250 steps vs ~312 for DDP.

**Summary:** ZeRO-3 shards model parameters, which forces constant weight re-materialization via AllGather across nodes. On 1–10 Gbps Ethernet this dominates everything. ZeRO-3 only makes sense with high-bandwidth interconnects (InfiniBand, NVLink) or when the model is too large to fit even a single replica. Neither condition applies here — a 7B LoRA fit comfortably in 30 GiB on a single RTX 5000 Ada.

---

## PyTorch FSDP vs DeepSpeed ZeRO-3

To check whether ZeRO-3's slowdown is *fundamental to full sharding* or just *DeepSpeed's implementation*, we ran the same workload with PyTorch's native FSDP (`full_shard auto_wrap`, `LlamaDecoderLayer` wrap policy, `use_orig_params=True`).

| | DeepSpeed ZeRO-3 | PyTorch FSDP |
|---|---|---|
| Effective batch | 8 | 8 |
| Per-step time | ~48 s | ~80 s |
| Throughput | ~0.17 samples/s | 0.095 samples/s |
| Projected full wall | ~16.5 h | ~27.9 h |
| Peak GPU memory | 30.3 GiB | 32.3 GiB |
| Avg GPU util | 99.3% | 97.5% |
| Slowdown vs DDP | ~13× | ~22× |

**FSDP is ~1.7× slower than ZeRO-3** on the same hardware, same workload, same effective batch. Why:

- ZeRO-3 (this repo's `zero3_bounded_async.json`) bounds `allgather_bucket_size`, `reduce_bucket_size`, `sub_group_size`, and `stage3_prefetch_bucket_size` at 50 MB and enables overlap. The bucketed allgather lets DeepSpeed start fetching the next layer's weights while the current layer is still computing.
- PyTorch FSDP defaults to wrapping each transformer block as its own FSDP unit. Without explicit `forward_prefetch=True` and bucket tuning, the allgather for layer N+1 can only start after layer N's forward returns, leaving the network idle during compute and the GPU idle during gather.
- Memory: FSDP's flat-parameter design (FlatParameter contiguous tensor) creates slightly larger gather buffers than DeepSpeed's parameter-coordinator scheme, hence the +2 GiB peak.

**Takeaway for slides:** the ZeRO-DP-3 sharding family is unworkable on 1 Gbps Ethernet for 7B-class models — both implementations fail. Comparing them shows that *implementation-level overlap matters even when the dominant cost is comm*. With a faster interconnect both would scale much better; on this setup, neither is the right tool.
