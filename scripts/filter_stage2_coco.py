"""
Filter llava_v1_5_mix665k.json to COCO-only samples and take a random subset.

Usage:
    python scripts/filter_stage2_coco.py --n 10000 --seed 42

The resulting file is used as --data_path for Stage 2 training experiments.
"""
import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="LLaVA/playground/data/llava_v1_5_mix665k.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="LLaVA/playground/data/llava_v1_5_coco_subset.json",
    )
    parser.add_argument("--n", type=int, default=10000, help="Subset size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    with open(args.input, "r") as f:
        data = json.load(f)
    print(f"  Total samples: {len(data):,}")

    # Breakdown by image source
    sources = {}
    for sample in data:
        if "image" not in sample:
            sources.setdefault("text_only", 0)
            sources["text_only"] += 1
        else:
            src = sample["image"].split("/")[0]
            sources.setdefault(src, 0)
            sources[src] += 1
    print("  Breakdown by source:")
    for src, n in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {src:20s} {n:>8,}")

    # Keep only COCO samples
    coco_samples = [
        s for s in data if "image" in s and s["image"].startswith("coco/")
    ]
    print(f"\n  COCO samples available: {len(coco_samples):,}")

    if len(coco_samples) < args.n:
        print(
            f"  WARNING: requested {args.n:,} but only {len(coco_samples):,} available"
        )
        subset = coco_samples
    else:
        random.seed(args.seed)
        subset = random.sample(coco_samples, args.n)

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"\nWrote {len(subset):,} samples to {out_path}")


if __name__ == "__main__":
    main()
