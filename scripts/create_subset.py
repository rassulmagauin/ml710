"""
Create a smaller subset of LLaVA training data for quick parallelism experiments.
We sample N examples from the full dataset so each training run completes in <1 hour.
"""
import json
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to full JSON data file")
    parser.add_argument("--output", type=str, required=True, help="Path to output subset JSON")
    parser.add_argument("--n", type=int, default=5000, help="Number of samples to keep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    print(f"Full dataset: {len(data)} samples")
    random.seed(args.seed)
    subset = random.sample(data, min(args.n, len(data)))
    print(f"Subset: {len(subset)} samples")

    with open(args.output, "w") as f:
        json.dump(subset, f, indent=2)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
