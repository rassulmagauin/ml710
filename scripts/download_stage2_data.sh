#!/bin/bash
# =============================================================================
# Download Stage 2 instruction-tuning data for LLaVA parallelism benchmarks
#
# Downloads:
#   - LLaVA-Instruct-150K annotations (~1.1 GB JSON)
#   - COCO 2014 train images (~13 GB zip)
# Then creates a 10K subset for fast experiments (< 1 hour per run).
#
# Run from the project root:
#   bash scripts/download_stage2_data.sh
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_DIR/LLaVA/playground/data"
export DATA_DIR

echo "=== Stage 2 data setup ==="
echo "Project root : $PROJECT_DIR"
echo "Data dir     : $DATA_DIR"
echo ""

# ── 1. LLaVA-Instruct-150K annotation JSON ────────────────────────────────────
ANNOT_URL="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json"
ANNOT_PATH="$DATA_DIR/llava_instruct_150k.json"

if [ -f "$ANNOT_PATH" ]; then
    echo "[skip] $ANNOT_PATH already exists"
else
    echo "[download] LLaVA-Instruct-150K annotations..."
    wget -c "$ANNOT_URL" -O "$ANNOT_PATH"
    echo "[done] annotations saved to $ANNOT_PATH"
fi

# ── 2. COCO 2014 train images ──────────────────────────────────────────────────
# llava_instruct_150k.json image fields are bare filenames (e.g. COCO_train2014_*.jpg)
# that live in the train2014/ directory.
COCO_DIR="$DATA_DIR/coco"
COCO_ZIP="$COCO_DIR/train2014.zip"
COCO_IMG_DIR="$COCO_DIR/train2014"
COCO_URL="http://images.cocodataset.org/zips/train2014.zip"

mkdir -p "$COCO_DIR"

if [ -d "$COCO_IMG_DIR" ] && [ "$(ls -A "$COCO_IMG_DIR" 2>/dev/null | wc -l)" -gt 10000 ]; then
    echo "[skip] COCO train2014 images already present in $COCO_IMG_DIR"
else
    echo "[download] COCO 2014 train images (~13 GB, this will take a while)..."
    wget -c "$COCO_URL" -O "$COCO_ZIP"
    echo "[unzip] Extracting images..."
    unzip -q "$COCO_ZIP" -d "$COCO_DIR"
    echo "[done] Images in $COCO_IMG_DIR"
    # Keep the zip in case re-extraction is needed; remove to save space:
    # rm "$COCO_ZIP"
fi

# ── 3. Create 10K subset ───────────────────────────────────────────────────────
SUBSET_PATH="$DATA_DIR/llava_instruct_10k.json"

if [ -f "$SUBSET_PATH" ]; then
    echo "[skip] $SUBSET_PATH already exists"
else
    echo "[subset] Sampling 10K examples from 150K..."
    python "$PROJECT_DIR/scripts/create_subset.py" \
        --input  "$ANNOT_PATH" \
        --output "$SUBSET_PATH" \
        --n 10000 \
        --seed 42
    echo "[done] Subset saved to $SUBSET_PATH"
fi

# ── 3b. Normalize COCO filenames in-place ─────────────────────────────────────
echo "[fix] Normalizing COCO image filenames in $SUBSET_PATH ..."
python - <<'EOF'
import json
import os

subset_path = os.path.join(os.environ["DATA_DIR"], "llava_instruct_10k.json")

with open(subset_path, "r") as f:
    data = json.load(f)

updated = 0
for ex in data:
    img = ex.get("image")
    if img and not img.startswith("COCO_train2014_"):
        ex["image"] = f"COCO_train2014_{img}"
        updated += 1

with open(subset_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Updated {updated} image paths")
print(f"First image : {data[0].get('image', '<none>')}")
EOF

# ── 4. Quick sanity check ──────────────────────────────────────────────────────
echo ""
echo "=== Sanity check ==="
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate llava 2>/dev/null || true

python - <<'EOF'
import json, os, sys

base = os.environ.get("DATA_DIR", "LLaVA/playground/data")
subset_path = os.path.join(base, "llava_instruct_10k.json")
img_dir     = os.path.join(base, "coco/train2014")

if not os.path.exists(subset_path):
    print(f"ERROR: subset not found at {subset_path}")
    sys.exit(1)

data = json.load(open(subset_path))
print(f"Subset size  : {len(data)} samples")
print(f"Sample keys  : {list(data[0].keys())}")
print(f"Sample image : {data[0].get('image', '<none>')}")

missing = [
    d["image"] for d in data
    if "image" in d and not os.path.exists(os.path.join(img_dir, d["image"]))
]
if missing:
    print(f"WARNING: {len(missing)}/{len(data)} images not found under {img_dir}")
    print(f"  First missing: {missing[0]}")
else:
    print(f"Images check : all {len(data)} images found under {img_dir}")
EOF

echo ""
echo "=== Setup complete ==="
echo "Annotation  : $ANNOT_PATH"
echo "Subset      : $SUBSET_PATH"
echo "Images      : $COCO_IMG_DIR"
echo ""
echo "Next step: sbatch scripts/run_stage2_1gpu.sh"
