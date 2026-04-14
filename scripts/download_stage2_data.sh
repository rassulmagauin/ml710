#!/bin/bash
# =============================================================================
# Download LLaVA Stage 2 data (COCO-only, slim version)
# =============================================================================
# For parallelism experiments we don't need all 5 image sources — COCO is
# enough for a representative subset. Saves ~60GB of download.
#
# Run inside a tmux session on a worker node:
#   tmux new -s download
#   salloc -p ws-ia -N1 -n12 -w ws-l4-XXX --mem=32G --time=04:00:00
#   bash scripts/download_stage2_data.sh
#
# Total size: ~20GB. Takes ~20-40 min on the HPC network.
# After download, run scripts/filter_stage2_coco.py to filter + subset the JSON.
# =============================================================================

set -e

DATA_DIR=/home/rassul.magauin/ml710/LLaVA/playground/data
mkdir -p $DATA_DIR
cd $DATA_DIR

echo "=== Target: $DATA_DIR ==="
df -h $DATA_DIR

# -----------------------------------------------------------------------------
# 1. Instruction JSON (llava_v1_5_mix665k.json) — ~1GB
# -----------------------------------------------------------------------------
if [ ! -f llava_v1_5_mix665k.json ]; then
    echo ""
    echo "=== [1/2] Downloading llava_v1_5_mix665k.json ==="
    wget -c https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json
else
    echo "[1/2] llava_v1_5_mix665k.json already exists, skipping"
fi

# -----------------------------------------------------------------------------
# 2. COCO train2017 (~19GB)
# -----------------------------------------------------------------------------
mkdir -p coco && cd coco
if [ ! -d train2017 ]; then
    echo ""
    echo "=== [2/2] Downloading COCO train2017 (~19GB) ==="
    wget -c http://images.cocodataset.org/zips/train2017.zip
    echo "Extracting COCO train2017..."
    unzip -q train2017.zip
    rm train2017.zip
else
    echo "[2/2] coco/train2017 already exists, skipping"
fi
cd $DATA_DIR

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=== Download summary ==="
du -sh llava_v1_5_mix665k.json 2>/dev/null || echo "MISSING: llava_v1_5_mix665k.json"
du -sh coco/train2017 2>/dev/null          || echo "MISSING: coco/train2017"
echo ""
echo "Next: filter the JSON to COCO-only samples:"
echo "  python scripts/filter_stage2_coco.py --n 10000 --seed 42"
echo ""
echo "=== All done ==="
