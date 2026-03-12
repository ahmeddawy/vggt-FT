#!/bin/bash
# Run demo_colmap.py on all sequences in a split file.
#
# Usage (from vggt-FT/ directory):
#   CKPT_PATH=/path/to/checkpoint.pt \
#   OUTPUT_BASE=/path/to/output \
#   SPLIT_FILE=/path/to/split.txt \
#   bash run_eval_split.sh
#
# All variables have defaults and can be overridden via env.

set -eo pipefail

SPLIT_FILE="${SPLIT_FILE:-/home/oem/Rembrand/Foundation_MC_finetune/vggt-FT/dataset/splits/train.txt}"
DATASET_DIR="${DATASET_DIR:-/home/oem/Rembrand/Foundation_MC_finetune/vggt-FT/dataset}"
OUTPUT_BASE="${OUTPUT_BASE:-/home/oem/Rembrand/Foundation_MC_finetune/vggt/eval_exps/eval_output}"
CKPT_PATH="${CKPT_PATH:-/home/oem/Rembrand/Foundation_MC_finetune/vggt-FT/model.pt}"
SEED="${SEED:-42}"
CONF_THRES="${CONF_THRES:-1.0}"
USE_MASKS="${USE_MASKS:-0}"
MASK_BACKGROUND_VALUE="${MASK_BACKGROUND_VALUE:-0}"
MASK_CONF_SUPPRESSED_VALUE="${MASK_CONF_SUPPRESSED_VALUE:-1.0}"
SAVE_DEPTH="${SAVE_DEPTH:-1}"

echo "===== Eval Config ====="
echo "  SPLIT_FILE : $SPLIT_FILE"
echo "  DATASET_DIR: $DATASET_DIR"
echo "  OUTPUT_BASE: $OUTPUT_BASE"
echo "  CKPT_PATH  : $CKPT_PATH"
echo "  USE_MASKS  : $USE_MASKS"
echo "  SAVE_DEPTH : $SAVE_DEPTH"
echo "======================="

FAILED=()
TOTAL=0
OK=0

# Read split file line by line — handles spaces in sequence names
while IFS= read -r SEQ || [[ -n "$SEQ" ]]; do
    # Skip empty lines and comments
    [[ -z "$SEQ" || "$SEQ" == \#* ]] && continue

    TOTAL=$((TOTAL + 1))
    SCENE_DIR="$DATASET_DIR/$SEQ"
    OUT_DIR="$OUTPUT_BASE/$SEQ/sparse"

    echo "[$TOTAL] $SEQ"

    if [[ ! -d "$SCENE_DIR" ]]; then
        echo "  [SKIP] scene dir not found: $SCENE_DIR"
        FAILED+=("$SEQ (missing scene dir)")
        continue
    fi

    mkdir -p "$OUT_DIR"

    CMD=(
        python demo_colmap.py
        --scene_dir "$SCENE_DIR"
        --output_dir "$OUT_DIR"
        --checkpoint "$CKPT_PATH"
        --seed "$SEED"
        --conf_thres_value "$CONF_THRES"
    )

    if [[ "$SAVE_DEPTH" == "1" ]]; then
        CMD+=(--save_depth --save_depth_conf)
    fi

    if [[ "$USE_MASKS" == "1" ]]; then
        MASKS_DIR=""
        for d in "$SCENE_DIR/masks" "$SCENE_DIR/mask"; do
            [[ -d "$d" ]] && MASKS_DIR="$d" && break
        done
        if [[ -n "$MASKS_DIR" ]]; then
            CMD+=(
                --masks_dir "$MASKS_DIR"
                --mask_background_value "$MASK_BACKGROUND_VALUE"
                --mask_conf_suppressed_value "$MASK_CONF_SUPPRESSED_VALUE"
            )
        fi
    fi

    if "${CMD[@]}"; then
        echo "  [OK]"
        OK=$((OK + 1))
    else
        echo "  [FAILED]"
        FAILED+=("$SEQ")
    fi

done < "$SPLIT_FILE"

echo ""
echo "===== Done: $OK/$TOTAL succeeded ====="
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed (${#FAILED[@]}):"
    for s in "${FAILED[@]}"; do echo "  - $s"; done
fi
echo "Outputs: $OUTPUT_BASE"
