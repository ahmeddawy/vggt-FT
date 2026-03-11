#!/bin/bash
# Run demo_colmap.py on all 20 training sequences
# Run from the vggt/ directory on GCP pod: bash /workspace/vggt-FT/run_eval_all_train.sh

set -u -o pipefail

DATASET_DIR="/mnt/bucket/dawy/vggt_finetune/dataset"
OUTPUT_BASE="/mnt/bucket/dawy/vggt_finetune/eval_outputs/50_epoch"
SEED=42
CONF_THRES=1.0
CKPT_PATH="${CKPT_PATH:-/mnt/bucket/dawy/vggt_finetune/checkpoint_5.pt}"
USE_MASKS=1
MASKS_SUBDIR="${MASKS_SUBDIR:-masks}"
MASK_BACKGROUND_VALUE=0
MASK_CONF_SUPPRESSED_VALUE=1.0

SEQUENCES=(
    "crqol8dkbpscqjzweb88cucu_122"
    "00444043_ES02"
    "cn3lv4iwdv2w2ewrjfowziuu_189"
    "cn9lxvc7w4afxhfqa59d1fpu_7"
    "00450780_es03"
    "c7pqzv9qa3i28esbn2irxwku__335"
    "Interstellar-seg69"
    "cj7bhkk96vh8lwpof9fbolnu_2"
    "cu54odpsje5sf2nlkc4g7l4u_0"
    "00475148_es02"
    "c6jsmt29z06oa6qcxzg6ii6u"
    "Amradib2_shot_008"
    "cvwads5a0g1y3ni0bfpv0m9u_60"
    "cn9lxvc7w4afxhfqa59d1fpu_176"
    "cvwads5a0g1y3ni0bfpv0m9u_619"
    "ca5p2o16ynhj9lq3yshjpjiu"
    "cvwads5a0g1y3ni0bfpv0m9u_59"
    "cvmubhidkjtii1mnk63xbnnu_1"
    "00444043_ES04"
    )

TOTAL=${#SEQUENCES[@]}
FAILED=()
echo "Running demo_colmap.py on $TOTAL training sequences"
echo "Output base: $OUTPUT_BASE"
echo "Checkpoint: $CKPT_PATH"
echo "Masks subdir: $MASKS_SUBDIR"
echo "Use masks: $USE_MASKS (background=${MASK_BACKGROUND_VALUE}, conf_suppressed=${MASK_CONF_SUPPRESSED_VALUE})"
echo "---"

for i in "${!SEQUENCES[@]}"; do
    SEQ="${SEQUENCES[$i]}"
    NUM=$((i + 1))
    echo "[$NUM/$TOTAL] Processing: $SEQ"

    SCENE_DIR="$DATASET_DIR/$SEQ"
    MASKS_DIR=""
    if [[ -d "$SCENE_DIR/$MASKS_SUBDIR" ]]; then
        MASKS_DIR="$SCENE_DIR/$MASKS_SUBDIR"
    elif [[ -d "$SCENE_DIR/masks" ]]; then
        MASKS_DIR="$SCENE_DIR/masks"
    elif [[ -d "$SCENE_DIR/mask" ]]; then
        MASKS_DIR="$SCENE_DIR/mask"
    fi
    OUT_DIR="$OUTPUT_BASE/$SEQ/sparse"
    mkdir -p "$OUT_DIR"

    CMD=(
        python demo_colmap.py
        --scene_dir "$SCENE_DIR"
        --output_dir "$OUT_DIR"
        --seed "$SEED"
        --checkpoint "$CKPT_PATH"
        --conf_thres_value "$CONF_THRES"
    )

    if [[ "$USE_MASKS" == "1" ]]; then
        if [[ -n "$MASKS_DIR" ]]; then
            CMD+=(
                --masks_dir "$MASKS_DIR"
                --mask_background_value "$MASK_BACKGROUND_VALUE"
                --mask_conf_suppressed_value "$MASK_CONF_SUPPRESSED_VALUE"
            )
            echo "[$NUM/$TOTAL] Masks: enabled ($MASKS_DIR)"
        else
            echo "[$NUM/$TOTAL] Masks: not found for $SEQ, running without masks."
        fi
    fi

    if "${CMD[@]}"; then
        echo "[$NUM/$TOTAL] Done: $SEQ"
    else
        echo "[$NUM/$TOTAL] FAILED: $SEQ"
        FAILED+=("$SEQ")
    fi
    echo "---"
done

echo "All $TOTAL sequences processed."
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED sequences (${#FAILED[@]}):"
    for s in "${FAILED[@]}"; do echo "  - $s"; done
else
    echo "All sequences succeeded."
fi
echo "Outputs saved to: $OUTPUT_BASE"
