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
SAVE_CONF_MASK_REPORT="${SAVE_CONF_MASK_REPORT:-1}"
RUN_FORCED_MASK_CONF="${RUN_FORCED_MASK_CONF:-1}"
RUN_LEARNED_CONF_ONLY="${RUN_LEARNED_CONF_ONLY:-1}"
FORCED_EXP_NAME="${FORCED_EXP_NAME:-forced_mask_conf}"
LEARNED_EXP_NAME="${LEARNED_EXP_NAME:-learned_conf_only}"

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
EXP_COUNT=0
if [[ "$RUN_FORCED_MASK_CONF" == "1" ]]; then
    EXP_COUNT=$((EXP_COUNT + 1))
fi
if [[ "$RUN_LEARNED_CONF_ONLY" == "1" ]]; then
    EXP_COUNT=$((EXP_COUNT + 1))
fi
if [[ "$EXP_COUNT" -eq 0 ]]; then
    echo "Nothing to run: set RUN_FORCED_MASK_CONF=1 and/or RUN_LEARNED_CONF_ONLY=1"
    exit 1
fi
TOTAL_JOBS=$((TOTAL * EXP_COUNT))
JOB_IDX=0

echo "Running demo_colmap.py on $TOTAL training sequences"
echo "Total jobs: $TOTAL_JOBS ($EXP_COUNT experiments per sequence)"
echo "Output base: $OUTPUT_BASE"
echo "Checkpoint: $CKPT_PATH"
echo "Masks subdir: $MASKS_SUBDIR"
echo "Use masks: $USE_MASKS (background=${MASK_BACKGROUND_VALUE}, conf_suppressed=${MASK_CONF_SUPPRESSED_VALUE})"
echo "Experiments: forced=$RUN_FORCED_MASK_CONF (${FORCED_EXP_NAME}), learned_only=$RUN_LEARNED_CONF_ONLY (${LEARNED_EXP_NAME})"
echo "Save conf report: $SAVE_CONF_MASK_REPORT"
echo "---"

run_experiment() {
    local exp_name="$1"
    local disable_mask_suppression="$2"  # 0 or 1
    local out_dir="$OUTPUT_BASE/$exp_name/$SEQ/sparse"

    JOB_IDX=$((JOB_IDX + 1))
    mkdir -p "$out_dir"

    local -a cmd=(
        python demo_colmap.py
        --scene_dir "$SCENE_DIR"
        --output_dir "$out_dir"
        --seed "$SEED"
        --checkpoint "$CKPT_PATH"
        --conf_thres_value "$CONF_THRES"
    )

    if [[ "$SAVE_CONF_MASK_REPORT" == "1" ]]; then
        cmd+=(--save_conf_mask_report)
    fi

    if [[ "$USE_MASKS" == "1" ]]; then
        if [[ -n "$MASKS_DIR" ]]; then
            cmd+=(
                --masks_dir "$MASKS_DIR"
                --mask_background_value "$MASK_BACKGROUND_VALUE"
                --mask_conf_suppressed_value "$MASK_CONF_SUPPRESSED_VALUE"
            )
        else
            echo "[$NUM/$TOTAL][$exp_name] Masks: not found for $SEQ, running without masks."
        fi
    fi

    if [[ "$disable_mask_suppression" == "1" ]]; then
        cmd+=(--disable_mask_conf_suppression)
    fi

    echo "[job $JOB_IDX/$TOTAL_JOBS][$NUM/$TOTAL][$exp_name] OUT: $out_dir"
    if "${cmd[@]}"; then
        echo "[job $JOB_IDX/$TOTAL_JOBS][$NUM/$TOTAL][$exp_name] Done: $SEQ"
    else
        echo "[job $JOB_IDX/$TOTAL_JOBS][$NUM/$TOTAL][$exp_name] FAILED: $SEQ"
        FAILED+=("${SEQ}:${exp_name}")
    fi
}

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

    if [[ "$USE_MASKS" == "1" ]]; then
        if [[ -n "$MASKS_DIR" ]]; then
            echo "[$NUM/$TOTAL] Masks: enabled ($MASKS_DIR)"
        else
            echo "[$NUM/$TOTAL] Masks: not found for $SEQ, running without masks."
        fi
    fi

    if [[ "$RUN_FORCED_MASK_CONF" == "1" ]]; then
        run_experiment "$FORCED_EXP_NAME" 0
    fi

    if [[ "$RUN_LEARNED_CONF_ONLY" == "1" ]]; then
        run_experiment "$LEARNED_EXP_NAME" 1
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
