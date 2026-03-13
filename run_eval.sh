#!/usr/bin/env bash
# run_eval.sh — VGGT inference + evaluation on all dataset sequences.
#
# Usage:
#   bash run_eval.sh \
#     --vanilla-ckpt   model.pt \
#     --finetuned-ckpt checkpoints/model_finetuned.pt \
#     [--dataset-dir   dataset/]          \
#     [--output-dir    eval_outputs/]     \
#     [--skip-inference]                  \
#     [--vanilla-tag   vanilla]           \
#     [--finetuned-tag finetuned]
#
# Outputs per model:
#   <output-dir>/<tag>/<seq>/sparse/  — COLMAP model + depths.npy
#
# Final eval results:
#   <output-dir>/camera_eval_train.json, camera_eval_test.json, camera_eval_all.json
#   <output-dir>/depth_eval_train.json,  depth_eval_test.json,  depth_eval_all.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/../vggt/eval" && pwd)"

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
VANILLA_CKPT="${SCRIPT_DIR}/model.pt"
FINETUNED_CKPT=""
DATASET_DIR="${SCRIPT_DIR}/dataset"
OUTPUT_DIR="${SCRIPT_DIR}/eval_outputs"
VANILLA_TAG="vanilla"
FINETUNED_TAG="finetuned"
SKIP_INFERENCE=0

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vanilla-ckpt)   VANILLA_CKPT="$2";   shift 2 ;;
        --finetuned-ckpt) FINETUNED_CKPT="$2"; shift 2 ;;
        --dataset-dir)    DATASET_DIR="$2";    shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2";     shift 2 ;;
        --vanilla-tag)    VANILLA_TAG="$2";    shift 2 ;;
        --finetuned-tag)  FINETUNED_TAG="$2";  shift 2 ;;
        --skip-inference) SKIP_INFERENCE=1;    shift   ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$FINETUNED_CKPT" ]]; then
    echo "[ERROR] --finetuned-ckpt is required."
    exit 1
fi

if [[ ! -f "$VANILLA_CKPT" ]]; then
    echo "[ERROR] Vanilla checkpoint not found: $VANILLA_CKPT"
    exit 1
fi

if [[ ! -f "$FINETUNED_CKPT" ]]; then
    echo "[ERROR] Finetuned checkpoint not found: $FINETUNED_CKPT"
    exit 1
fi

SPLITS_DIR="${DATASET_DIR}/splits"
TRAIN_SPLIT="${SPLITS_DIR}/train.txt"
TEST_SPLIT="${SPLITS_DIR}/test.txt"

echo "============================================================"
echo " VGGT Eval Runner"
echo "============================================================"
echo "  vanilla  ckpt : $VANILLA_CKPT"
echo "  finetuned ckpt: $FINETUNED_CKPT"
echo "  dataset dir   : $DATASET_DIR"
echo "  output dir    : $OUTPUT_DIR"
echo "  vanilla tag   : $VANILLA_TAG"
echo "  finetuned tag : $FINETUNED_TAG"
echo "  skip inference: $SKIP_INFERENCE"
echo "============================================================"

# --------------------------------------------------------------------------
# Discover all sequences (any subdir of dataset/ that has an images/ folder)
# --------------------------------------------------------------------------
SEQUENCES=()
while IFS= read -r -d '' seq_dir; do
    seq=$(basename "$seq_dir")
    [[ "$seq" == "splits" || "$seq" == "reports" ]] && continue
    SEQUENCES+=("$seq")
done < <(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

echo "Found ${#SEQUENCES[@]} sequences."

# --------------------------------------------------------------------------
# Helper: run inference for one (sequence, checkpoint, tag)
# --------------------------------------------------------------------------
run_inference() {
    local seq="$1"
    local ckpt="$2"
    local tag="$3"

    local scene_dir="${DATASET_DIR}/${seq}"
    local out_dir="${OUTPUT_DIR}/${tag}/${seq}/sparse"
    local depths_path="${out_dir}/depths.npy"

    if [[ -f "$depths_path" ]]; then
        echo "  [SKIP] ${tag}/${seq} — depths.npy already exists."
        return 0
    fi

    echo "  [RUN]  ${tag}/${seq} ..."
    mkdir -p "$out_dir"
    python "${SCRIPT_DIR}/demo_colmap.py" \
        --scene_dir  "$scene_dir" \
        --output_dir "$out_dir"  \
        --checkpoint "$ckpt"     \
        --save_depth             \
        2>&1 | tail -5
    echo "  [DONE] ${tag}/${seq}"
}

# --------------------------------------------------------------------------
# Inference loop
# --------------------------------------------------------------------------
if [[ "$SKIP_INFERENCE" -eq 0 ]]; then
    echo ""
    echo "--- Inference: ${VANILLA_TAG} ---"
    for seq in "${SEQUENCES[@]}"; do
        run_inference "$seq" "$VANILLA_CKPT" "$VANILLA_TAG"
    done

    echo ""
    echo "--- Inference: ${FINETUNED_TAG} ---"
    for seq in "${SEQUENCES[@]}"; do
        run_inference "$seq" "$FINETUNED_CKPT" "$FINETUNED_TAG"
    done
else
    echo "[INFO] Skipping inference (--skip-inference set)."
fi

VANILLA_ROOT="${OUTPUT_DIR}/${VANILLA_TAG}"
FINETUNED_ROOT="${OUTPUT_DIR}/${FINETUNED_TAG}"

# --------------------------------------------------------------------------
# Helper: generate a combined all.txt from train + test splits
# --------------------------------------------------------------------------
ALL_SPLIT="${OUTPUT_DIR}/all_sequences.txt"
if [[ -f "$TRAIN_SPLIT" && -f "$TEST_SPLIT" ]]; then
    cat "$TRAIN_SPLIT" "$TEST_SPLIT" | sort -u > "$ALL_SPLIT"
    echo ""
    echo "[INFO] Combined split: $ALL_SPLIT ($(wc -l < "$ALL_SPLIT") sequences)"
fi

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
echo ""
echo "--- Evaluation ---"

run_camera_eval() {
    local split_arg="$1"
    local split_label="$2"
    local json_out="${OUTPUT_DIR}/camera_eval_${split_label}.json"

    echo ""
    echo "[camera_eval] split=${split_label}"
    python "${EVAL_DIR}/camera_eval.py" \
        --gt-root        "$DATASET_DIR"    \
        --vanilla-root   "$VANILLA_ROOT"   \
        --finetuned-root "$FINETUNED_ROOT" \
        --gt-subdir      colmap            \
        --pred-subdir    sparse            \
        ${split_arg:+--split-file "$split_arg"} \
        --output-json    "$json_out"
    echo "[camera_eval] saved: $json_out"
}

run_depth_eval() {
    local split_arg="$1"
    local split_label="$2"
    local json_out="${OUTPUT_DIR}/depth_eval_${split_label}.json"

    echo ""
    echo "[depth_eval] split=${split_label}"
    python "${EVAL_DIR}/depth_eval.py" \
        --gt-root        "$DATASET_DIR"    \
        --vanilla-root   "$VANILLA_ROOT"   \
        --finetuned-root "$FINETUNED_ROOT" \
        --gt-subdir      colmap            \
        --gt-depth-dir   depth             \
        --pred-subdir    sparse            \
        --pred-depth-file depths.npy       \
        ${split_arg:+--split-file "$split_arg"} \
        --output-json    "$json_out"
    echo "[depth_eval] saved: $json_out"
}

# Train split
if [[ -f "$TRAIN_SPLIT" ]]; then
    run_camera_eval "$TRAIN_SPLIT" "train"
    run_depth_eval  "$TRAIN_SPLIT" "train"
fi

# Test split
if [[ -f "$TEST_SPLIT" ]]; then
    run_camera_eval "$TEST_SPLIT" "test"
    run_depth_eval  "$TEST_SPLIT" "test"
fi

# All sequences
if [[ -f "$ALL_SPLIT" ]]; then
    run_camera_eval "$ALL_SPLIT" "all"
    run_depth_eval  "$ALL_SPLIT" "all"
fi

echo ""
echo "============================================================"
echo " Done. Results in: $OUTPUT_DIR"
echo "  camera_eval_train.json, camera_eval_test.json, camera_eval_all.json"
echo "  depth_eval_train.json,  depth_eval_test.json,  depth_eval_all.json"
echo "============================================================"
