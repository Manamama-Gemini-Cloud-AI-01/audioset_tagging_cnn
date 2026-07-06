#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# audio_me.sh - Simplified launcher for Sound Event Detection
# Uses the 'Cautious Listener' (DecisionLevelMax) model as the daily driver.
# ------------------------------------------------------------------------------

MODEL_TYPE="Cnn14_DecisionLevelMax"

# Path to the preferred model checkpoint
CHECKPOINT_PATH="/media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    CHECKPOINT_PATH="/data/data/com.termux/files/home/storage/external_SD/LLMs/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth"
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    CHECKPOINT_PATH="/storage/emulated/0/LLMs/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth"
fi
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    CHECKPOINT_PATH="$HOME/Downloads/GitHub/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth"
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "--- AudioSet Tagging CNN Inference (SED Mode) ---"
echo "Model Type: $MODEL_TYPE"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Input file and parameters: $@"
echo -e "Add  \033[1;34m--dynamic_eventogram \033[1;0m  or  \033[1;34m--static_eventogram\033[1;0m  for a video of the graph with a moving marker."
echo 

echo "Note: Paths provided as command-line arguments should be absolute to avoid ambiguity."

cd "$HOME/Downloads/GitHub/audioset_tagging_cnn/" || {
    echo "ERROR: Cannot cd to project directory"
    exit 1
}

echo "We are checking the real path of the first argument..."
INPUT_FILE=$(realpath "$1")

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: The input file has not been provided or has not been found at '$INPUT_FILE' "
    exit 1
fi

# ====================== TEMP DIRECTORY SANITY CHECK ======================
TMPDIR="${TMPDIR:-${TMP:-/tmp}}"


#We remove the lot silently: 

rm -f "$TMPDIR"/temp_cbr_* "$TMPDIR"/temp_cfr_*


if [[ -d "$TMPDIR" ]]; then
    OUR_TEMPS=$(find "$TMPDIR" -name 'temp_cbr_*' -o -name 'temp_cfr_*' 2>/dev/null)
    
    if [[ -n "$OUR_TEMPS" ]]; then
        echo -e "\033[1;33m⚠️  WARNING: Old temporary files from previous runs detected\033[0m"
        echo "in $TMPDIR"
        echo
        echo "$OUR_TEMPS"
        echo
        echo "These can cause 'Permission denied' if owned by root (from sudo runs)."
        echo
        echo "Recommended fix:"
        echo "    sudo rm -f \"$TMPDIR\"/temp_cbr_* \"$TMPDIR\"/temp_cfr_*"
        echo "    # or"
        echo "    sudo chown -R \$(whoami) \"$TMPDIR\""
        echo
        read -r -p "Press Enter to continue anyway or Ctrl+C to abort... "
    fi
fi
# ====================== END TEMP CHECK ======================

# ====================== ARGUMENT FILTERING & PATH STABILIZATION ======================
PY_ARGS=()
SHAPASH=false

# 1. Stabilize the input path: Seed the Python arguments with the absolute resolved path
PY_ARGS+=("$INPUT_FILE")

# 2. Filter subsequent arguments to trap custom launcher flags
for arg in "${@:2}"; do
    if [[ "$arg" == "--shapash" ]]; then
        SHAPASH=true
    else
        PY_ARGS+=("$arg")
    fi
done
# =====================================================================================

# Run inference with our pristine, filtered parameters
$PREFIX/bin/time -v python  "$HOME/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py" \
    --model_type="$MODEL_TYPE" \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --cuda \
    "${PY_ARGS[@]}"    

# --- STREAMLINING: Auto-launch Shapash Dashboard ---
INPUT_FILENAME=$(basename "$INPUT_FILE")
INPUT_FILENAME="${INPUT_FILENAME%.*}"          # remove extension

INPUT_DIR=$(dirname "$INPUT_FILE")
CHECKPOINT_FILENAME=$(basename "$CHECKPOINT_PATH")

OUTPUT_DIR="${INPUT_DIR}/${INPUT_FILENAME}_${CHECKPOINT_FILENAME}_audioset_tagging_cnn"
H5_PATH="${OUTPUT_DIR}/full_event_log.h5"

if [[ "$SHAPASH" == true ]]; then
    echo
    echo "📊  Launching Unified Shapash Dashboard..."
    echo "Note: This dashboard explains the Top 50 sounds detected."
    python "$HOME/Downloads/GitHub/audioset_tagging_cnn/scripts/Shapash_visualization/launch_multi_target_dashboard.py" "$H5_PATH"
fi

echo


