#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# audio_me.sh - Simplified launcher for Sound Event Detection
# Uses the 'Cautious Listener' (DecisionLevelMax) model as the daily driver.
# ------------------------------------------------------------------------------

# Path to the preferred model checkpoint
CHECKPOINT_PATH="/media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth"
MODEL_TYPE="Cnn14_DecisionLevelMax"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "--- AudioSet Tagging CNN Inference (SED Mode) ---"
echo "Model Type: $MODEL_TYPE"
echo "Input:      $@"

echo "Note: If you see errors with sox / mp3 parsing → install sox + libsox-fmt-mp3, test it via 'sox --info "$@" ' "

# Run inference with dynamic eventogram and auto-sanitization enabled
time python3 pytorch/audioset_tagging_cnn_inference_6.py \
    "$@" \
    --model_type="$MODEL_TYPE" \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --dynamic_eventogram \
    --cuda
