#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# audio_me.sh - Simplified launcher for Sound Event Detection
# Uses the 'Cautious Listener' (DecisionLevelMax) model as the daily driver.
# ------------------------------------------------------------------------------


MODEL_TYPE="Cnn14_DecisionLevelMax"

# Path to the preferred model checkpoint
CHECKPOINT_PATH="/media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth"


if [[ ! -f "$CHECKPOINT_PATH" ]]; then
#Maybe we are in Termux? 
CHECKPOINT_PATH="/data/data/com.termux/files/home/storage/external_SD/LLMs/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth"

CHECKPOINT_PATH="/storage/emulated/0/LLMs/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth"

fi


if [[ ! -f "$CHECKPOINT_PATH" ]]; then
#Maybe local? 

CHECKPOINT_PATH="$HOME/Downloads/GitHub/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth"


fi



if [[ ! -f "$CHECKPOINT_PATH" ]]; then
#We give up
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "--- AudioSet Tagging CNN Inference (SED Mode) ---"
echo "Model Type: $MODEL_TYPE"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Input file and parameters: $@"
echo -e "Add  \033[1;34m--dynamic_eventogram \033[1;0m  or  \033[1;34m--static_eventogram\033[1;0m  for a video of the graph with a moving marker."
echo 

echo "Note: If you see errors with sox / mp3 parsing, then apt install 'sox' and 'libsox-fmt-mp3', test it via 'sox --info "$@" ' "
echo "Note: Paths provided as command-line arguments should be absolute to avoid ambiguity."

#Obsolete by now: 
#echo "Note: if no torchaudio backends, do find a patch file, usually called speechbrain.patch, and patch speechbrain itself. " 

cd $HOME/Downloads/GitHub/audioset_tagging_cnn/



echo "We are checking the real path of the first argument, which may hang if the file be not accessible..."
INPUT_FILE=$(realpath "$1")



if [[ ! -f "$INPUT_FILE" ]]; then
#We give up
    echo "ERROR: The input file has not been provided or has not been found at '$INPUT_FILE' "
    exit 1
fi


echo 


# Run inference

$PREFIX/bin/time -v python  "$HOME/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py" \
    --model_type="$MODEL_TYPE" \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --cuda \
    "$@"    

# --- STREAMLINING: Auto-launch Shapash Dashboard ---

INPUT_FILENAME=$(basename "$INPUT_FILE" | sed 's/\.[^.]*$//')
CHECKPOINT_FILENAME=$(basename "$CHECKPOINT_PATH")
INPUT_DIR=$(dirname "$INPUT_FILE")

# Reconstruct the output directory path (must match Python logic)
OUTPUT_DIR="${INPUT_DIR}/${INPUT_FILENAME}_${CHECKPOINT_FILENAME}_audioset_tagging_cnn"
H5_PATH="${OUTPUT_DIR}/full_event_log.h5"

# Check if --no-shapash is in the arguments
SKIP_SHAPASH=false
for arg in "$@"; do
    if [[ "$arg" == "--no-shapash" ]]; then
        SKIP_SHAPASH=true
        break
    fi
done

if [[ "$SKIP_SHAPASH" == true ]]; then
    echo
    echo "⏭️  Skipping Shapash dashboard launch (--no-shapash specified)."
elif [[ -f "$H5_PATH" ]]; then
    echo
    echo "📊  Launching Unified Shapash Dashboard..."
    echo "Note: This dashboard explains the Top 50 sounds detected."
    echo "Use the GUI dropdown to switch between sound targets."
    # Speed Hack: Uses 'Tiny Forest' (10 estimators) and 'Strategic Sampling' (200 points) 
    # to maintain sub-10s 'Brain' generation on mobile devices.
    python "$HOME/Downloads/GitHub/audioset_tagging_cnn/scripts/Shapash_visualization/launch_multi_target_dashboard.py" "$H5_PATH"
else
    echo
    echo "⚠️  Warning: full_event_log.h5 not found at $H5_PATH. Skipping dashboard."
fi


echo
