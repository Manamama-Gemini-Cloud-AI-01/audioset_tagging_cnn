
# Model Personalities:


# Cnn14_DecisionLevelMax_mAP=0.385.pth (The Cautious Listener): This model is the true "Sound Event Detector."
# It provides precise, time-stamped Eventogram visualizations. It's a careful observer, sometimes missing subtle
# events but reliably capturing time-sensitive changes. It excels at detecting nuanced human sounds like laughter.

# checkpoint_path=Cnn14_mAP=0.431.pth
# Cnn14_mAP=0.431.pth (The Enthusiastic Amplifier): This model is actually a "Sound Tagger" forced into SED.
# It tends to be overconfident, acting as an "Enthusiastic Amplifier" for many sound events. It often
# exaggerates detections (like "barking" for splashing dog sounds) but can sometimes capture the overall
# emotional tone or dominant events of a scene more vividly due to its broad-stroke tagging nature. It is less precise, less fine. It detects "Silence" and "Male speech" more readily. 







#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# audioset_tagging_cnn inference launcher – model + type paired automatically
# ------------------------------------------------------------------------------

set -u   # treat unset variables as error
set -e   # exit on error

# Base folder where all .pth files live
MODELS_DIR="/media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn"

# ──────────────────────────────────────────────────────────────────────────────
#  Model catalog – add new entries here when you download more checkpoints
#  Format:  nickname="model_type checkpoint_filename description"
# ──────────────────────────────────────────────────────────────────────────────

declare -A MODELS

MODELS["cautious"]="Cnn14_DecisionLevelMax     Cnn14_DecisionLevelMax_mAP=0.385.pth      The Cautious Listener – true SED, precise timestamps, good at subtle human sounds (laughter, speech nuances), careful with boundaries"

MODELS["enthusiastic"]="Cnn14                  Cnn14_mAP=0.431.pth                        The Enthusiastic Amplifier – clip-level tagger forced into SED, overconfident, broad strokes, exaggerates some events, better at silence/male speech detection"

MODELS["hybrid-strong"]="Wavegram_Logmel_Cnn14  Wavegram_Logmel_Cnn14_mAP=0.439.pth        Strongest reported model (paper mAP 0.439), hybrid waveform+logmel, potentially best overall sensitivity"

MODELS["16k-high"]="Cnn14_16k                  Cnn14_16k_mAP=0.438.pth                    Slightly higher mAP than base CNN14, trained at 16 kHz – good for distant/low-SNR speech if input is resampled"

# ──────────────────────────────────────────────────────────────────────────────
#  Choose which model to use – change only this line
# ──────────────────────────────────────────────────────────────────────────────

SELECTED="cautious"           # ← change this to: cautious / enthusiastic / hybrid-strong / 16k-high / etc.

# ──────────────────────────────────────────────────────────────────────────────
#  Resolve model type + path
# ──────────────────────────────────────────────────────────────────────────────

if [[ -z "${MODELS[$SELECTED]-}" ]]; then
    echo "ERROR: Unknown model nickname '$SELECTED'"
    echo "Available choices:"
    for k in "${!MODELS[@]}"; do
        echo "  $k"
    done
    exit 1
fi

read -r model_type filename description <<< "${MODELS[$SELECTED]}"

checkpoint_path="$MODELS_DIR/$filename"

if [[ ! -f "$checkpoint_path" ]]; then
    echo "ERROR: Checkpoint not found:"
    echo "  $checkpoint_path"
    echo ""
    echo "Did you download it? Expected filename: $filename"
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
#  Show what we are doing
# ──────────────────────────────────────────────────────────────────────────────

cat <<EOF

┌──────────────────────────────────────────────────────────────┐
│  Selected model:   $SELECTED
│  Nickname:         $description
│  Model type:       $model_type
│  Checkpoint:       $checkpoint_path
└──────────────────────────────────────────────────────────────┘

EOF
 
echo "Reminder:"
echo "  • If you see errors with sox / mp3 parsing → install sox + libsox-fmt-mp3, test it via 'sox --info "$@" ' "
echo "  • Some models (non-DecisionLevel*) may give averaged / less precise eventograms"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
#  Run inference
# ──────────────────────────────────────────────────────────────────────────────

time python3 pytorch/audioset_tagging_cnn_inference_6.py \
    --model_type="$model_type" \
    --checkpoint_path="$checkpoint_path" \
    --dynamic_eventogram \
    "$@"
    
    

#time python3 pytorch/audioset_tagging_cnn_inference_6.py  --model_type=Cnn14_DecisionLevelMax --checkpoint_path="$checkpoint_path"   --dynamic_eventogram "$@"

# FYI: You cannot use '--model_type=Cnn14 --checkpoint_path=Cnn14_mAP=0.431.pth' directly with this script
# as Cnn14 does not provide 'framewise_output', causing a KeyError. This script requires a Decision-level model.
 


