#!/usr/bin/env bash
INPUT_DIR="/media/zezen/HP_P7_Data/Videos/Watch_me/SRT/ducks_scenes_10_gemma-Q2"
MODELS_DIR="/media/zezen/HP_P7_Data/Temp/AI_models/audioset_tagging_cnn"

declare -A MODELS
MODELS["cautious"]="Cnn14_DecisionLevelMax Cnn14_DecisionLevelMax_mAP=0.385.pth"
MODELS["enthusiastic"]="Cnn14 Cnn14_mAP=0.431.pth"
MODELS["hybrid"]="Wavegram_Logmel_Cnn14 Wavegram_Logmel_Cnn14_mAP=0.439.pth"
MODELS["16k"]="Cnn14_16k Cnn14_16k_mAP=0.438.pth"

for audio in "$INPUT_DIR"/*_sanitized.wav; do
    base_name=$(basename "$audio")
    echo "================================================================================"
    echo " PROCESSING: $base_name"
    echo "================================================================================"

    for key in "cautious" "enthusiastic" "hybrid" "16k"; do
        read -r model_type filename <<< "${MODELS[$key]}"
        checkpoint="$MODELS_DIR/$filename"
        echo ">>> Model: $key ($model_type)"
        
        # Set sample rate based on model requirements
        sr=32000
        if [[ "$key" == "16k" ]]; then sr=16000; fi

        python3 pytorch/inference.py audio_tagging \
            --model_type="$model_type" \
            --checkpoint_path="$checkpoint" \
            --audio_path="$audio" \
            --sample_rate=$sr \
            --cuda
        echo "--------------------------------------------------------------------------------"
    done
done
