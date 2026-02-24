

checkpoint_path=Cnn14_DecisionLevelMax_mAP=0.385.pth
#checkpoint_path=Cnn14_mAP=0.431.pth

# Model Personalities:
# Cnn14_DecisionLevelMax_mAP=0.385.pth (The Cautious Listener): This model is the true "Sound Event Detector."
# It provides precise, time-stamped Eventogram visualizations. It's a careful observer, sometimes missing subtle
# events but reliably capturing time-sensitive changes. It excels at detecting nuanced human sounds like laughter.

# Cnn14_mAP=0.431.pth (The Enthusiastic Amplifier): This model is actually a "Sound Tagger" forced into SED.
# It tends to be overconfident, acting as an "Enthusiastic Amplifier" for many sound events. It often
# exaggerates detections (like "barking" for splashing dog sounds) but can sometimes capture the overall
# emotional tone or dominant events of a scene more vividly due to its broad-stroke tagging nature.

echo "We are using $checkpoint_path here. "

echo "If problems here with 'sox --info "$@" ', then do install sox and MP3 parser, e.g. 'libsox-fmt-mp3' for torch to parse the source file. "


time python3 pytorch/audioset_tagging_cnn_inference_6.py  --model_type=Cnn14_DecisionLevelMax --checkpoint_path="$checkpoint_path"   --dynamic_eventogram "$@"

# FYI: You cannot use '--model_type=Cnn14 --checkpoint_path=Cnn14_mAP=0.431.pth' directly with this script
# as Cnn14 does not provide 'framewise_output', causing a KeyError. This script requires a Decision-level model.
 


