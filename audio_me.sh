echo  "If problems here with 'sox --info "$@" ', then do install sox and MP3 parser, e.g. 'libsox-fmt-mp3' for torch to parse the source file. "

time python3 pytorch/audioset_tagging_cnn_inference_6.py  --model_type=Cnn14_DecisionLevelMax --checkpoint_path=Cnn14_DecisionLevelMax_mAP=0.385.pth --dynamic_eventogram "$@"
