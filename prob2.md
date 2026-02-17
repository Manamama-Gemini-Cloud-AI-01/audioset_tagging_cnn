~/.../GitHub/audioset_tagging_cnn $ bash audio_me.sh "/storage/emulated/0/Music/Recordings/Standard Recordings/Birds1 Standard recording 2.mp3"
If problems here with 'sox info /storage/emulated/0/Music/Recordings/Standard Recordings/Birds1 Standard recording 2.mp3 ', then do install sox and MP3 parser, e.g. 'libsox-fmt-mp3' for torch to parse the source file.
Eventogrammer, version 6.1.2. Recently changed:  * Conversion to aac audio codec, always * Using new logic for key audio events
Notes: a file of duration of 30 mins requires 6GB RAM to process, with the processing time ratio: 1 second of orignal duration : 10 seconds to process on a regular 200 GFLOPs, 4 core CPU.
If the file is too long, use e.g. this to split:
mkdir split_input_media && cd split_input_media && ffmpeg -i /storage/emulated/0/Music/Recordings/Standard Recordings/Birds1 Standard recording 2.mp3 -c copy -f segment -segment_time 1200 output_%03d.mp4
This script is an adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn so see there if something be amiss.
Using moviepy version: 2.1.2
Using torchaudio version (better be pinned at version 2.8.0 for a while...): 2.9.1

Using device: cpu
Using CPU.
Copied AI analysis guide to: /storage/emulated/0/Music/Recordings/Standard Recordings/Birds1 Standard recording 2_audioset_tagging_cnn/auditory_cognition_guide_template.md
/data/data/com.termux/files/usr/lib/python3.12/site-packages/numba/cpython/hashing.py:477: UserWarning: FNV hashing is not implemented in Numba. See PEP 456 https://www.python.org/dev/peps/pep-0456/ for rationale over not using FNV. Numba will continue to work, but hashes for built in types will be computed using siphash24. This will permit e.g. dictionaries to continue to behave as expected, however anything relying on the value of the hash opposed to hash as a derived property is likely to not work as expected.
  warnings.warn(msg)
⏲  🗃️  Input file duration: 0:00:40
audio_me.sh: line 3: 14992 Segmentation fault         python3 pytorch/audioset_tagging_cnn_inference_6.py --model_type=Cnn14_DecisionLevelMax --checkpoint_path=Cnn14_DecisionLevelMax_mAP=0.385.pth --dynamic_eventogram "$@"

real    0m27.176s
user    0m18.039s
sys     0m5.831s
~/.../GitHub/audioset_tagging_cnn $
