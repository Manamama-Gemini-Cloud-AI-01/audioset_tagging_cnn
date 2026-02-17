it used to work:


Environment at the end of script, local 🎋 prooted system:
Linux localhost 6.17.0-PRoot-Distro #1 SMP PREEMPT_DYNAMIC Fri, 10 Oct 2025 00:00:00 +0000 aarch64 GNU/Linux
PATH: /usr/local/lib/python3.13/dist-packages/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/games:/usr/games:/system/bin:/system/xbin:/data/data/com.termux/files/home/.local/bin/:/root/go/bin
LD_LIBRARY_PATH: /usr/local/lib:/usr/local/lib/python3.13/dist-packages/lib:/usr/local/lib/python3.13/dist-packages/torchaudio/lib:/usr/local/lib/python3.13/dist-packages/lib:/usr/local/lib/python3.13/dist-packages/llama_cpp/lib:
CFLAGS:
LDFLAGS: -lpython3.13
CPPFLAGS:
C_INCLUDE_PATH:
CPLUS_INCLUDE_PATH: /sources/third_party/vulkan/src/include/:
USE_VULKAN:
LLVM_ROOT:
PKG_CONFIG_PATH:
bash: /root/.cargo/env: No such file or directory
root@localhost:~# cd /data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn/
root@localhost:/data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn# bash audio_me.sh "/storage/emulated/0/Music/Recordings/Standard Recordings/Birds1 Standard recording 2.mp3"


Eventogrammer, version 6.1.2. Recently changed:  * Conversion to aac audio codec, always * Using new logic for key audio events
Notes: a file of duration of 30 mins requires 6GB RAM to process, with the processing time ratio: 1 second of orignal duration : 10 seconds to process on a regular 200 GFLOPs, 4 core CPU.
If the file is too long, use e.g. this to split:
mkdir split_input_media && cd split_input_media && ffmpeg -i /storage/emulated/0/Music/Recordings/Standard Recordings/Birds1 Standard recording 2.mp3 -c copy -f segment -segment_time 1200 output_%03d.mp4
This script is an adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn so see there if something be amiss.
Using moviepy version: 2.1.2
Using torchaudio version (better be pinned at version 2.8.0 for a while...): 2.6.0

Using device: cpu
Using CPU.
Copied AI analysis guide to: /storage/emulated/0/Music/Recordings/Standard Recordings/Birds1 Standard recording 2_audioset_tagging_cnn/auditory_cognition_guide_template.md
⏲  🗃️  Input file duration: 0:00:40
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py", line 897, in <module>
    sound_event_detection(args)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py", line 430, in sound_event_detection
    waveform, sr = torchaudio.load(video_input_path)
                   ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/torchaudio/_backend/utils.py", line 205, in load
    return backend.load(uri, frame_offset, num_frames, normalize, channels_first, format, buffer_size)
           ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3/dist-packages/torchaudio/_backend/sox.py", line 44, in load
    ret = sox_ext.load_audio_file(uri, frame_offset, num_frames, normalize, channels_first, format)
  File "/usr/lib/python3/dist-packages/torch/_ops.py", line 1123, in __call__
    return self._op(*args, **(kwargs or {}))
           ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Error loading audio file: failed to open file /storage/emulated/0/Music/Recordings/Standard Recordings/Birds1 Standard recording 2.mp3

real    0m48.414s
user    0m22.542s
sys     0m8.942s
root@localhost:/data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn#
root@localhost:/data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn#
root@localhost:/data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn# apt list | grep torch          
WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

libtorch-cuda-2.6/stable 2.6.0+dfsg-7+b1 arm64
libtorch-cuda-dev/stable 2.6.0+dfsg-7+b1 arm64
libtorch-cuda-test/stable 2.6.0+dfsg-7+b1 arm64
libtorch-dev/stable,now 2.6.0+dfsg-7 arm64 [installed,automatic]
libtorch-test/stable,now 2.6.0+dfsg-7 arm64 [installed,automatic]
libtorch2.6/stable,now 2.6.0+dfsg-7 arm64 [installed,automatic]
python3-torch-cluster/stable 1.6.3-2 arm64
python3-torch-cuda/stable 2.6.0+dfsg-7+b1 arm64
python3-torch-ignite/stable 0.5.1-1 all
python3-torch-scatter/stable 2.1.2-4 arm64
python3-torch-sparse/stable 0.6.18-3 arm64
python3-torch/stable,now 2.6.0+dfsg-7 arm64 [installed,automatic]
python3-torchaudio/stable,now 2.6.0-1 arm64 [installed]
python3-torchvision/stable,now 0.21.0-3 arm64 [installed]
root@localhost:/data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn#
