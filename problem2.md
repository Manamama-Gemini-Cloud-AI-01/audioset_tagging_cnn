Something has changed with python modules and see below: 


root@localhost:~/Downloads/GitHub/audioset_tagging_cnn# bash audio_me.sh "/storage/emulated/0/Music/Recordings/20260227T093045 coffe maker_2.mp3"
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /storage/emulated/0/LLMs/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /storage/emulated/0/Music/Recordings/20260227T093045 coffe maker_2.mp3
Add  '--dynamic_eventogram'  or  '--static_eventogram'  for a video of the graph with a moving marker.

Note: If you see errors with sox / mp3 parsing → install sox + libsox-fmt-mp3, test it via 'sox --info /storage/emulated/0/Music/Recordings/20260227T093045 coffe maker_2.mp3 ' 
Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
Eventogrammer, version 6.3.4. Recently changed:  * Fixed path bug in output directory naming. Added auto-sanitization of corrupt audio streams via FFmpeg. Integrated portable interactive Plotly dashboard (Top 50). Removed manifest generation. Added a static_eventogram argument. Reorder artifact creation logic.
Notes: a file of duration of 30 mins requires 6GB RAM to process, with the processing time ratio: 1 second of orignal duration : 10 seconds to process on a regular 200 GFLOPs, 4 core CPU.
If the file is too long, use e.g. this to split:
mkdir split_input_media && cd split_input_media && ffmpeg -i /storage/emulated/0/Music/Recordings/20260227T093045 coffe maker_2.mp3 -c copy -f segment -segment_time 1200 output_%03d.mp4
This script is an adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn so see there if something be amiss.
Note on models:
* `Cnn14_DecisionLevelMax_mAP=0.385.pth` (Sound Event Detection): This model uses Decision-level pooling. It calculates classification probabilities for every small segment of time first, and only then takes the maximum probability to represent the whole clip. Resolution: Cnn14_DecisionLevelMax is specifically designed for Sound Event Detection (SED). Because it maintains the time dimension through the classifier, it can output the framewise_output used by the inference script to generate the 'Eventograms' and the CSV logs.
* The other models are good for audio tagging: use the '--audio_tagging' switch for that mode.

Using moviepy version: 2.1.2
Using torchaudio version: 2.6.0

Using device: cpu
Using CPU.
Copied AI analysis guide to: /storage/emulated/0/Music/Recordings/20260227T093045 coffe maker_2_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/auditory_cognition_guide_template.md
Traceback (most recent call last):
  File "/root/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py", line 1052, in <module>
    sound_event_detection(args)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/root/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py", line 335, in sound_event_detection
    model = Model(sample_rate=sample_rate, window_size=window_size,
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                  classes_num=classes_num)
  File "/data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn/pytorch/models.py", line 3027, in __init__
    self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                 ~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        win_length=window_size, window=window, center=center, pad_mode=pad_mode,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        freeze_parameters=True)
        ^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torchlibrosa/stft.py", line 645, in __init__
    self.stft = STFT(n_fft=n_fft, hop_length=hop_length,
                ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        win_length=win_length, window=window, center=center,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        pad_mode=pad_mode, freeze_parameters=True)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torchlibrosa/stft.py", line 190, in __init__
    fft_window = librosa.filters.get_window(window, self.win_length, fftbins=True)
                 ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/lazy_loader/__init__.py", line 79, in __getattr__
    return importlib.import_module(f"{package_name}.{name}")
           ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.13/importlib/__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 1026, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/usr/local/lib/python3.13/dist-packages/librosa/filters.py", line 52, in <module>
    from numba import jit
  File "/usr/local/lib/python3.13/dist-packages/numba/__init__.py", line 92, in <module>
    from numba.core.decorators import (cfunc, jit, njit, stencil,
                                       jit_module)
  File "/usr/local/lib/python3.13/dist-packages/numba/core/decorators.py", line 13, in <module>
    from numba.stencils.stencil import stencil
  File "/usr/local/lib/python3.13/dist-packages/numba/stencils/stencil.py", line 11, in <module>
    from numba.core import types, typing, utils, ir, config, ir_utils, registry
  File "/usr/local/lib/python3.13/dist-packages/numba/core/ir_utils.py", line 14, in <module>
    from numba.core.extending import _Intrinsic
  File "/usr/local/lib/python3.13/dist-packages/numba/core/extending.py", line 20, in <module>
    from numba.core.pythonapi import box, unbox, reflect, NativeValue  # noqa: F401
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/numba/core/pythonapi.py", line 12, in <module>
    from numba.core import (
        types, utils, config, lowering, cgutils, imputils, serialize,
    )
  File "/usr/local/lib/python3.13/dist-packages/numba/core/lowering.py", line 19, in <module>
    from numba.misc.coverage_support import get_registered_loc_notify
  File "/usr/local/lib/python3.13/dist-packages/numba/misc/coverage_support.py", line 114, in <module>
    class NumbaTracer(coverage.types.Tracer):
                      ^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'coverage.types' has no attribute 'Tracer'

real	0m16.654s
user	0m7.796s
sys	0m2.798s

root@localhost:~/Downloads/GitHub/audioset_tagging_cnn# mediainfo  "/storage/emulated/0/Music/Recordings/20260227T093045 coffe maker_2.mp3"
General
Complete name                            : /storage/emulated/0/Music/Recordings/20260227T093045 coffe maker_2.mp3
Format                                   : MPEG Audio
File size                                : 341 KiB
Duration                                 : 1 min 9 s
Overall bit rate mode                    : Variable
Overall bit rate                         : 40.0 kb/s
Writing library                          : LAME3.100

Audio
Format                                   : MPEG Audio
Format version                           : Version 2
Format profile                           : Layer 3
Duration                                 : 1 min 9 s
Bit rate mode                            : Variable
Bit rate                                 : 40.0 kb/s
Minimum bit rate                         : 8 000 b/s
Channel(s)                               : 1 channel
Sampling rate                            : 16.0 kHz
Frame rate                               : 27.778 FPS (576 SPF)
Compression mode                         : Lossy
Stream size                              : 341 KiB (100%)
Writing library                          : LAME3.100
Encoding settings                        : -m m -V 2 -q 0 -lowpass 8 --vbr-new -b 8


root@localhost:~/Downloads/GitHub/audioset_tagging_cnn# 
