This used to work. Do we need this torchcodec smth btw? 

~/.../GitHub/audioset_tagging_cnn $ bash audio_me.sh /data/data/com.termux/files/home/storage/music/Recordings/20260307T184758.mp3
--- AudioSet Tagging CNN Inference (SED Mode) ---
Model Type: Cnn14_DecisionLevelMax
Checkpoint path: /storage/emulated/0/LLMs/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth
Input file and parameters: /data/data/com.termux/files/home/storage/music/Recordings/20260307T184758.mp3
Add  '--dynamic_eventogram'  or  '--static_eventogram'  for a video of the graph with a moving marker.

Note: If you see errors with sox / mp3 parsing → install sox + libsox-fmt-mp3, test it via 'sox --info /data/data/com.termux/files/home/storage/music/Recordings/20260307T184758.mp3 ' 
Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.
Eventogrammer, version 6.4.1. Recently changed:  * Added a static_eventogram argument. Reorder artifact creation logic. Added check for coverage package version.
Notes: a file of duration of 30 mins requires 6GB RAM to process, with the processing time ratio: 1 second of orignal duration : 10 seconds to process on a regular 200 GFLOPs, 4 core CPU.
If the file is too long, use e.g. this to split:
mkdir split_input_media && cd split_input_media && ffmpeg -i /data/data/com.termux/files/home/storage/music/Recordings/20260307T184758.mp3 -c copy -f segment -segment_time 1200 output_%03d.mp4
This script is an adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn so see there if something be amiss.
Note on models:
* `Cnn14_DecisionLevelMax_mAP=0.385.pth` (Sound Event Detection): This model uses Decision-level pooling. It calculates classification probabilities for every small segment of time first, and only then takes the maximum probability to represent the whole clip. Resolution: Cnn14_DecisionLevelMax is specifically designed for Sound Event Detection (SED). Because it maintains the time dimension through the classifier, it can output the framewise_output used by the inference script to generate the 'Eventograms' and the CSV logs.
* The other models are good for audio tagging: use the '--audio_tagging' switch for that mode.

Using moviepy version: 2.1.2
Using torchaudio version: 2.10.0

Using device: cpu
Using CPU.
Copied AI analysis guide to: /data/data/com.termux/files/home/storage/music/Recordings/20260307T184758_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/auditory_cognition_guide_template.md
⏲  🗃️  File duration: 0:08:59
Warning: Failed to load audio directly (Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6, 7, and 8, and we attempt to load libtorchcodec
             for each of those versions. Errors for versions not installed on
             your system are expected; only the error for your installed FFmpeg
             version is relevant. On Windows, ensure you've installed the
             "full-shared" version which ships DLLs.
          2. The PyTorch version (2.10.0) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.

        The following exceptions were raised as we tried to load libtorchcodec:
        
[start of libtorchcodec loading traceback]
FFmpeg version 8:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 7:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 6:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 5:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 4:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported
[end of libtorchcodec loading traceback].). Attempting FFmpeg sanitization...
Error: Unexpected failure during audio loading: Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6, 7, and 8, and we attempt to load libtorchcodec
             for each of those versions. Errors for versions not installed on
             your system are expected; only the error for your installed FFmpeg
             version is relevant. On Windows, ensure you've installed the
             "full-shared" version which ships DLLs.
          2. The PyTorch version (2.10.0) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.

        The following exceptions were raised as we tried to load libtorchcodec:
        
[start of libtorchcodec loading traceback]
FFmpeg version 8:
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py", line 435, in sound_event_detection
    waveform, sr = torchaudio.load(video_input_path)
                   ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchaudio/__init__.py", line 86, in load
    return load_with_torchcodec(
        uri,
    ...<6 lines>...
        backend=backend,
    )
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchaudio/_torchcodec.py", line 82, in load_with_torchcodec
    from torchcodec.decoders import AudioDecoder
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/__init__.py", line 12, in <module>
    from . import decoders, encoders, samplers, transforms  # noqa
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/decoders/__init__.py", line 7, in <module>
    from .._core import AudioStreamMetadata, VideoStreamMetadata
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/__init__.py", line 8, in <module>
    from ._metadata import (
    ...<5 lines>...
    )
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/_metadata.py", line 16, in <module>
    from torchcodec._core.ops import (
    ...<3 lines>...
    )
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 109, in <module>
    ffmpeg_major_version, core_library_path = load_torchcodec_shared_libraries()
                                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 76, in load_torchcodec_shared_libraries
    raise RuntimeError(
    ...<16 lines>...
    )
RuntimeError: Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6, 7, and 8, and we attempt to load libtorchcodec
             for each of those versions. Errors for versions not installed on
             your system are expected; only the error for your installed FFmpeg
             version is relevant. On Windows, ensure you've installed the
             "full-shared" version which ships DLLs.
          2. The PyTorch version (2.10.0) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.

        The following exceptions were raised as we tried to load libtorchcodec:
        
[start of libtorchcodec loading traceback]
FFmpeg version 8:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 7:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 6:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 5:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 4:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported
[end of libtorchcodec loading traceback].

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 7:
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py", line 435, in sound_event_detection
    waveform, sr = torchaudio.load(video_input_path)
                   ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchaudio/__init__.py", line 86, in load
    return load_with_torchcodec(
        uri,
    ...<6 lines>...
        backend=backend,
    )
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchaudio/_torchcodec.py", line 82, in load_with_torchcodec
    from torchcodec.decoders import AudioDecoder
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/__init__.py", line 12, in <module>
    from . import decoders, encoders, samplers, transforms  # noqa
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/decoders/__init__.py", line 7, in <module>
    from .._core import AudioStreamMetadata, VideoStreamMetadata
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/__init__.py", line 8, in <module>
    from ._metadata import (
    ...<5 lines>...
    )
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/_metadata.py", line 16, in <module>
    from torchcodec._core.ops import (
    ...<3 lines>...
    )
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 109, in <module>
    ffmpeg_major_version, core_library_path = load_torchcodec_shared_libraries()
                                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 76, in load_torchcodec_shared_libraries
    raise RuntimeError(
    ...<16 lines>...
    )
RuntimeError: Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6, 7, and 8, and we attempt to load libtorchcodec
             for each of those versions. Errors for versions not installed on
             your system are expected; only the error for your installed FFmpeg
             version is relevant. On Windows, ensure you've installed the
             "full-shared" version which ships DLLs.
          2. The PyTorch version (2.10.0) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.

        The following exceptions were raised as we tried to load libtorchcodec:
        
[start of libtorchcodec loading traceback]
FFmpeg version 8:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 7:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 6:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 5:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 4:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported
[end of libtorchcodec loading traceback].

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 6:
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py", line 435, in sound_event_detection
    waveform, sr = torchaudio.load(video_input_path)
                   ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchaudio/__init__.py", line 86, in load
    return load_with_torchcodec(
        uri,
    ...<6 lines>...
        backend=backend,
    )
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchaudio/_torchcodec.py", line 82, in load_with_torchcodec
    from torchcodec.decoders import AudioDecoder
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/__init__.py", line 12, in <module>
    from . import decoders, encoders, samplers, transforms  # noqa
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/decoders/__init__.py", line 7, in <module>
    from .._core import AudioStreamMetadata, VideoStreamMetadata
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/__init__.py", line 8, in <module>
    from ._metadata import (
    ...<5 lines>...
    )
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/_metadata.py", line 16, in <module>
    from torchcodec._core.ops import (
    ...<3 lines>...
    )
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 109, in <module>
    ffmpeg_major_version, core_library_path = load_torchcodec_shared_libraries()
                                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 76, in load_torchcodec_shared_libraries
    raise RuntimeError(
    ...<16 lines>...
    )
RuntimeError: Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6, 7, and 8, and we attempt to load libtorchcodec
             for each of those versions. Errors for versions not installed on
             your system are expected; only the error for your installed FFmpeg
             version is relevant. On Windows, ensure you've installed the
             "full-shared" version which ships DLLs.
          2. The PyTorch version (2.10.0) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.

        The following exceptions were raised as we tried to load libtorchcodec:
        
[start of libtorchcodec loading traceback]
FFmpeg version 8:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 7:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 6:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 5:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 4:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported
[end of libtorchcodec loading traceback].

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 5:
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py", line 435, in sound_event_detection
    waveform, sr = torchaudio.load(video_input_path)
                   ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchaudio/__init__.py", line 86, in load
    return load_with_torchcodec(
        uri,
    ...<6 lines>...
        backend=backend,
    )
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchaudio/_torchcodec.py", line 82, in load_with_torchcodec
    from torchcodec.decoders import AudioDecoder
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/__init__.py", line 12, in <module>
    from . import decoders, encoders, samplers, transforms  # noqa
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/decoders/__init__.py", line 7, in <module>
    from .._core import AudioStreamMetadata, VideoStreamMetadata
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/__init__.py", line 8, in <module>
    from ._metadata import (
    ...<5 lines>...
    )
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/_metadata.py", line 16, in <module>
    from torchcodec._core.ops import (
    ...<3 lines>...
    )
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 109, in <module>
    ffmpeg_major_version, core_library_path = load_torchcodec_shared_libraries()
                                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 76, in load_torchcodec_shared_libraries
    raise RuntimeError(
    ...<16 lines>...
    )
RuntimeError: Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6, 7, and 8, and we attempt to load libtorchcodec
             for each of those versions. Errors for versions not installed on
             your system are expected; only the error for your installed FFmpeg
             version is relevant. On Windows, ensure you've installed the
             "full-shared" version which ships DLLs.
          2. The PyTorch version (2.10.0) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.

        The following exceptions were raised as we tried to load libtorchcodec:
        
[start of libtorchcodec loading traceback]
FFmpeg version 8:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 7:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 6:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 5:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 4:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported
[end of libtorchcodec loading traceback].

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 4:
Traceback (most recent call last):
  File "/data/data/com.termux/files/home/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py", line 435, in sound_event_detection
    waveform, sr = torchaudio.load(video_input_path)
                   ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchaudio/__init__.py", line 86, in load
    return load_with_torchcodec(
        uri,
    ...<6 lines>...
        backend=backend,
    )
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchaudio/_torchcodec.py", line 82, in load_with_torchcodec
    from torchcodec.decoders import AudioDecoder
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/__init__.py", line 12, in <module>
    from . import decoders, encoders, samplers, transforms  # noqa
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/decoders/__init__.py", line 7, in <module>
    from .._core import AudioStreamMetadata, VideoStreamMetadata
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/__init__.py", line 8, in <module>
    from ._metadata import (
    ...<5 lines>...
    )
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/_metadata.py", line 16, in <module>
    from torchcodec._core.ops import (
    ...<3 lines>...
    )
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 109, in <module>
    ffmpeg_major_version, core_library_path = load_torchcodec_shared_libraries()
                                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 76, in load_torchcodec_shared_libraries
    raise RuntimeError(
    ...<16 lines>...
    )
RuntimeError: Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6, 7, and 8, and we attempt to load libtorchcodec
             for each of those versions. Errors for versions not installed on
             your system are expected; only the error for your installed FFmpeg
             version is relevant. On Windows, ensure you've installed the
             "full-shared" version which ships DLLs.
          2. The PyTorch version (2.10.0) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.

        The following exceptions were raised as we tried to load libtorchcodec:
        
[start of libtorchcodec loading traceback]
FFmpeg version 8:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 7:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 6:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 5:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported

FFmpeg version 4:
Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported
[end of libtorchcodec loading traceback].

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_core/ops.py", line 56, in load_torchcodec_shared_libraries
    core_library_path = _get_extension_path(core_library_name)
  File "/data/data/com.termux/files/usr/lib/python3.13/site-packages/torchcodec/_internally_replaced_utils.py", line 25, in _get_extension_path
    raise NotImplementedError(f"{sys.platform = } is not not supported")
NotImplementedError: sys.platform = 'android' is not not supported
[end of libtorchcodec loading traceback].

real	0m36.971s
user	0m33.182s
sys	0m4.398s

⚠️  Warning: full_event_log.csv not found at /storage/emulated/0/Music/Recordings/20260307T184758_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/full_event_log.csv. Skipping dashboard.

~/.../GitHub/audioset_tagging_cnn $ pipe torchcodec
Special pip install, verbose, upgrade-strategy=only-if-needed etc: 
Using pip 26.0.1 from /data/data/com.termux/files/usr/lib/python3.13/site-packages/pip (python 3.13)
Looking in indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cpu, https://termux-user-repository.github.io/pypi/
Requirement already satisfied: torchcodec in /data/data/com.termux/files/usr/lib/python3.13/site-packages (0.10.0a0)
	Command being timed: "pip install -U --no-build-isolation --upgrade-strategy=only-if-needed -v --retries 10 --resume-retries 10 --timeout 60 --no-binary :all --extra-index-url https://termux-user-repository.github.io/pypi/ torchcodec"
	User time (seconds): 5.43
	System time (seconds): 1.05
	Percent of CPU this job got: 73%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:08.87
	Average shared text size (kbytes): 0
	Average unshared data size (kbytes): 0
	Average stack size (kbytes): 0
	Average total size (kbytes): 0
	Maximum resident set size (kbytes): 61888
	Average resident set size (kbytes): 0
	Major (requiring I/O) page faults: 184
	Minor (reclaiming a frame) page faults: 31710
	Voluntary context switches: 2723
	Involuntary context switches: 229
	Swaps: 0
	File system inputs: 220624
	File system outputs: 280
	Socket messages sent: 0
	Socket messages received: 0
	Signals delivered: 0
	Page size (bytes): 4096
	Exit status: 0
~/.../GitHub/audioset_tagging_cnn $ 


