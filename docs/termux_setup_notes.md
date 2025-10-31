# Notes on setting up `audioset_tagging_cnn` on Termux

This document outlines the steps taken to get the `audioset_tagging_cnn` script working in a Termux environment on Android.

## Summary of Fixes

To get the script to run successfully, we had to:

1.  **Patch the `numba` library:** The installed version of `numba` was incompatible with the installed version of `llvmlite`. We patched the `numba` source code to bypass the version check.
    *   In `/data/data/com.termux/files/usr/lib/python3.12/site-packages/numba/__init__.py`, we commented out the `llvmlite` version check.
    *   In `/data/data/com.termux/files/usr/lib/python3.12/site-packages/numba/core/typed_passes.py`, we commented out calls to the missing `dump_refprune_stats` function.

2.  **Modify the script to use `soundfile`:** The `torchaudio.load` function was unable to load audio files, even with the `ffmpeg` backend specified. We modified the `pytorch/audioset_tagging_cnn_inference_6.py` script to use the `soundfile` library instead.

3.  **Convert MP3 to WAV:** The `soundfile` library was unable to read the input MP3 file, reporting an "unimplemented format" error. We used `ffmpeg` to convert the MP3 to a WAV file before processing.

4.  **Use an absolute path:** The script failed with a `FileNotFoundError` when using a relative path to the audio file. We modified the `audio_me.sh` script to use the absolute path to the audio file.

## Example Workflow

1.  Convert your input audio file to WAV format using `ffmpeg`:
    ```bash
    ffmpeg -i /path/to/your/audio.mp3 /path/to/your/audio.wav
    ```

2.  Run the `audio_me.sh` script with the absolute path to your WAV file:
    ```bash
    bash audio_me.sh /path/to/your/audio.wav
    ```
