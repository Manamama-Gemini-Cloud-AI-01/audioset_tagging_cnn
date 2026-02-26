import torchaudio
import soundfile as sf
import os
import sys

path = "/home/zezen/Downloads/Test/Walk2_fixed_maybe.mp3"

print(f"Testing loading of: {path}")

print("\n1. Testing torchaudio.load()...")
try:
    waveform, sr = torchaudio.load(path)
    print(f"SUCCESS: torchaudio loaded waveform with shape {waveform.shape} and sr {sr}")
except Exception as e:
    print(f"FAILURE: torchaudio failed: {e}")

print("\n2. Testing soundfile.read()...")
try:
    data, sr = sf.read(path)
    print(f"SUCCESS: soundfile read data with shape {data.shape} and sr {sr}")
except Exception as e:
    print(f"FAILURE: soundfile failed: {e}")

print("\n3. Testing torchaudio with backend='ffmpeg' (if available)...")
try:
    # Some versions allow backend specification
    waveform, sr = torchaudio.load(path, backend="ffmpeg")
    print(f"SUCCESS: torchaudio (ffmpeg backend) loaded waveform with shape {waveform.shape} and sr {sr}")
except Exception as e:
    print(f"FAILURE: torchaudio (ffmpeg backend) failed: {e}")
