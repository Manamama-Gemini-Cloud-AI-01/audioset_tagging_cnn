# Essentia Music Analysis Suite

This directory contains scripts for "Acoustic Archeology" using the Essentia library. It supports both classic signal processing and advanced neural inference.

## 1. Scripts

### `analyze_music.py` (The Laboratory)
*   **Purpose:** Measures the "Physical Bones" of a track.
*   **Capabilities:** Pitch/Melody extraction, Chord detection, HPCP (Chromagram), and BPM.
*   **Requirements:** Standard `essentia` or `essentia-tensorflow`.

### `essentia_acoustic_dna.py` (The Neural Eye)
*   **Purpose:** Unlocks high-level "Acoustic DNA" using deep learning.
*   **Capabilities:** Multi-target Genre classification (400 classes), Acoustic/Electronic detection, and Voice/Instrumental identification.
*   **Requirements:** Supports two modes:
    1.  **Standard Mode:** Uses `essentia-tensorflow` (Best for PC/Desktop).
    2.  **Bridge Mode:** Uses standard `essentia` + `tensorflow` (Fallback for Termux/Debian PRoot).

## 2. Installation

### Option A: Standard (PC / Desktop Linux)
Best for high-speed analysis using the built-in C++ TensorFlow runtime.
```bash
pip install essentia-tensorflow
```

### Option B: Hybrid Fallback (Termux / Debian PRoot)
Required if `essentia-tensorflow` fails due to ABI compatibility (glibc vs bionic) issues.
```bash
# Install standard essentia (usually built from source or native package)
pip install essentia
# Install standard tensorflow for the "Universal Bridge"
pip install tensorflow
```

## 3. Troubleshooting (Termux / Debian PRoot)

Running neural networks in a mobile PRoot environment can be fragile. If you encounter issues:

1.  **ABI Mismatches:** If you see `ImportError: libtensorflow.so: cannot open shared object file`, you are likely trying to use the PC-based `essentia-tensorflow` wheel in a mobile environment. Switch to **Option B** (Bridge Mode).
2.  **Memory Crashes:** PRoot environments are sensitive to multi-threading. The script automatically limits TensorFlow to 2 threads in Bridge Mode to prevent "Kill" signals from the kernel.
3.  **Path Issues:** Always ensure you are running the script from a directory that can access the `models/` folder in the project root.

## 3. Neural Models Setup

The `essentia_acoustic_dna.py` script expects models to be located in a `models/` directory relative to the project root.

### Download Command
Run these commands from the **project root** to download the required Discogs-EffNet models:

```bash
mkdir -p models && cd models

# Discogs-EffNet Backbone (Feature Extractor)
wget --no-check-certificate https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb

# Genre Classifier (400 labels)
wget --no-check-certificate https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb
wget --no-check-certificate https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json

# Mood/Acoustic Classifier
wget --no-check-certificate https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1.pb
wget --no-check-certificate https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1.json

# Voice/Instrumental Classifier
wget --no-check-certificate https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb
wget --no-check-certificate https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.json
```

## 4. Usage

```bash
# Classic Forensic Analysis
python scripts/Essentia_music_analysis/analyze_music.py "path/to/music.mp3"

# Advanced Neural DNA Analysis
python scripts/Essentia_music_analysis/essentia_acoustic_dna.py "path/to/music.mp3"
```

The scripts will create a dedicated `_essentia_forensics` or `_acoustic_dna` directory alongside your input file containing CSV/JSON artifacts and visualization plots.
