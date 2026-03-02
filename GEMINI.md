# Technical Documentation

**Source:** Adapted from the original PANNs repository: https://github.com/qiuqiangkong/audioset_tagging_cnn

## 1. Overview

The main script is `audioset_tagging_cnn_inference_6.py` one. This script is a heavily modified inference tool for the PANNs (Pretrained Audio Neural Networks) models, which primary function is to perform **sound event detection (SED)** on the audio track of any given media file. It identifies what sounds are present at any given moment and generates a series of outputs, including data files and video visualizations.

The core of the script is its ability to take a pretrained model, feed it an audio waveform, and produce a time-stamped probability matrix of the 527 sound classes from the AudioSet ontology.

Also there are other experiments here, e.g. /audioset_tagging_cnn/scripts/Shapash_visualization oe. 

## 2. Core Dependencies

- **PyTorch & Torchaudio:** For loading the model and performing tensor operations.
- **NumPy & SciPy:** For numerical data manipulation.
- **Matplotlib:** For generating the static and dynamic "Eventogram" visualizations.
- **MoviePy:** For creating the final video outputs, including the animated marker and overlaying the visualization onto the source video.
- **FFmpeg (system dependency):** Used extensively for media inspection (`ffprobe`) and all video/audio processing tasks (re-encoding, overlaying).

## 3. Execution Flow

The script operates in a single mode: `sound_event_detection`. The process is as follows:

### Step 1: Initialization and Media Inspection

1. **Argument Parsing:** The script takes several command-line arguments, with `audio_path`, `model_type`, and `checkpoint_path` being mandatory.
2. **Device Selection:** It automatically selects a CUDA device if available, otherwise falls back to CPU.
3. **Output Directory:** It creates a unique output directory named after the input file (e.g., `my_video.mp4_audioset_tagging_cnn/`).
4. **Media Probing (`get_duration_and_fps`):** It uses `ffprobe` to extract critical metadata from the input file, including duration, FPS, and resolution. This information is vital for the subsequent visualization steps.
5. **VFR Check:** It checks for Variable Frame Rate (VFR) video. If detected, it re-encodes the video to a Constant Frame Rate (CFR) temporary file to prevent audio/video synchronization issues in the final render. This is a critical robustness feature.

### Step 2: Model Inference

1. **Model Loading:** It loads the specified PANNs model (e.g., `Cnn14_DecisionLevelMax`) and the pretrained weights from the checkpoint file.
2. **Audio Loading & Resampling:** It uses `torchaudio` to load the audio from the media file, converts it to mono, and resamples it to the model's required sample rate (e.g., 32000 Hz).
3. **Chunked Processing:** To handle long audio files without running out of memory, the script processes the audio in 3-minute chunks. It iterates through the waveform, feeding each chunk to the model.
4. **Inference:** For each chunk, the model outputs a `framewise_output` tensor, which is a matrix of `(time_steps, classes_num)` containing the probability of each of the 527 sound classes for each frame.
5. **Concatenation:** The results from all chunks are concatenated into a single, large `framewise_output` matrix representing the entire audio duration.

### Step 3: Visualization and Data Export

1. **Eventogram Generation (`matplotlib`:
   
   *   **Static Image:** It generates a static PNG (`eventogram.png`) showing a spectrogram of the entire audio file on top and a heatmap of the top 10 most prominent sound events on the bottom. It dynamically calculates the left margin to ensure y-axis labels are never cut off.
   
   *   **Dynamic Video (Optional):** If `--dynamic_eventogram` is used, it generates a video where the eventogram is a moving window, showing the top 10 events local to the current playback time.

2. **Video Rendering (`moviepy`):
   
   *   It creates a video clip from the generated eventogram image.
   
   *   It adds an animated red marker that moves across the eventogram, synchronized with the audio timeline.
   
   *   It attaches the original audio to this visualization.
   
   *   If the source was a video, it uses `ffmpeg` to overlay the eventogram video (with adjustable translucency) onto the original source video.

3. **Data Export:**
   
   *   `full_event_log.csv`: A detailed CSV file (wide matrix format) logging every detected sound event probability for every time frame. This is the **primary input** for the Shapash dashboard.
   
   *   `summary_events.csv`: A user-friendly summary that identifies continuous blocks of sound events.
   
   *   `detailed_events_delta_ai_attention_friendly.json`: A momentum-aware map of sound events using probability deltas, specifically optimized for AI attention mechanisms. It uses **Run-Length Encoding (RLE)** to collapse periods of steady-state audio into `{"skip": N}` markers, making it significantly more token-efficient for AI readers. It is the primary tool for forensic analysis of **attacks**, **decays**, and **rhythm**.
   
   *   `interactive_dashboard.html`: A self-contained, portable Plotly dashboard for exploring the top 50 sound events with zoom and filtering.
   
   *   `summary_manifest.json`: A JSON file cataloging all the generated artifacts.

## 4. Shapash Correlations Dashboard

Located at `scripts/Shapash_visualization/launch_correlations_dashboard.py`, this tool provides a deep-dive into acoustic correlations. It uses a Random Forest regressor to explain why a specific sound was detected by looking at the presence of other sounds as features.

**Workflow:**

1. **Step 1 (Inference):** Run `pytorch/audioset_tagging_cnn_inference_6.py` on your audio/video file.
2. **Step 2 (Visualization):** Run `launch_correlations_dashboard.py` on the resulting `full_event_log.csv`.

## 5. Model Performance & Personality Analysis

Based on cross-model comparison tests (e.g., "duck chase" and "marketing hall" recordings), the available PANNs models exhibit distinct behaviors:

### 1. The Cautious Listener (`Cnn14_DecisionLevelMax_mAP=0.385.pth`)

* **Personality:** Sober, conservative, and precise.
* **Pros:** The only model supporting full **Sound Event Detection (SED)** with high-res `framewise_output`. Excellent for precise time-stamping and capturing short, sharp events (like a car "toot").
* **Cons:** Lower overall sensitivity; can miss faint background sounds.
* **Verdict:** **Primary daily-driver** for timeline-accurate analysis.

### 2. The Enthusiastic Amplifier (`Cnn14_mAP=0.431.pth`)

* **Personality:** Overconfident and aggressive.
* **Pros:** High sensitivity; finds "Dog" or "Speech" even in noisy or distant recordings.
* **Cons:** High False Positive rate (hallucinates detail). Only supports **Global Tagging** (no timeline).
* **Verdict:** Best for general "Cast List" identification of long clips.

### 3. The Sensitive Hybrid (`Wavegram_Logmel_Cnn14_mAP=0.439.pth`)

* **Personality:** Sharp but fragile.
* **Pros:** Strongest overall detection (best mAP). Balanced sensitivity between species (e.g., identifies both Dog and Duck correctly).
* **Cons:** Computationally heavier. Prone to padding-related `RuntimeError` on very short (<2s) clips. Global tagging only.
* **Verdict:** Best for complex multi-source scenes where high accuracy is required.

### 4. Technical Constraints

* **The 16k Model:** Requires strict 16kHz/512 window configuration; currently incompatible with the 32kHz pipeline.
* **Sensitivity Thresholding:** PANNs is fundamentally a "Change Detector." The **Detailed Delta JSON** is recommended for distinguishing between biological sources and human imitation (performance).

### 5. Pragmatic Interpretation: Acoustic Metaphors
Because the model was trained on diverse web audio, it often uses "semantic metaphors" to describe sounds it doesn't have a specific label for. The **Detailed Delta JSON** allows an analyst to look past the label and see the **physical signature**:

*   **"Horse/Clip-clop"** → Rhythmic high-heel footsteps on hard pavement.
*   **"Printer/Vacuum"** → The whirring of grinders or the pulsing of pumps in industrial machines (e.g., coffee makers).
*   **"Thunder/Rain"** → Microphone handling noise, cable friction, or wind hitting the device.
*   **"Animal"** → The rhythmic locomotion of the person carrying the recording device.

By analyzing the **Deltas** (Attack/Decay) and **Skips** (Steady state), an AI or human can perform "Acoustic Archeology" to reconstruct the physical reality of the scene.

## 6. Key Technical Details

- **Execution Context:** The script MUST be run from the project's root directory, as it relies on relative paths to load configuration files (e.g., `config.py` which points to the class labels CSV).
- **Eventogram X-Axis Alignment:** The alignment of the animated marker with the static eventogram image is achieved by dynamically calculating the plot's left margin based on the pixel width of the y-axis labels, and then using those same fractional coordinates to guide the marker's animation. This ensures perfect synchronization.
- **Memory Management:** The use of 3-minute chunks for inference is a key feature that allows the script to process very large files without consuming excessive RAM.
- **Auto-Sanitization:** The script includes a "Sanitary Gate" that automatically uses FFmpeg to transcode corrupt audio streams (e.g., missing MPEG headers) into clean PCM format before loading.

## 6. Usage Example

```bash
# Ensure you are in the root directory of the audioset_tagging_cnn project

MODEL_TYPE="Cnn14_DecisionLevelMax"
CHECKPOINT_PATH="/path/to/Cnn14_DecisionLevelMax_mAP=0.385.pth"
INPUT_VIDEO="/path/to/your/video.mp4"

python3 pytorch/audioset_tagging_cnn_inference_6.py \
    "$INPUT_VIDEO" \
    --model_type="$MODEL_TYPE" \
    --checkpoint_path="$CHECKPOINT_PATH" \
    --cuda
```

### Quick Reference Table for Your Files

| Your .pth file                       | Required --model_type    | Expected behavior in your script                | Notes / Recommendation                    |
| ------------------------------------ | ------------------------ | ----------------------------------------------- | ----------------------------------------- |
| Cnn14_DecisionLevelMax_mAP=0.385.pth | `Cnn14_DecisionLevelMax` | Full SED + eventogram (your current baseline)   | Keep using for precise timestamps         |
| Cnn14_mAP=0.431.pth                  | `Cnn14`                  | Good tagging, weaker localization               | Test for comparison on speech strength    |
| Cnn14_16k_mAP=0.438.pth              | `Cnn14_16k`              | Slightly better mAP, needs 16 kHz input ideally | Try if distant speech detection improves  |
| Wavegram_Logmel_Cnn14_mAP=0.439.pth  | `Wavegram_Logmel_Cnn14`  | Potentially strongest overall, hybrid input     | Recommended next test — best paper result |

The values in the model's names correspond to the mAP on AudioSet eval 

You can download them via e.g.  `wget https://zenodo.org/record/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth?download=1` 
