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

## 7. Usage Example

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

## 8. Strategic Roadmap & Modern Research (2024 Context)

While PANNs (2020) remains a highly reliable "Toyota Hilux" for audio analysis, recent research (2023-2024) has introduced significant architectural upgrades and paradigm shifts.

### 1. The "State of the Art" Gap
Modern successors have pushed the performance boundaries beyond the original PANNs mAP of 0.439:
- **ConvNeXt-Audio (2023):** Achieves **0.471 mAP** by replacing the Cnn14 backbone with a ConvNeXt architecture. It is more accurate and uses 3x fewer parameters (Paper: `2306.00830`).
- **Transformers (AST/MAE):** Moving away from CNNs toward Audio Spectrogram Transformers. These are "smarter" at global context but often heavier and less precise for millisecond-level Sound Event Detection (SED) than this repo's implementation.
- **Synthio (2024):** Explores augmenting datasets with synthetic audio generated by LLMs to improve niche sound detection (Paper: `2410.02056`).

### 2. Strategic Assessment: Codebase Longevity
- **Verdict:** This codebase is **"Classic but not Obsolete."**
- **Core Strength:** Its **Temporal Precision** for Sound Event Detection (SED) and unique **Detailed Delta JSON** (Acoustic Archeology) output make it superior to many modern "tag-only" models for forensic analysis.
- **Recommendation:** Do not rewrite the infrastructure. Instead, perform an **"Engine Swap"**: Keep the high-value "glue code" (MoviePy sync, Eventograms, RLE JSON export) and focus on integrating modern backbones (like ConvNeXt) into the existing inference pipeline.

### 3. How to Search for Future Discoveries
To stay updated with the latest AI audio research, use the Hugging Face Papers API via `curl` to bypass CLI version mismatches:

```bash
# Search for recent audio classification papers
curl -s "https://huggingface.co/api/papers/search?q=audio+classification+AudioSet&limit=10" | jq '.[] | {id: .paper.id, title: .title, date: .publishedAt}'

# Fetch full markdown summary of a specific paper (e.g., ConvNeXt paper)
curl -s -H "Accept: text/markdown" "https://huggingface.co/papers/2306.00830"
```

## 9. Memory-Efficient Architecture (v6.6.0 Update)

To handle long-form recordings (e.g., >30 minutes) on resource-constrained devices like Android (Termux), the inference script was refactored to eliminate "Aggregation Spikes."

### 1. The "Aggregation Bottleneck" Problem
Previously, the script would process 3-minute chunks but keep all high-resolution results (100 FPS) in RAM. For a 34-minute file, this created a massive cumulative memory load:
- **Full Waveform (32kHz):** ~270 MB
- **Full Framewise Matrix (527 classes @ 100 FPS):** ~450 MB
- **Full Spectrogram (Complex STFT):** ~850 MB
- **Result:** Usage would spike toward 3GB, triggering OOM crashes in Termux.

### 2. The "Lean" Solution
The new architecture uses a **"Stream and Prune"** strategy:
- **Direct Disk Streaming:** The high-resolution (100 FPS) probability matrix is written directly to `full_event_log.csv` during the inference loop. RAM usage for this data is now constant (single row) rather than cumulative.
- **50x Internal Downsampling:** All internal RAM structures used for visualization (Eventograms, Dashboard, Spectrogram) are now **max-pooled to 2 FPS** as they are generated. This reduces the visualization RAM footprint by 98% (e.g., 900MB → 18MB).
- **Chunked Post-Processing:** The Spectrogram (STFT) is calculated in 3-minute segments and immediately downsampled, eliminating the whole-file "kill shot" memory spike.
- **Peak Preservation:** By using `max-pooling` during downsampling, the script ensures that short, sharp sound events (like a car toot or a click) are still accurately represented in the 2 FPS graphs even if they fall between sample points.

### 3. Verification in Termux
When running version 6.6.0, the memory usage (`top` or `free`) remains remarkably flat throughout the entire duration of long files, making it safe for mobile deployment.

### 4. Video Rendering Speed Hack (v6.6.3 Update)
To enable full-length video visualization on mobile devices (Android/Termux), the rendering pipeline was refactored to eliminate the "Rendering Bottleneck" caused by redundant Matplotlib and MoviePy compositing calls.

*   **The Problem (30 FPS Redundancy):** Previously, both Static and Dynamic modes redrew the entire visualization 30 times per second. Since the internal data resolution is downsampled to 2 FPS, the script was performing identical, expensive calculations 15 times for every unique data point. This locked the CPU's Global Interpreter Lock (GIL), preventing multi-core utilization.
*   **The Solution (Data-Synchronous Caching):**
    *   **Precompute Phase:** All "Top 10 Events" and "Adaptive Window" logic is now executed once per unique data frame (2 FPS) before rendering begins.
    *   **Frame Caching:** The `make_frame` function now caches the last rendered image. For 14 out of every 15 video frames, it simply returns a pre-rendered buffer from RAM, bypassing Matplotlib and MoviePy's blending engine entirely.
    *   **OO-Matplotlib Transition:** Switched from `pyplot` to the Object-Oriented API for thread safety and faster canvas draws.
*   **Results:** 
    *   **Speed:** Rendering jumped from ~17 it/s to **130+ it/s** (a ~7x improvement).
    *   **Efficiency:** CPU utilization increased from ~150% to **460%+**, finally saturating all available cores for encoding.
    *   **Mobile Impact:** This optimization makes `--dynamic_eventogram` viable on Android devices, where CPU cycles are precious.
