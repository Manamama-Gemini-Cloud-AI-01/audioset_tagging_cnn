# Technical Documentation

**Source:** Adapted from the original PANNs repository: https://github.com/qiuqiangkong/audioset_tagging_cnn

## 1. Overview

The main script is `audioset_tagging_cnn_inference_6.py` one. This script is a heavily modified inference tool for the PANNs (Pretrained Audio Neural Networks) models, which primary function is to perform **sound event detection (SED)** on the audio track of any given media file. It identifies what sounds are present at any given moment and generates a series of outputs, including data files and video visualizations.

The core of the script is its ability to take a pretrained model, feed it an audio waveform, and produce a time-stamped probability matrix of the 527 sound classes from the AudioSet ontology.

Also there are other experiments here, e.g. /audioset_tagging_cnn/scripts/Shapash_visualization oe. 

## 2. Core Dependencies

- **PyTorch & Torchaudio:** For loading the model and performing tensor operations.
- **TorchCodec:** For high-performance, sample-accurate, and memory-efficient audio chunked decoding.
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
   
   * **Static Image:** It generates a static PNG (`eventogram.png`) showing a spectrogram of the entire audio file on top and a heatmap of the top 10 most prominent sound events on the bottom. It dynamically calculates the left margin to ensure y-axis labels are never cut off.
   
   * **Dynamic Video (Optional):** If `--dynamic_eventogram` is used, it generates a video where the eventogram is a moving window, showing the top 10 events local to the current playback time.

2. **Video Rendering (`moviepy`):
   
   * It creates a video clip from the generated eventogram image.
   
   * It adds an animated red marker that moves across the eventogram, synchronized with the audio timeline.
   
   * It attaches the original audio to this visualization.
   
   * If the source was a video, it uses `ffmpeg` to overlay the eventogram video (with adjustable translucency) onto the original source video.

3. **Data Export:**
   
   * `full_event_log.csv`: A detailed CSV file (wide matrix format) logging every detected sound event probability for every time frame. This is the **primary input** for the Shapash dashboard.
   
   * `summary_events.csv`: A user-friendly summary that identifies continuous blocks of sound events.
   
   * `detailed_events_delta_ai_attention_friendly.json`: A momentum-aware map of sound events using probability deltas, specifically optimized for AI attention mechanisms. It uses **Run-Length Encoding (RLE)** to collapse periods of steady-state audio into `{"skip": N}` markers, making it significantly more token-efficient for AI readers. It is the primary tool for forensic analysis of **attacks**, **decays**, and **rhythm**.
   
   * `interactive_dashboard.html`: A self-contained, portable Plotly dashboard for exploring the top 50 sound events with zoom and filtering.
   
   * `summary_manifest.json`: A JSON file cataloging all the generated artifacts.

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

* **"Horse/Clip-clop"** → Rhythmic high-heel footsteps on hard pavement.
* **"Printer/Vacuum"** → The whirring of grinders or the pulsing of pumps in industrial machines (e.g., coffee makers).
* **"Thunder/Rain"** → Microphone handling noise, cable friction, or wind hitting the device.
* **"Animal"** → The rhythmic locomotion of the person carrying the recording device.

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

## 9. Memory-Efficient Architecture & Aesthetic Decoupling (v6.12.3 Update)

To handle long-form recordings (e.g., >10 hours) and ensure high-fidelity visualization, the inference script uses a **Dual-Stream Data Architecture**. This "Aesthetic Decoupling" prevents forensic data settings from distorting visual outputs.

### 1. The Dual-Stream Pipeline

The script maintains two separate data paths to optimize for different goals:

* **The Forensic Stream (HDF5):**
  
  * **Source:** Direct inference output at `csv_fps` (Default 5 Hz, up to 100 Hz).
  * **Destination:** `full_event_log.h5`.
  * **Purpose:** Scientific record for downstream analysis (Shapash, AI training).
  * **Persistence:** High-resolution probability matrices and precise timestamps.

* **The Visual Stream (RAM Buffer & HDF5 `framewise_viz`):**
  
  * **Source:** Max-pooled inference output at `effective_viz_fps` (capped to ~2500 horizontal columns).
  * **Destination:** Static Eventogram (PNG) and Interactive Dashboard (Plotly).
  * **Purpose:** Smooth, time-aligned visualization for human "Acoustic Archeology."
  * **Stability Fix:** This stream is generated *before* Phase 10 reloads high-res data, ensuring the "Staircase/Blockiness" bug (caused by sampling rate mismatches) is eliminated.

### 2. The "TorchCodec" Engine (v6.12.3)

The project has moved beyond `soundfile` and `torchaudio.load()` for chunked processing.

- **Surgical Decoding:** We now use `torchcodec.decoders.AudioDecoder` for sample-accurate, O(1) seeking. 
- **Direct Tensor I/O:** Audio is decoded directly into PyTorch tensors, eliminating the overhead of NumPy conversion and minimizing memory pressure.
- **Constant Memory Profile:** RAM usage remains flat (approx. 2.4 GB RSS for the WALL-E test case) regardless of the file duration, as only active 3-minute chunks are decoded and processed at a time.

### 3. HDF5 Persistence & Skip-Inference Path

Visualization fidelity is now preserved across reruns:

- **Metadata Persistence:** `effective_viz_fps` and the `framewise_viz` buffer are stored as HDF5 attributes/datasets.
- **Skip Logic:** When skipping inference, the script reloads the `framewise_viz` stream specifically for Plotly and PNG generation, ensuring identical visual results even if `--csv_fps` was modified between runs.

## 10. HDF5 Bottleneck Removal (CPU Saturation)

To achieve 100% CPU utilization (800% on 8 cores), the HDF5 logging pipeline was optimized:

- **No In-Loop Compression:** Single-threaded Gzip compression was removed from the inference loop. This prevents the "GIL Lock" that previously caused the CPU to idle while waiting for one core to compress a chunk.
- **Pre-Allocation:** HDF5 datasets are now pre-allocated based on the file duration. This eliminates the metadata overhead of resizing the file 170+ times during the run.
- **Buffered Writes:** Redundant `h5_file.flush()` calls were removed, allowing the OS to manage disk I/O in the background without stalling the AI math.

## 11. Performance Baseline (Verified 2026)

On an 8-core CPU system (Ubuntu), the optimized pipeline achieves:

- **Throughput:** ~20x real-time speed (8.6 hours of audio analyzed in <26 minutes).
- **RAM Footprint:** Constant 3.2 GB Max RSS (No swapping).

On Android/Termux (aarch64) with the `master` stable branch (CBR MP3 + HDF5 Visualization):

- **Throughput:** ~3.2x real-time speed (1.5 hours of audio analyzed in ~26 minutes).
- **RAM Footprint:** ~3.8 GB Peak RSS (Comfortably within 5.7 GB RAM limit).
- **Stability:** CBR pre-encoding + HDF5 persistence eliminates decoder-drift and 'libmpg123' sync errors on long-form (1h+) media.



More results, further to: `time python test_specs.py` tests: 

| Category                  | Proot Debian (6.17 kernel) | Termux (4.14 Android kernel) | Ratio / Winner           |
| ------------------------- | -------------------------- | ---------------------------- | ------------------------ |
| **Wall time** (full diag) | **53.3s**                  | **106.9s**                   | **~2.0x** Proot          |
| **CPU %**                 | 345%                       | 411% (but misleading)        | Proot more effective     |
| **User time**             | 178.6s                     | 362.8s                       | ~2x more work in Termux  |
| **System time**           | **5.4s**                   | **76.9s**                    | **~14x** more overhead   |
| **Minor page faults**     | **718k**                   | **6.73M**                    | **~9.4x** more in Termux |
| **Involuntary ctx sw.**   | **27k**                    | **1.83M**                    | **~67x** more in Termux  |



### Detailed Rediscussion of Hypotheses (Now With Both Sides)

**H1 — BLAS / Threading Backend (Primary culprit)**

- **Proot**: Uses proper **OpenBLAS + libgomp**. Thread scaling shows real (though imperfect) gains. Matmul benefits from multi-threading.
- **Termux**: USE_EIGEN_FOR_BLAS=ON dominates despite OpenBLAS being linked. Scaling is **dead** — 8 threads perform almost identically to 1 thread. This is the biggest single reason for the original script's 176% vs 403% CPU utilization.
  → Eigen in this Termux wheel is not effectively using OpenMP / multi-core for the dense linear algebra that PyTorch falls back to.

**H2 — Memory Allocator & Page Fault Pressure**

- Proot (glibc ptmalloc) reuses memory cleanly → very low minor faults (5.7k in churn test).
- Termux (bionic + scudo/jemalloc-like) releases pages aggressively between chunks → **~12x more minor faults** in the short test, and it scaled to **14x** in the real audio run.
  Cold/warm access ratio is also worse (2.01x vs 1.23x). Android’s allocator + older kernel interacts poorly with PyTorch’s caching allocator under repeated large tensor churn.

**H3 — Thread Scheduling & Kernel Friction**

- Proot: Very low involuntary preemptions (good cache behavior).
- Termux: Extremely high involuntary switches (ratio 306 vs 0.65). The Android kernel (4.14, heavily patched for mobile) + Termux’s LD_PRELOAD (libtermux-exec-direct-ld-preload.so) + possible cpuset/background restrictions keep kicking threads off cores. This destroys locality for convolution and matmul kernels.

**H4 — Raw Hot-Path Throughput (CNN ops)** Proot is **substantially faster** on the exact operations the audio script uses:

- MelSpectrogram: **198ms** vs **364ms** (~1.83x)
- Conv2d block4 (heaviest): **99ms** vs **227ms** (~2.3x)
- MiniCnn14 forward: **354ms** vs **649ms** (~1.83x)

This directly explains why the real inference loop runs ~2x faster in Proot even though both environments "see" 8 cores.









## 12. Plotly Dashboard: The "Staircase" Bug Post-Mortem

Historically, the interactive Plotly dashboard exhibited "blocky" or "staircase" shaped traces. 

- **The Cause:** A "State Mismatch" where the Plotly phase was accidentally using high-resolution forensic data (100 Hz) but attempting to plot it against a low-resolution visualization timeline (5 Hz).
- **The Fix:** Phase 12 (Plotly) was moved to immediately follow Phase 9 (PNG), ensuring it uses the **Aesthetic/Visualization Stream** before any forensic data reloads occur. This ensures organic, smooth, and time-accurate traces.

### 4. Video Rendering Speed Hack (v6.6.3 Update)

To enable full-length video visualization on mobile devices (Android/Termux), the rendering pipeline was refactored to eliminate the "Rendering Bottleneck" caused by redundant Matplotlib and MoviePy compositing calls.

* **The Problem (30 FPS Redundancy):** Previously, both Static and Dynamic modes redrew the entire visualization 30 times per second. Since the internal data resolution is downsampled to 2 FPS, the script was performing identical, expensive calculations 15 times for every unique data point. This locked the CPU's Global Interpreter Lock (GIL), preventing multi-core utilization.
* **The Solution (Data-Synchronous Caching):**
  * **Precompute Phase:** All "Top 10 Events" and "Adaptive Window" logic is now executed once per unique data frame (2 FPS) before rendering begins.
  * **Frame Caching:** The `make_frame` function now caches the last rendered image. For 14 out of every 15 video frames, it simply returns a pre-rendered buffer from RAM, bypassing Matplotlib and MoviePy's blending engine entirely.
  * **OO-Matplotlib Transition:** Switched from `pyplot` to the Object-Oriented API for thread safety and faster canvas draws.
* **Results:** 
  * **Speed:** Rendering jumped from ~17 it/s to **130+ it/s** (a ~7x improvement).
  * **Efficiency:** CPU utilization increased from ~150% to **460%+**, finally saturating all available cores for encoding.
  * **Mobile Impact:** This optimization makes `--dynamic_eventogram` viable on Android devices, where CPU cycles are precious.

## 10. Shapash Unified Dashboard (Performance & Logic)

The "Unified Acoustic Brain" (`launch_multi_target_dashboard.py`) is designed for sub-10 second execution on Android/Termux through four key optimizations:

1. **The "Tiny Forest" Strategy:** It uses a Random Forest with only 10 estimators and a max depth of 5. While small, this is sufficient to capture the dominant acoustic correlations (e.g., "Music" correlating with "Piano") without the heavy compute cost of a 100+ tree model.
2. **Strategic Sampling:** Instead of explaining every single time-frame (which could be 30,000+ for a 5-minute clip), it samples 200 representative points. This "High-Resolution Snapshot" approach reduces SHAP calculation time from hours to seconds.
3. **Unified Ingestion:** The model trains on all 527 sound classes as features simultaneously. This allows the dashboard to show how different sounds compete or coexist in the same environment.
4. **Identity Masking:** To prevent "Semantic Noise," the script automatically masks self-correlations. A sound is never allowed to "explain" itself, forcing the model to find the most relevant *external* predictors (e.g., explaining "Bark" via "Animal" or "Dog").

## 12. Strategic Roadmap: Aesthetic Decoupling (Plan)

While the **Surgical Load** refactor (v6.8.12) solved the audio loading bottleneck, a secondary **Aggregation Creep** still exists in the visualization pipeline. For ultra-long archives (100h+), storing the 5Hz summary data in RAM becomes a new O(n) bottleneck.

### 1. The Concept: "Canvas-Bound Resolution"

The current script "over-samples the canvas"—feeding 180,000 data points into a PNG file that is only 1,275 pixels wide. 

### 2. Proposed Solution: The Visual Cap

- **Decoupled Streams:** Distinguish between the **Forensic Stream** (100Hz data written directly to disk) and the **Visual Stream** (summarized data for the Eventogram and Dashboard).
- **Fixed-Width Aggregation:** Cap the horizontal resolution of in-RAM arrays (`stft` and `framewise_output`) to a constant value (e.g., 2,500 columns). 
- **Running Max-Pool:** As the inference loop processes 3-minute chunks, it will perform a running max-pool to collapse hundreds of data points into a single "Visual Representative." This ensures that a 10ms click is still visually rendered as a dot even in a 10-hour overview.

### 3. Expected Impact

- **RAM Scaling:** Transition from **O(n)** (linear growth) to **O(1)** (constant footprint).
- **Stability:** Analyzing a 10,000-hour file will consume the same amount of RAM as a 10-minute file.
- **Infinite Analysis:** Enables "Forever Recorders" to be analyzed in a single pass on 8GB machines without ever triggering a Swap-storm.

## 13. Operational Insights & Cross-Tool Integration (2026 Update)

### 1. High-Resolution Inference Safety

Contrary to standard research scripts, running `pytorch/audioset_tagging_cnn_inference_6.py` (v6.8.12) on full tracks is **not "scary"** for system resources or terminal context. 

- **Terminal Hygiene:** The script is engineered to output only concise high-level progress and info messages (e.g., "Chunk at Xm finished"). 
- **Data Redirection:** All massive probability matrices and frame-wise logs are redirected directly to disk (`full_event_log.csv`). 
- **CLI Compatibility:** The Gemini CLI environment safely handles any unexpected long printouts by redirecting them to temporary files, ensuring your active context window remains clean for analytical reasoning.

### 2. The "32kHz Ritual" (Inference vs. Native SR)

You may notice that **Inference SR** (32,000 Hz) is often higher than the **Native SR** (e.g., 16,000 Hz). 

- **Mandatory Compatibility:** The CNN models were trained exclusively on 32kHz audio. 
- **Acoustic Focus:** Upsampling does not add new information, but it prevents the model from perceiving the world as "half-speed/octave-lower." It ensures the mathematical "paintbrushes" of the model land on the correct frequencies.

### 3. Environment & Dependency Hacks

- **The Numba/Coverage Conflict:** Some environments (Ubuntu 24.04+) have a version clash between `numba` and `python3-coverage`. If `AttributeError: module 'coverage.types' has no attribute 'Tracer'` occurs, **uninstall the system coverage package** (`sudo apt remove python3-coverage`).
- **The "Hollow" Torchaudio:** If `AttributeError: module 'torchaudio' has no attribute 'info'` occurs, it indicates a binary mismatch (often after a `whisperx` update). The script now detects this and recommends a clean reinstall of `torch` and `torchaudio` via the `--extra-index-url` provided in the "Tips" section.

### 4. Forensic Complement: Essentia

While PANNs excel at identifying the **"Physical Reality"** (e.g., specific instruments, textures, or "Acoustic Metaphors"), it does not track musical grammar like Tempo or Key. To achieve a complete "Acoustic Archeology" profile, integrate **Essentia**:

## 14. Work Log (June 2026)

- **Sanitized Video Handling:** Implemented a "Sanitary Audio Gate" (Phase 4) in `pytorch/audioset_tagging_cnn_inference_6.py` to automatically extract audio from video/non-supported containers to an intermediate MP3 using `ffmpeg`.
- **TorchCodec Transition:** Replaced `soundfile` with `torchcodec` for sample-accurate, memory-efficient chunked audio decoding.
- **Mandatory CBR Stabilization:** Enforced unconditional pre-processing of all input files to CBR MP3, eliminating MP3 seek-drift.
- **HDF5 Visual Persistence:** Added `stft_viz` dataset to HDF5, ensuring background spectrogram visualization is preserved across reruns.
- **Interactive Dashboard Enhancements:** Added "The Acoustic Explorer" functionality. Integrated HTML5 `<video>` player, CSS fixes for full width, and JavaScript for click-to-seek functionality from the Plotly graph.
- **Documentation:** Updated `docs/auditory_cognition_guide_template.md` to reflect HDF5-based data storage and updated the CLI commands.
- **Diagnostic Resolution:** Fixed `BrokenPipeError` issues by refining the sanitization process and resolving script-level import conflicts (`UnboundLocalError`).

## 15. Forensic Temporal Resolution & The "Resolution Bottleneck" (GIGO Warning)

Through extensive stress-testing and artifact inspection (June 2026), a fundamental architectural limit was discovered that renders high-resolution settings (e.g., 100 Hz) largely deceptive for the current pipeline.

### 1. The 3.125 Hz "Thinking Rate"
The primary model, `Cnn14_DecisionLevelMax`, is a 100 Hz "Container" built on a **3.125 Hz Engine**. 
*   **The Culprit:** The model architecture contains 5 sequential pooling layers in the time dimension. 
*   **The Math:** $2 \times 2 \times 2 \times 2 \times 2 = 32$ (downsampling factor). 
*   **The Result:** $100\text{ Hz} / 32 = \mathbf{3.125\text{ Hz}}$. The model only calculates a fresh "thought" every 320ms.

### 2. Deceptive High Resolution (The "Staircase" Effect)
The model uses `nearest` neighbor interpolation to stretch its 3.125 Hz thoughts into a 100 Hz timeline.
*   **The GIGO:** When running at `--log_fps 100`, the script records 100 frames per second, but **24 out of every 25 frames are identical copies** of the first one.
*   **Visual Impact:** This creates "Minecraft-style" flat plateaus in Plotly and MP4 visualizations.
*   **Resource Waste:** Setting FPS to 100 increases file sizes (HDF5, HTML) by 5-10x and rendering time by 10x, but provides **zero new forensic information**.

### 3. The "Ex Nihilo" Lesson for Future Agents
Future AI agents are strictly forbidden from attempting to "optimize" Plotly or MP4 resolution via FPS flags alone. No amount of "Tactical Bulldogging" can carve resolution *ex nihilo* (out of nothing) when the model itself is the bottleneck.

### 4. How to achieve True High Resolution (Plan)
If forensic-grade temporal precision is required, the current `framewise_output` method must be abandoned in favor of:
*   **Sliding Window SED:** Manually moving a 50ms window across the audio and running a **Global Tagging** model (`Cnn14`) on every step. This is computationally expensive but is the only way to recover sub-100ms "Acoustic DNA."

**Current Recommendation:** Use `--log_fps 5 --ui_fps 5` for 99% of use cases. It is the only "Honest" setting for this architecture.

## Branching & Experimental Strategy

- **`master`**: Contains the stable, "traditional" implementation: HDF5 visualization persistence, mandatory CBR MP3 re-encoding, and `torchcodec`-based sample-accurate chunked decoding.
- **`test-torchcodec`**: Used for experimental prototyping of high-risk refactors. Do not use this for production analysis until merged into master.

Tips & Environment Hacks:

* For 'undefined symbol: torch_library_impl' or 'NotImplementedError':
  Run: pip install -U torch torchaudio torchcodec --extra-index-url https://download.pytorch.org/whl/cpu
* All input audio files are now automatically re-encoded to CBR MP3 to ensure consistent seekability and prevent decoder drift issues.
* If Torchcodec errors (e.g., 'libnvrtc.so.13 not found'), simply remove it:
  Run: pip uninstall torchcodec  (The script will safely fallback to FFmpeg)
