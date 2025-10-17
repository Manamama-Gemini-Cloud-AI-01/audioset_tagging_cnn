# Implementation Plan: Dynamic Scene-Based Eventogram

This document outlines the implementation strategy for creating a dynamic, scene-based eventogram visualization from audio and video files.

## 1. Objective

To create a seamless and visually continuous MP4 eventogram that dynamically updates the displayed sound events based on their prominence in different segments ("scenes") of the media file. This will provide a more accurate and informative visualization than the current static, globally-determined eventogram.

## 2. Core Methodology: "Post-processing Scene Detection"

The implementation will follow a "Post-processing Scene Detection" approach. This involves analyzing the entire audio's sound event data first, then identifying scene boundaries, and finally rendering the video in segments corresponding to these scenes.

This approach ensures a visually seamless final product where the timeline and marker are continuous, and the eventogram's Y-axis labels change only when the soundscape itself changes significantly.

## 3. Implementation Steps

### Step 1: Initial Full-File Analysis

- **Goal:** Obtain the complete `framewise_output` data matrix (timestamps vs. event probabilities) for the entire audio file.
- **Action:** The initial audio loading and model inference process will be used to generate this data.
- **Challenge (OOM):** For very large files, this step can cause Out-of-Memory errors. The existing chunking mechanism within `inference.py` (processing in 3-minute segments) will be leveraged to build the complete `framewise_output` in memory without crashing.

### Step 2: Scene Detection Logic

- **Goal:** Programmatically identify the boundaries of "scenes" where the set of top N active sound events is stable.
- **Action:** A new function, `detect_scenes(framewise_output, scene_change_threshold)`, will be implemented.
    - This function will iterate through the `framewise_output` data in small time steps (e.g., every second).
    - At each step, it will determine the top N (e.g., 10) most probable events.
    - It will compare the current set of top events with the set from the previous step. If the difference (e.g., number of new events entering the top list) exceeds a certain threshold, a "scene change" is declared, and the timestamp is recorded.
    - The function will return a list of scene start and end times.

### Step 3: Scene-by-Scene Video Clip Generation

- **Goal:** Create an individual `moviepy` video clip for each detected scene.
- **Action:** A loop will iterate through the list of scenes identified in Step 2. For each scene:
    1.  **Generate Scene Eventogram:** A static `matplotlib` image of the eventogram will be generated. This image will display only the top N events relevant to that specific scene. The X-axis will correspond to the full duration of the media file to ensure the marker position is always correct relative to the whole timeline.
    2.  **Create Scene Video Clip:** The generated image will be used to create a `moviepy.ImageClip` with a duration equal to the scene's length.
    3.  **Animate Marker:** The moving marker will be animated on top of this clip, with its position calculated relative to the full media duration, ensuring it moves correctly across the scene.

### Step 4: Final Video Assembly

- **Goal:** Combine all scene-based clips into a single, final MP4 file.
- **Action:**
    1.  The list of `moviepy` clips generated in Step 3 will be concatenated in order using `moviepy.concatenate_videoclips`.
    2.  The original audio from the source file will be extracted and set as the audio for the concatenated video clip.
    3.  The final video will be rendered to a file (e.g., `..._dynamic_eventogram.mp4`).

### Step 5: Preservation of Existing Outputs

- **Static Summary PNG:** The script will **continue to generate the static `.png` file** based on the global top-10 events. This provides a useful high-level summary and maintains backward compatibility.
- **Detailed CSV:** The full CSV log will also be generated as it is now, serving as the complete "truth" of all detected events.

## 4. New Python File

This logic will be implemented in a new file, `pytorch/inference_dynamic.py`, to separate it from the existing, stable `inference.py` script. The new file will start as a copy of the original and will be modified to include the scene detection and dynamic rendering logic.
