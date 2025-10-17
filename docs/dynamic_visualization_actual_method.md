# Final Method: Dynamic Eventogram via Per-Frame Rendering

This document outlines the final, successful implementation for creating the dynamic eventogram video. This method is a corrected and refined version of the per-frame, sliding-window approach attempted by Grok AI.

## 1. Core Principle: "Lazy" Per-Frame Generation

The initial plans involving scene detection and image concatenation proved to be flawed, primarily due to the difficulty of maintaining a stable plotting canvas across different `matplotlib` images, which resulted in visual corruption.

The successful method abandons pre-rendering images. Instead, it generates each frame of the video on-the-fly using `moviepy`'s `VideoClip` class, which is specifically designed for this purpose.

## 2. How It Works: The `make_frame` Function

The entire logic is built around a single function, `make_frame(t)`, which is called by `moviepy` for every single frame at time `t`.

1.  **Sliding Window:** For the given time `t`, the function selects a "window" of data (e.g., 30 seconds) from the full `framewise_output` matrix. This window is centered around `t`.

2.  **Stable Canvas:** A new `matplotlib` figure is created for the frame. Crucially, to prevent the "jumping" and "shifting" seen in previous attempts, it uses the globally pre-calculated `left_frac` margin. This ensures that the plot area has the exact same size and position in every single frame, regardless of the content.

3.  **Synchronized Plotting:**
    *   Both the upper spectrogram and the lower eventogram are drawn using only the data from the sliding window.
    *   The time-ticks (x-axis labels) for **both plots** are recalculated for every frame to correspond to the time range of the current sliding window. This keeps them perfectly synchronized.

4.  **Corrected Marker:**
    *   A full-height red marker is drawn on both plots.
    *   Its position is calculated to be in the **exact center** of the plot area. This represents the "now" moment, and as the `make_frame` function is called for progressing values of `t`, the underlying data scrolls behind the stationary central marker, creating the desired sliding window effect.
    *   Special logic handles the beginning and end of the video, where the window is not full, ensuring the marker smoothly moves from the left edge to the center and from the center to the right edge.

## 3. Memory Stability (The OOM Fix)

The critical fix that prevented the Out-of-Memory crashes was replacing Grok's flawed `CompositeVideoClip([...])` implementation with `VideoClip(make_frame, duration=...)`.

*   The flawed version created a massive list of all video frames in memory *before* starting to render.
*   The correct `VideoClip` implementation acts as a **generator**. It calls `make_frame` for one frame, renders it, discards it from memory, and then moves to the next. This results in a low and stable memory footprint, making the script capable of processing very long video files without crashing.

## 4. Final Refinements

*   **Static Mode:** The marker in the original static mode was also fixed to use the calculated `left_frac` margin, preventing it from drawing over the Y-axis labels.
*   **Filenaming:** The script now adds a `_dynamic` suffix to the filename of videos generated with the `--dynamic_eventogram` flag, allowing both static and dynamic versions to coexist.
