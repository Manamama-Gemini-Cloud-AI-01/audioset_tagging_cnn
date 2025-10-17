# Dynamic Eventogram Visualization via Chunking

## 1. Problem Statement

The current `inference.py` script generates eventogram visualizations (PNG images and MP4 videos) based on a globally determined "top 10" most prominent sound event categories across the *entire duration* of the input audio/video file. While the generated CSV output accurately captures all detected events above a specified probability threshold (e.g., 0.2), the visualizations are limited to this fixed set of 10 events. This means that transient or locally significant sound events, even if their probabilities exceed the threshold, will not appear in the visual output if their overall prominence across the entire file is not high enough to place them in the global top 10.

For example, an event like "Coin (dropping)" might be present in the detailed CSV log for a specific time segment, but it would be absent from the eventogram visualization if other events were globally more dominant throughout the entire media file.

## 1.1. Illustrative Example from Test Data

This problem is clearly demonstrated by comparing the output of two test runs: one on a full audio file (`20251016T091134.mp3`) and another on a smaller chunk of that same audio (`20251016T091134_second_part.mp3`).

-   **Full Audio Run (`...T091134_audioset_tagging_cnn.csv`):** The CSV for the full audio file contains a wide variety of events, including "Ocean," "Waves, surf," "Speech," and "Coin (dropping)." However, the corresponding eventogram visualization only shows the top 10 *globally* dominant events, which may not include "Coin (dropping)" if it was not prominent enough across the entire duration.
-   **Chunked Audio Run (`...second_part_audioset_tagging_cnn.csv`):** When analyzing just the chunk where the "Coin (dropping)" event is prominent, this event *does* make it into the top 10 for that specific segment and is therefore visualized in the corresponding eventogram.

This discrepancy confirms that the current visualization method fails to represent locally significant events when they are part of a larger audio file with other, more globally dominant sounds.

## 2. Goal

To enhance the interpretability and completeness of the eventogram visualization, especially for MP4 video output, by dynamically displaying the most relevant sound events within the *currently viewed time segment*, rather than a static, globally determined set.

## 3. Conceptual Solution for MP4 (Dynamic Visualization)

The core architectural shift required is to move from a **global, static selection of top events** for visualization to a **local, dynamic selection of top events** that changes as the video progresses.

### 3.1. Frame-by-Frame (or Segment-by-Segment) Analysis

Instead of generating a single `top_result_mat` from the overall top 10 events for the entire file, the `framewise_output` will be analyzed in smaller temporal chunks. For each chunk (corresponding to a video frame or a short time segment), the `N` (e.g., 10, or a configurable `display_top_k`) most probable sound events *within that specific time window* will be identified.

### 3.2. Dynamic Y-axis Labels

The Y-axis labels of the eventogram in the MP4 will no longer be fixed. For each generated video frame, the labels will dynamically update to show the `N` most prominent events in that particular time slice. This will allow the eventogram to reflect the shifting soundscape accurately.

### 3.3. Dynamic Eventogram Data

The actual colored bars representing event probabilities will also be dynamically generated for each frame, displaying only the probabilities of the `N` currently selected events.

### 3.4. Visual Transitions and Stability

*   **Smoothness:** To prevent jarring visual changes, mechanisms for smooth transitions will be considered. This could involve interpolation or fading effects for events as they enter or leave the dynamic "top N" list. For instance, an event could gradually fade in as its probability rises and fade out as it drops below the threshold or is replaced by a more prominent event.
*   **Label Stability:** To ensure readability, strategies to maintain some level of label stability will be explored. This might involve a small temporal buffer (look-ahead/look-back) to prevent rapid flickering of labels, ensuring that only significant changes in event prominence trigger label updates.

### 3.5. Spectrogram

The spectrogram component of the visualization can largely remain as is, or be generated for the current time window. The primary focus of dynamic updates will be the eventogram section.

## 4. Implications for PNG (Static Visualization)

Since a static PNG image cannot inherently display dynamic content, it will likely retain its current role as a summary visualization. It can continue to show the overall top 10 (or a slightly increased fixed number) events across the entire duration. This provides a useful, albeit non-dynamic, overview of the most dominant sound events throughout the file.

## 5. Existing Chunking Necessity

It is important to note that for very large audio/video files (e.g., exceeding 30 minutes), the current processing pipeline already faces Out-of-Memory (OOM) issues, necessitating chunking of the input media via tools like `ffmpeg`. The proposed dynamic visualization approach naturally aligns with this existing requirement, as it inherently involves processing and rendering the media in smaller, manageable temporal segments. This synergy can simplify the overall implementation by leveraging a chunked processing workflow for both inference and visualization generation.
