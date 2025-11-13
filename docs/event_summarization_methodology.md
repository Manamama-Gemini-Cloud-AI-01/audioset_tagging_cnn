# Methodology for Improved Event Summarization

**Version:** 1.0
**Date:** 2025-11-13

## 1. The Problem

The initial summarization logic for `summary_events.csv` was based on a simple metric (`duration * average_probability`) and a fixed limit of 40 events. This caused short-duration but high-intensity events (e.g., a "Gunshot", "Dog bark") to be frequently missed, as they were out-scored by long, low-intensity ambient sounds (e.g., "Wind", "Traffic"). The goal is to create a more specific, useful, and statistically sound summary.

---

## 2. Version 1 (Implemented): Top-N Events Per Class

This was the first iteration of improvement, which has already been implemented in `pytorch/audioset_tagging_cnn_inference_6.py`.

### 2.1. Methodology

Instead of creating a single global list of all detected events and sorting it, this method first groups events by their sound class. It then selects the top 3 most significant events from each class, based on `peak_probability`. These top events are then combined and sorted chronologically.

This ensures that at least the most intense occurrences of every detected sound type are represented in the final summary, preventing them from being drowned out.

### 2.2. Implemented Code

```python
# --- AI-Friendly Summary Generation (Event Block Detection) ---
print("📊  Generating AI-friendly event summary files…")

# 1. Generate summary_events.csv
summary_csv_path = os.path.join(output_dir, 'summary_events.csv')

onset_threshold = 0.20  # Probability to start an event
offset_threshold = 0.15 # Probability to end an event
min_event_duration_seconds = 0.5 # Minimum duration to be considered a valid event
top_n_per_class = 3 # Get the top N most intense events for each sound class

events_by_class = {label: [] for label in labels}

# Find event blocks for each sound class
for i, label in enumerate(labels):
    in_event = False
    event_start_frame = 0
    
    for frame_index, prob in enumerate(framewise_output[:, i]):
        if not in_event and prob > onset_threshold:
            in_event = True
            event_start_frame = frame_index
        elif in_event and prob < offset_threshold:
            in_event = False
            event_end_frame = frame_index
            
            duration_frames = event_end_frame - event_start_frame
            duration_seconds = duration_frames / frames_per_second
            
            if duration_seconds >= min_event_duration_seconds:
                event_block_probs = framewise_output[event_start_frame:event_end_frame, i]
                
                events_by_class[label].append({
                    'sound_class': label,
                    'start_time_seconds': round(event_start_frame / frames_per_second, 3),
                    'end_time_seconds': round(event_end_frame / frames_per_second, 3),
                    'duration_seconds': round(duration_seconds, 3),
                    'peak_probability': float(np.max(event_block_probs)),
                    'average_probability': float(np.mean(event_block_probs))
                })

# Select the top N events from each class based on peak probability, then combine and sort chronologically
top_events = []
for label, events in events_by_class.items():
    if events:
        # Sort events within the class by peak_probability
        sorted_class_events = sorted(events, key=lambda x: x['peak_probability'], reverse=True)
        # Add the top N events to the final list
        top_events.extend(sorted_class_events[:top_n_per_class])

# Sort the final combined list of top events by their start time
final_sorted_events = sorted(top_events, key=lambda x: x['start_time_seconds'])

with open(summary_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['sound_class', 'start_time_seconds', 'end_time_seconds', 'duration_seconds', 'peak_probability', 'average_probability']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(final_sorted_events)
print(f'Saved summary events CSV to: \033[1;34m{summary_csv_path}\033[1;0m')
```

---

## 3. Version 2 (Proposed): Event Detection via Peak Prominence

This is a more advanced and statistically robust approach grounded in established signal processing theory.

### 3.1. Methodology

This method treats each sound class's probability stream as a time-series signal and uses a dedicated scientific library (`SciPy`) to identify significant events.

1.  **Identify Peaks:** For each sound class, use the `scipy.signal.find_peaks` function.
2.  **Use Prominence:** The key parameter is `prominence`, which measures how much a peak "stands out" from the surrounding signal. This is a highly effective, context-aware way to filter for only statistically significant events, replacing the need for manual derivative calculations or fixed thresholds.
3.  **Extract Boundaries:** The `find_peaks` function can directly return the start and end boundaries of each detected peak, providing a precise event duration.
4.  **Generate Summary:** The events (peaks) identified this way are collected, characterized (with peak probability, duration, etc.), and sorted chronologically.

This method is superior as it detects events based on their dynamic significance rather than their absolute probability values.

---

## 4. Alternative Idea: K-Means Clustering

K-means is a powerful unsupervised classification algorithm, but it solves a different problem than event detection.

*   **Primary Use Case:** K-means could be used to **segment** the audio into different "soundscapes" (e.g., cluster 1 is "traffic", cluster 2 is "quiet room"). Transitions between these clusters would signify a change in the overall audio scene.
*   **Complementary Use Case:** A more powerful application would be to use K-means as a **secondary analysis step**. After detecting individual events with the `find_peaks` method, K-means could be used to **cluster the events themselves** based on their features (duration, intensity, etc.). This could automatically group events into categories like "short impulse sounds," "sustained ambient sounds," and "vocalizations," adding another layer of abstraction and insight to the summary.
