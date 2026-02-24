import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import moviepy
from moviepy import ImageClip, CompositeVideoClip, AudioFileClip, ColorClip, VideoClip
import subprocess

# Add utils directory to sys.path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../utils'))
import config

def visualize_deltas():
    matched_csv = 'result/ducks_audioset_tagging_cnn_matched/full_event_log.csv'
    mismatched_csv = 'result/ducks_audioset_tagging_cnn_mismatched/full_event_log.csv'
    video_path = '/home/zezen/Downloads/GitHub/audioset_tagging_cnn/result/ducks.mp4'
    output_dir = 'result/ducks_delta_analysis'
    os.makedirs(output_dir, exist_ok=True)

    print("Loading labels...")
    labels = config.labels
    label_to_ix = {label: i for i, label in enumerate(labels)}
    
    print("Reading CSVs (this may take a few seconds)...")
    df_matched = pd.read_csv(matched_csv)
    df_mismatched = pd.read_csv(mismatched_csv)

    # Determine dimensions
    times = sorted(df_matched['time'].unique())
    frames_num = len(times)
    classes_num = len(labels)
    fps = 100 # Default hop_size/sample_rate logic

    print(f"Reconstructing matrices ({frames_num} frames x {classes_num} classes)...")
    
    def reconstruct_matrix(df):
        # Pivot is faster than manual loops
        mat = df.pivot(index='time', columns='sound', values='probability')
        # Reorder columns to match config.labels
        mat = mat.reindex(columns=labels, fill_value=0)
        return mat.values

    mat_matched = reconstruct_matrix(df_matched)
    mat_mismatched = reconstruct_matrix(df_mismatched)

    # Calculate Delta
    delta_mat = mat_mismatched - mat_matched
    
    # Finding top 10 sounds with largest ABSOLUTE disagreement for the visualization
    abs_delta = np.abs(delta_mat)
    max_disagreement = np.max(abs_delta, axis=0)
    sorted_indexes = np.argsort(max_disagreement)[::-1]
    top_k = 10
    top_result_mat = delta_mat[:, sorted_indexes[0:top_k]]
    top_labels = np.array(labels)[sorted_indexes[0:top_k]]

    # Static PNG visualization
    print("Creating Delta Eventogram image...")
    fig_width_px = 1280
    fig_height_px = 480
    dpi = 100
    
    # Custom plotting logic
    fig = plt.figure(figsize=(fig_width_px / dpi, fig_height_px / dpi), dpi=dpi)
    # Using a single plot for the delta heatmap
    ax = fig.add_subplot(111)
    
    # Use 'bwr' (Blue-White-Red) colormap. 
    # Red = Mismatched higher, Blue = Matched higher, White = Neutral
    im = ax.matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='bwr', vmin=-1, vmax=1)
    
    ax.set_yticks(np.arange(0, top_k))
    ax.set_yticklabels(top_labels, fontsize=12)
    ax.set_xlabel('Seconds', fontsize=14)
    ax.set_title('Model Disagreement Delta (Red: Mismatched higher | Blue: Matched higher)', fontsize=14)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Probability Delta')
    cbar.set_ticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['-1.00', '-0.75', '-0.50', '-0.25', '0.00', '0.25', '0.50', '0.75', '1.00'])
    
    # Custom X-axis ticks for better readability and alignment
    duration = times[-1]
    # Ensure tick_interval is at least 1, and scale with duration
    tick_interval = max(1.0, round(duration / 10.0, 1)) 
    x_ticks_data_coords = np.arange(0, frames_num + 1, fps * tick_interval) # Data coordinates for ticks
    x_labels = [f"{times[int(t_idx)]:.1f}" if t_idx < frames_num else '' for t_idx in x_ticks_data_coords]
    
    ax.xaxis.set_ticks(x_ticks_data_coords)
    ax.xaxis.set_ticklabels(x_labels, rotation=45, ha='right')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlim(-0.5, frames_num - 0.5) # Adjust xlim to match matshow extent


    fig_path = os.path.join(output_dir, 'delta_eventogram.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    print(f"Saved delta visualization to {fig_path}")

    # Output Delta to CSV
    print("Writing delta event log to CSV...")
    delta_csv_path = os.path.join(output_dir, 'delta_event_log.csv')
    with open(delta_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'sound', 'probability_delta'])
        for time_idx in range(frames_num):
            timestamp = times[time_idx]
            for label_idx in range(len(labels)):
                sound = labels[label_idx]
                delta_val = delta_mat[time_idx, label_idx]
                # Only write non-zero deltas for readability, or significant ones
                if abs(delta_val) > 1e-6: # Filter out very small floating point noise
                    writer.writerow([round(timestamp, 3), sound, float(delta_val)])
    print(f"Saved delta event log to {delta_csv_path}")


    # Video rendering logic (Simplified)
    print("Rendering delta overlay video...")
    output_video_path = os.path.join(output_dir, 'ducks_delta_overlay.mp4')
    
    # Re-create figure to get correct bbox for marker alignment
    fig = plt.figure(figsize=(fig_width_px / dpi, fig_height_px / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    
    # Plotting again (invisible) just to get the bbox right for the marker
    ax.matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='bwr', vmin=-1, vmax=1)
    ax.set_xlim(-0.5, frames_num - 0.5) # Ensure consistent xlim
    
    # Remove all labels and ticks to get a clean bbox for the plot area itself.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    
    # Important: Adjust subplot parameters to remove all padding, get tight plot area
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0) 
    
    fig.canvas.draw()
    
    # Get the bounding box of the axes in figure coordinates (0-1 range)
    bbox_axes_fig_coords = ax.get_position()

    # Get the overall figure size in pixels
    fig_width_pixels, fig_height_pixels = fig.get_size_inches() * fig.dpi

    # Calculate the plot area's pixel bounds
    plot_left_pixel = bbox_axes_fig_coords.x0 * fig_width_pixels
    plot_right_pixel = bbox_axes_fig_coords.x1 * fig_width_pixels
    plot_width_pixel = plot_right_pixel - plot_left_pixel
    
    plt.close(fig) # Close the figure, it's just for bbox now.


    static_clip = ImageClip(fig_path, duration=duration)

    def make_frame_with_marker(t):
        img_array = static_clip.get_frame(t)
        
        # Calculate X position in pixels relative to the image
        # Map time to a frame index (data coordinate)
        frame_idx = t * fps
        
        # Scale frame_idx (0 to frames_num) to pixel position within the plot area
        # Ensure that frame_idx is within the range of x_data_min to x_data_max, if set
        x_data_min, x_data_max = -0.5, frames_num - 0.5 # These are the xlims for matshow
        
        # Normalize frame_idx within the data range and then scale to pixel width
        x_pos_pixels_relative_to_plot = ((frame_idx - x_data_min) / (x_data_max - x_data_min)) * plot_width_pixel
        
        # Add the offset for the left margin of the plot
        x_pos_pixels = int(plot_left_pixel + x_pos_pixels_relative_to_plot)
        
        # Draw a vertical green line (marker) on the frame
        marker_color = [0, 255, 0] # Green
        marker_width = 3
        
        # Ensure x_pos_pixels is within bounds
        x_pos_pixels = max(0, min(x_pos_pixels, static_clip.w - marker_width -1 )) # -1 to prevent out of bounds
        
        # Apply marker to the full height of the image
        img_array[:, x_pos_pixels:x_pos_pixels + marker_width, :] = marker_color
        return img_array

    final_video = VideoClip(make_frame_with_marker, duration=duration)
    audio_clip = AudioFileClip(video_path)
    final_video = final_video.with_audio(audio_clip)
    final_video.fps = 24
    
    final_video.write_videofile(output_video_path, codec="libx264")
    print(f"FINISHED: {output_video_path}")

if __name__ == '__main__':
    visualize_deltas()
