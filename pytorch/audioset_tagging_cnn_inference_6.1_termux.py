#!/usr/bin/env python3
# See: origin https://github.com/qiuqiangkong/audioset_tagging_cnn, heavily modified.
# Uses two .py files that should be in the same folder for imports: pytorch_utils.py and models.py.

# AI EXECUTION CONTEXT WARNING:
# This script is designed to be run from the PROJECT ROOT directory
# (the parent directory of this `pytorch` folder).
#
# MANDATORY ACTION: The Current Working Directory (CWD) MUST be the project root.
# This is because imported modules (like `config.py`) use relative paths
# (e.g., 'metadata/class_labels_indices.csv') that are resolved from the CWD.
#
# Correct execution example:
# `cd /path/to/audioset_tagging_cnn/`
# `python3 pytorch/audioset_tagging_cnn_inference_6.py /path/to/video.mp4 --checkpoint_path ...`
#
# Note: Paths provided as command-line arguments should be absolute to avoid ambiguity.



import os
import sys
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Torch and torchaudio must be ==2.8 for now, avoid for a while: python -m pip install torch torchaudio--upgrade --extra-index-url https://download.pytorch.org/whl/cpu

import torch
import torchaudio
import csv
import datetime
import time
import subprocess
import shutil
import moviepy
import warnings
import platform
import soundfile as sf

from moviepy import ImageClip, CompositeVideoClip, AudioFileClip, ColorClip, VideoClip
import json
from scipy.stats import entropy

# Suppress torchaudio deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# Add utils directory to sys.path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../utils'))
from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config

def is_video_file(audio_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_streams', '-print_format', 'json', audio_path],
            capture_output=True, text=True, check=True
        )
        streams = json.loads(result.stdout).get('streams', [])
        return any(stream['codec_type'] == 'video' and stream.get('codec_name') not in ['mjpeg', 'png'] for stream in streams)
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False


def get_duration_and_fps(input_media_path):
    """
    Extract duration, FPS, and resolution from a media file using FFprobe.
    Handles audio-only and video files universally, prioritizing format duration.
    """
    try:
        # Run ffprobe to get format and stream info
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-print_format', 'json', input_media_path],
            capture_output=True, text=True
        )
        # Check if ffprobe returned valid JSON output
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            print(f"\033[1;31mError: Failed to parse FFprobe JSON output for {input_media_path}: {e}\033[0m")
            data = {}

        streams = data.get('streams', [])
        format_info = data.get('format', {})
        
        # Initialize outputs
        duration = None
        fps = None
        width = None
        height = None

        # Try format duration first (most reliable)
        if format_info.get('duration'):
            try:
                duration = float(format_info['duration'])
            except (ValueError, TypeError):
                print(f"\033[1;33mWarning: Invalid format duration in {input_media_path}\033[0m")

        # Check video stream for duration, FPS, and resolution
        video_stream = next((s for s in streams if s['codec_type'] == 'video'), None)
        if video_stream:
            if duration is None and video_stream.get('duration'):
                try:
                    duration = float(video_stream['duration'])
                except (ValueError, TypeError):
                    print(f"\033[1;33mWarning: Invalid video stream duration in {input_media_path}\033[0m")
            
            avg_frame_rate = video_stream.get('avg_frame_rate')
            if avg_frame_rate and '/' in avg_frame_rate:
                try:
                    num, den = map(int, avg_frame_rate.split('/'))
                    fps = num / den if den else None
                except (ValueError, TypeError):
                    print(f"\033[1;33mWarning: Invalid avg_frame_rate in {input_media_path}\033[0m")
            
            width = int(video_stream.get('width', 0)) if video_stream.get('width') else None
            height = int(video_stream.get('height', 0)) if video_stream.get('height') else None
            
            if duration is None and video_stream.get('nb_frames') and fps:
                try:
                    duration = int(video_stream['nb_frames']) / fps
                except (ValueError, TypeError):
                    print(f"\033[1;33mWarning: Invalid nb_frames or fps for duration calculation in {input_media_path}\033[0m")

        # Check audio stream for duration (for audio-only files like WebM/Opus)
        if duration is None:
            audio_stream = next((s for s in streams if s['codec_type'] == 'audio'), None)
            if audio_stream and audio_stream.get('duration'):
                try:
                    duration = float(audio_stream['duration'])
                except (ValueError, TypeError):
                    print(f"\033[1;33mWarning: Invalid audio stream duration in {input_media_path}\033[0m")
            
            # Check tags.DURATION for audio stream (e.g., WebM/Opus)
            if duration is None and audio_stream and audio_stream.get('tags', {}).get('DURATION'):
                try:
                    # Parse tags.DURATION (format: "HH:MM:SS.mmmmmmmmm")
                    duration_str = audio_stream['tags']['DURATION']
                    h, m, s = duration_str.split(':')
                    s, ms = s.split('.')
                    duration = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000000000
                except (ValueError, TypeError):
                    print(f"\033[1;33mWarning: Invalid tags.DURATION in {input_media_path}\033[0m")

        # Fallback: Direct ffprobe duration probe
        if duration is None:
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_media_path],
                    capture_output=True, text=True
                )
                duration = float(result.stdout.strip())
            except (subprocess.CalledProcessError, ValueError, TypeError) as e:
                print(f"\033[1;33mWarning: Fallback duration probe failed for {input_media_path}: {e}\033[0m")

        # If duration is still None, exit with error
        if duration is None:
            print(f"\033[1;31mError: Could not determine duration for {input_media_path}. Exiting.\033[0m")
            return None, None, None, None

        # Log results
        duration_str = str(datetime.timedelta(seconds=int(duration))) if duration else "?"
        print(f"⏲  🗃️  Input file duration: \033[1;34m{duration_str}\033[0m")
        if fps:
            print(f"🮲  🗃️  Input video FPS (avg): \033[1;34m{fps:.3f}\033[0m")
        if width and height:
            print(f"📽  🗃️  Input video resolution: \033[1;34m{width}x{height}\033[0m")
            if height > width:
                print("\n\033[1;33m--- WARNING: Portrait Video Detected ---\033[0m")
                print("\nProcessing a portrait video may result in a narrow or distorted overlay.")
                print("For best results, it is recommended to rotate the video to landscape before processing.")
                print("\nYou can use the following ffmpeg command to rotate the video:\n")
                print(f"\033[1;32mffmpeg -i \"{input_media_path}\" -vf \"transpose=1\" \"{os.path.splitext(input_media_path)[0]}_rotated.mp4\"\033[0m\n")
                print("To do so, break the script now (Ctrl+C) and run the command above.")
                print("Otherwise, we are proceeding with the portrait video in 2 seconds...")
                time.sleep(2) # Add a 2-second delay
                print("\033[1;33m------------------------------------------\033[0m\n")

        return duration, fps, width, height

    except subprocess.CalledProcessError as e:
        print(f"\033[1;31mError: FFprobe failed for {input_media_path}: {e}\033[0m")
        return None, None, None, None
    except Exception as e:
        print(f"\033[1;31mError: Unexpected failure parsing {input_media_path}: {e}\033[0m")
        return None, None, None, None

def compute_kl_divergence(p, q, eps=1e-10):
    """Compute KL divergence between two probability distributions."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))

def get_dynamic_top_events(framewise_output, start_idx, end_idx, top_k=10):
    """Get top k events for a given window of framewise_output."""
    window_output = framewise_output[start_idx:end_idx]
    if window_output.shape[0] == 0:
        return np.array([]), np.array([])
    max_probs = np.max(window_output, axis=0)
    sorted_indexes = np.argsort(max_probs)[::-1][:top_k]
    return window_output[:, sorted_indexes], sorted_indexes

def audio_tagging(args):
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels

    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                  classes_num=classes_num)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    except Exception as e:
        print(f"\033[1;31mError loading model checkpoint: {e}\033[0m")
        return

    if device.type == 'cuda':
        model.to(device)
        print(f'GPU number: {torch.cuda.device_count()}')
        model = torch.nn.DataParallel(model)

    data, sr = sf.read(audio_path, dtype='float32')
    waveform = torch.from_numpy(data).T
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    print(f"Loaded waveform shape: {waveform.shape}, sample rate: {sr}")
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
    waveform = waveform[None, :]  # Shape: [1, samples] for model input
    waveform = move_data_to_device(waveform, device)

    with torch.no_grad():
        model.eval()
        try:
            batch_output_dict = model(waveform, None)
        except Exception as e:
            print(f"\033[1;31mError during model inference: {e}\033[0m")
            return

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    print('Sound events detection result (time_steps x classes_num): {}'.format(clipwise_output.shape))

    for k in range(10):
        print('{}: {}'.format(np.array(labels)[sorted_indexes[k]], clipwise_output[sorted_indexes[k]]))

    if 'embedding' in batch_output_dict:
        embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        print('embedding: {}'.format(embedding.shape))

    return clipwise_output, labels

def sound_event_detection(args):
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    else:
        print('Using CPU.')

    classes_num = config.classes_num
    labels = config.labels
    
    audio_dir = os.path.dirname(audio_path)
    base_filename_for_dir = get_filename(audio_path)
    output_dir = os.path.join(audio_dir, f'{base_filename_for_dir}_audioset_tagging_cnn')
    create_folder(output_dir)

    # --- Idempotency Check ---
    manifest_path = os.path.join(output_dir, 'summary_manifest.json')
    if os.path.exists(manifest_path):
        print(f"✅ Skipping {audio_path} - summary_manifest.json already exists.")
        return

    base_filename = f'{base_filename_for_dir}_audioset_tagging_cnn'
    fig_path = os.path.join(output_dir, 'eventogram.png')
    csv_path = os.path.join(output_dir, 'full_event_log.csv')
    if args.dynamic_eventogram:
        output_video_path = os.path.join(output_dir, f'{base_filename}_eventogram_dynamic.mp4')
    else:
        output_video_path = os.path.join(output_dir, f'{base_filename}_eventogram.mp4')
    
    disk_usage = shutil.disk_usage(audio_dir)
    if disk_usage.free < 1e9:
        print(f"\033[1;31mError: Insufficient disk space ({disk_usage.free / 1e9:.2f} GB free). Exiting.\033[0m")
        return
    
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                  classes_num=classes_num)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    except Exception as e:
        print(f"\033[1;31mError loading model checkpoint: {e}\033[0m")
        return

    duration, video_fps, video_width, video_height = get_duration_and_fps(audio_path)
    if duration is None:
        print("\033[1;31mError: Could not determine audio duration. Exiting.\033[0m")
        return
    is_video = is_video_file(audio_path)
    
    if is_video and (video_width is None or video_height is None):
        video_width = 1280
        video_height = 720
        print(f"\033[1;33mWarning: Video dimensions not detected, using default {video_width}x{video_height}.\033[0m")
    
    video_input_path = audio_path
    temp_video_path = None
    if is_video:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate,avg_frame_rate', '-of', 'json', audio_path],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        if data.get('streams'):
            stream = data['streams'][0]
            r_frame_rate = stream.get('r_frame_rate')
            avg_frame_rate = stream.get('avg_frame_rate')
            if r_frame_rate and avg_frame_rate:
                r_num, r_den = map(int, r_frame_rate.split('/'))
                avg_num, avg_den = map(int, avg_frame_rate.split('/'))
                r_fps = r_num / r_den if r_den else 0
                avg_fps = avg_num / avg_den if avg_den else 0
                if abs(r_fps - avg_fps) > 0.01:
                    print("\033[1;33mDetected VFR video (r_frame_rate={r_fps:.3f}, avg_frame_rate={avg_fps:.3f}). Re-encoding to CFR.\033[0m")
                    temp_video_path = os.path.join(audio_dir, f'temp_cfr_{get_filename(audio_path)}.mp4')
                    try:
                        target_fps = video_fps
                        if not target_fps or target_fps <= 0:
                            if r_fps > 0:
                                target_fps = r_fps
                                print(f"\033[1;33mWarning: avg_frame_rate is invalid, falling back to r_frame_rate: {target_fps:.3f}\033[0m")
                            else:
                                target_fps = 30
                                print(f"\033[1;33mWarning: Both avg_frame_rate and r_frame_rate are invalid, falling back to default FPS: {target_fps}\033[0m")
                        
                        subprocess.run([
                            'ffmpeg', '-loglevel', 'warning', '-i', audio_path, '-r', str(target_fps), '-fps_mode', 'cfr', '-c:a', 'copy', temp_video_path, '-y'
                        ], check=True)
                        video_input_path = temp_video_path
                        print(f"Re-encoded to: \033[1;34m{temp_video_path}\033[1;0m")
                    except subprocess.CalledProcessError as e:
                        print(f"\033[1;31mError during VFR-to-CFR conversion: {e}\033[0m")
                        return

    data, sr = sf.read(video_input_path, dtype='float32')
    waveform = torch.from_numpy(data).T
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    print(f"Loaded waveform shape: {waveform.shape}, sample rate: {sr}")
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
    waveform = waveform.squeeze(0).numpy()  # Convert to numpy for STFT
    print(f"Processed waveform shape: {waveform.shape}")

    chunk_duration = 180  # 3 minutes
    chunk_samples = int(chunk_duration * sample_rate)
    framewise_outputs = []
    
    for start in range(0, len(waveform), chunk_samples):
        chunk = waveform[start:start + chunk_samples]
        if len(chunk) < sample_rate // 10:
            print(f"Skipping small chunk at start={start}, len={len(chunk)}")
            continue
        chunk = chunk[None, :]  # Shape: [1, samples]
        chunk = move_data_to_device(torch.from_numpy(chunk).float(), device)
        print(f"Processing chunk: start={start}, len={len(chunk)}")
        
        with torch.no_grad():
            model.eval()
            try:
                batch_output_dict = model(chunk, None)
                framewise_output_chunk = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
                print(f"Chunk output shape: {framewise_output_chunk.shape}")
                framewise_outputs.append(framewise_output_chunk)
            except Exception as e:
                print(f"\033[1;31mError processing chunk at start={start}: {e}\033[0m")
                continue
    
    if not framewise_outputs:
        print("\033[1;31mError: No valid chunks processed. Cannot generate eventogram.\033[0m")
        return
    
    framewise_output = np.concatenate(framewise_outputs, axis=0)
    print(f'Sound event detection result (time_steps x classes_num): \033[1;34m{framewise_output.shape}\033[1;0m')

    frames_per_second = sample_rate // hop_size
    waveform_tensor = torch.from_numpy(waveform).to(device)
    stft = torch.stft(
        waveform_tensor,
        n_fft=window_size,
        hop_length=hop_size,
        window=torch.hann_window(window_size).to(device),
        center=True,
        return_complex=True
    )
    stft = stft.abs().cpu().numpy()
    frames_num = int(duration * frames_per_second)
    
    if framewise_output.shape[0] < frames_num:
        pad_width = frames_num - framewise_output.shape[0]
        framewise_output = np.pad(framewise_output, ((0, pad_width), (0, 0)), mode='constant')
    
    # Static PNG visualization
    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]
    top_k = 10
    top_result_mat = framewise_output[:frames_num, sorted_indexes[0:top_k]]

    fig_width_px = 1280
    fig_height_px = 480
    dpi = 100
    fig = plt.figure(figsize=(fig_width_px / dpi, fig_height_px / dpi), dpi=dpi)

    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], left=0.0, right=1.0, top=0.95, bottom=0.08, hspace=0.05)
    axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]

    axs[0].matshow(np.log(stft + 1e-10), origin='lower', aspect='auto', cmap='jet')
    axs[0].set_ylabel('Frequency bins', fontsize=14)
    axs[0].set_title('Spectrogram and Eventogram', fontsize=14)
    axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)

    tick_interval = max(5, int(duration / 20))
    x_ticks = np.arange(0, frames_num + 1, frames_per_second * tick_interval)
    x_labels = np.arange(0, int(duration) + 1, tick_interval)
    axs[1].xaxis.set_ticks(x_ticks)
    axs[1].xaxis.set_ticklabels(x_labels[:len(x_ticks)], rotation=45, ha='right', fontsize=10)
    axs[1].set_xlim(0, frames_num)
    top_labels = np.array(labels)[sorted_indexes[0:top_k]]
    axs[1].set_yticks(np.arange(0, top_k))
    axs[1].set_yticklabels(top_labels, fontsize=14)
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds', fontsize=14)
    axs[1].xaxis.set_ticks_position('bottom')
    
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_label_width_px = 0
    for lbl in axs[1].yaxis.get_majorticklabels():
        bbox = lbl.get_window_extent(renderer=renderer)
        w = bbox.width
        if w > max_label_width_px:
            max_label_width_px = w

    pad_px = 8
    left_margin_px = int(max_label_width_px + pad_px + 6)
    fig_w_in = fig.get_size_inches()[0]
    fig_w_px = fig_w_in * dpi
    left_frac = left_margin_px / fig_w_px
    if left_frac < 0:
        left_frac = 0.0
    if left_frac > 0.45:
        left_frac = 0.45

    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], left=left_frac, right=1.0, top=0.95, bottom=0.08, hspace=0.05)
    fig.clear()
    axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]

    axs[0].matshow(np.log(stft + 1e-10), origin='lower', aspect='auto', cmap='jet')
    axs[0].set_ylabel('Frequency bins', fontsize=14)
    axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    axs[1].xaxis.set_ticks(x_ticks)
    axs[1].xaxis.set_ticklabels(x_labels[:len(x_ticks)], rotation=45, ha='right', fontsize=10)
    axs[1].set_xlim(0, frames_num)
    axs[1].yaxis.set_ticks(np.arange(0, top_k))
    axs[1].yaxis.set_ticklabels(top_labels, fontsize=14)
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds', fontsize=14)
    axs[1].xaxis.set_ticks_position('bottom')
    
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f'Saved sound event detection visualization to: \033[1;34m{fig_path}\033[1;0m')
    print(f'Computed left margin (px): \033[1;34m{left_margin_px}\033[1;00m, axes bbox (fig-fraction): \033[1;34m{axs[1].get_position()}\033[1;00m')

    with open(csv_path, 'w', newline='') as csvfile:
        threshold = 0.2
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'sound', 'probability'])
        for i in range(frames_num):
            timestamp = i / frames_per_second
            for j, label in enumerate(labels):
                prob = framewise_output[i, j]
                if prob > threshold:
                    writer.writerow([round(timestamp, 3), label, float(prob)])
    print(f'Saved full framewise CSV to: \033[1;34m{csv_path}\033[1;0m')

    # Video rendering
    fps = video_fps if video_fps else 24
    if args.dynamic_eventogram:
        print(f"🎞  Rendering dynamic eventogram video …")
        window_duration = args.window_duration
        window_frames = int(window_duration * frames_per_second)
        half_window_frames = window_frames // 2

        # Precompute unique window frames to improve performance
        frame_times = np.arange(0, duration, 1/fps)
        unique_windows = {}
        for t in frame_times:
            current_frame = int(t * frames_per_second)
            start_frame = max(0, current_frame - half_window_frames)
            end_frame = min(frames_num, current_frame + half_window_frames)
            window_key = (start_frame, end_frame)
            if window_key not in unique_windows:
                window_output, window_indexes = get_dynamic_top_events(framewise_output, start_frame, end_frame, top_k)
                if window_output.size == 0:
                    window_output = np.zeros((end_frame - start_frame, top_k))
                    window_indexes = sorted_indexes[:top_k]
                unique_windows[window_key] = (window_output, window_indexes)

        def make_frame(t):
            current_frame = int(t * frames_per_second)
            start_frame = max(0, current_frame - half_window_frames)
            end_frame = min(frames_num, current_frame + half_window_frames)
            
            # Adaptive window size (if enabled)
            if args.use_adaptive_window:
                kl_threshold = 0.5
                for offset in range(half_window_frames, half_window_frames + int(30 * frames_per_second), int(frames_per_second)):
                    if start_frame - offset >= 0:
                        prev_prob = np.mean(framewise_output[start_frame - offset:start_frame], axis=0)
                        curr_prob = np.mean(framewise_output[start_frame:start_frame + offset], axis=0)
                        kl_div = compute_kl_divergence(prev_prob, curr_prob)
                        if kl_div > kl_threshold:
                            start_frame = max(0, start_frame - offset // 2)
                            break
                    if end_frame + offset < frames_num:
                        curr_prob = np.mean(framewise_output[end_frame - offset:end_frame], axis=0)
                        next_prob = np.mean(framewise_output[end_frame:end_frame + offset], axis=0)
                        kl_div = compute_kl_divergence(curr_prob, next_prob)
                        if kl_div > kl_threshold:
                            end_frame = min(frames_num, end_frame + offset // 2)
                            break
            
            window_output, window_indexes = unique_windows.get((start_frame, end_frame), (np.zeros((end_frame - start_frame, top_k)), sorted_indexes[:top_k]))
            
            # Create frame
            fig = plt.figure(figsize=(fig_width_px / dpi, fig_height_px / dpi), dpi=dpi)
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], left=left_frac, right=1.0, top=0.95, bottom=0.08, hspace=0.05)
            axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]

            # Spectrogram for window
            stft_window = stft[:, start_frame:end_frame]
            axs[0].matshow(np.log(stft_window + 1e-10), origin='lower', aspect='auto', cmap='jet')
            axs[0].set_ylabel('Frequency bins', fontsize=14)
            axs[0].set_title(f'Spectrogram and Eventogram (t={t:.1f}s)', fontsize=14)
            print(f'; (t={t:.1f}s)')
            # Eventogram for window
            axs[1].matshow(window_output.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
            window_labels = np.array(labels)[window_indexes]
            axs[1].yaxis.set_ticks(np.arange(0, top_k))
            axs[1].yaxis.set_ticklabels(window_labels, fontsize=14)
            axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
            axs[1].set_xlabel('Seconds', fontsize=14)
            axs[1].xaxis.set_ticks_position('bottom')

            # Adjust x-axis ticks for both plots
            window_seconds = (end_frame - start_frame) / frames_per_second
            tick_interval_window = max(1, int(window_seconds / 5))
            x_ticks_window = np.arange(0, end_frame - start_frame + 1, frames_per_second * tick_interval_window)
            x_labels_window = np.arange(int(start_frame / frames_per_second), int(end_frame / frames_per_second) + 1, tick_interval_window)
            
            for ax in axs:
                ax.xaxis.set_ticks(x_ticks_window)
                ax.xaxis.set_ticklabels(x_labels_window[:len(x_ticks_window)], rotation=45, ha='right', fontsize=10)
                ax.set_xlim(0, end_frame - start_frame)

            # Add marker
            marker_x = current_frame - start_frame
            for ax in axs:
                ax.axvline(x=marker_x, color='red', linewidth=2, alpha=0.8)

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape((fig_height_px, fig_width_px, 4))[:, :, :3]  # Drop alpha channel
            plt.close(fig)
            return img

        # Generate dynamic video
        eventogram_clip = VideoClip(make_frame, duration=duration)
        audio_clip = AudioFileClip(video_input_path)
        eventogram_with_audio_clip = eventogram_clip.with_audio(audio_clip)
        eventogram_with_audio_clip.fps = fps

        eventogram_with_audio_clip.write_videofile(
            output_video_path,
            codec="libx264",
            fps=fps,
            threads=os.cpu_count()
        )
        print(f"🎹 Saved the dynamic eventogram video to: \033[1;34m{output_video_path}\033[1;0m")
    else:
        print(f"🎞  Rendering static eventogram video …")
        static_eventogram_clip = ImageClip(fig_path, duration=duration)

        def marker_position(t):
            w = static_eventogram_clip.w
            x_start = int(left_frac * w)
            x_end = w
            frac = np.clip(t / max(duration, 1e-8), 0.0, 1.0)
            x_pos = x_start + (x_end - x_start) * frac
            return (x_pos, 0)

        marker = ColorClip(size=(2, static_eventogram_clip.h), color=(255, 0, 0)).with_duration(duration)
        marker = marker.with_position(marker_position)
        eventogram_visual_clip = CompositeVideoClip([static_eventogram_clip, marker])
        audio_clip = AudioFileClip(video_input_path)
        eventogram_with_audio_clip = eventogram_visual_clip.with_audio(audio_clip)
        eventogram_with_audio_clip.fps = fps

        eventogram_with_audio_clip.write_videofile(
            output_video_path,
            codec="libx264",
            fps=fps,
            threads=os.cpu_count()
        )
        print(f"🎹 Saved the static eventogram video to: \033[1;34m{output_video_path}\033[1;0m")

    if is_video:
        print("🎬  Overlaying source media with the eventogram…")
        root, ext = os.path.splitext(output_video_path)
        final_output_path = f"{root}_overlay{ext}"
        print(f"🎬  Source resolution:")
        _, _, base_w, base_h = get_duration_and_fps(audio_path)
        print(f"💁  Overlay resolution:")
        _, _, ovr_w, ovr_h = get_duration_and_fps(output_video_path)
        if base_w >= ovr_w:
            target_width, target_height = base_w, base_h
        else:
            target_width = ovr_w
            target_height = int(base_h * ovr_w / base_w)
            if target_height % 2:
                target_height += 1
        print(f"🎯 Target resolution: {target_width}x{target_height}")
        main_input, overlay_input = audio_path, output_video_path

        overlay_cmd = [
            "time", "-v",
            "ffmpeg", "-y",
            "-i", main_input,
            "-i", overlay_input,
            "-loglevel", "warning",
            "-filter_complex",
            (
                f"[0:v]scale={target_width}:{target_height}[main];"
                f"[1:v]scale={target_width}:{int(target_height * args.overlay_size)}[ovr];"
                f"[ovr]format=rgba,colorchannelmixer=aa={args.translucency}[ovr_t];"
                "[main][ovr_t]overlay=x=0:y=H-h[v]"
            ),
            "-map", "[v]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
        ]
        if args.bitrate:
            overlay_cmd.extend(["-b:v", args.bitrate])
        else:
            overlay_cmd.extend(["-crf", str(args.crf)])
        overlay_cmd.extend([
            "-c:a", "copy",
            "-shortest",
            final_output_path
        ])

        try:
            subprocess.run(overlay_cmd, check=True)
            print(f"✅ 🎥 The new overlaid video has been saved to: \033[1;34m{final_output_path}\033[1;0m")
        except subprocess.CalledProcessError as e:
            print(f"\033[1;31mError during FFmpeg overlay: {e}\033[0m")
            return
        
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                print(f"Deleted temporary CFR video: \033[1;34m{temp_video_path}\033[1;0m")
            except Exception as e:
                print(f"\033[1;33mWarning: Failed to delete temporary CFR video {temp_video_path}: {e}\033[0m")
    else:
        print("🎧 Source is audio-only — the eventogram video is the final output.")

    # --- AI-Friendly Summary Generation (Event Block Detection) ---
    print("📊  Generating AI-friendly event summary files…")

    # 1. Generate summary_events.csv
    summary_csv_path = os.path.join(output_dir, 'summary_events.csv')
    
    onset_threshold = 0.20  # Probability to start an event
    offset_threshold = 0.15 # Probability to end an event
    min_event_duration_seconds = 0.5 # Minimum duration to be considered a valid event

    detected_events = []

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
                    
                    detected_events.append({
                        'sound_class': label,
                        'start_time_seconds': round(event_start_frame / frames_per_second, 3),
                        'end_time_seconds': round(event_end_frame / frames_per_second, 3),
                        'duration_seconds': round(duration_seconds, 3),
                        'peak_probability': float(np.max(event_block_probs)),
                        'average_probability': float(np.mean(event_block_probs))
                    })

    # Sort events by a score combining duration and average probability, then by start time
    detected_events.sort(key=lambda x: (x['duration_seconds'] * x['average_probability']), reverse=True)
    
    # Keep the top 40 most prominent events and then sort them chronologically
    top_events = sorted(detected_events[:40], key=lambda x: x['start_time_seconds'])

    with open(summary_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['sound_class', 'start_time_seconds', 'end_time_seconds', 'duration_seconds', 'peak_probability', 'average_probability']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top_events)
    print(f'Saved summary events CSV to: \033[1;34m{summary_csv_path}\033[1;0m')

    # 2. Generate summary_manifest.json
    manifest_path = os.path.join(output_dir, 'summary_manifest.json')
    final_overlay_path = f"{os.path.splitext(output_video_path)[0]}_overlay.mp4"
    
    artifacts = {
        'source_file': os.path.basename(audio_path),
        'analysis_type': 'Audio Event Detection with PANNs',
        'artifacts': {
            'summary_events.csv': 'A summary of the most prominent, continuous sound events, detailing their start, end, duration, and intensity.',
            'eventogram.png': 'Visualization of the top 10 sound events over time.',
            'full_event_log.csv': 'Full, detailed log of sound event probabilities for each time frame.',
            os.path.basename(output_video_path): 'Video of the eventogram with original audio.',
        }
    }
    if os.path.exists(final_overlay_path):
        artifacts['artifacts'][os.path.basename(final_overlay_path)] = 'Original video overlaid with the eventogram visualization.'

    with open(manifest_path, 'w') as f:
        json.dump(artifacts, f, indent=4)
    print(f'Saved manifest JSON to: \033[1;34m{manifest_path}\033[1;0m')
    
    print(f"⏲  🗃️  Reminder: input file duration: \033[1;34m{duration}\033[0m")

if __name__ == '__main__':
    print(f"Eventogrammer, version 6.1.0, changed: idempotency, soundaudio importer for Termux")

 
    print(f"Notes: a file of duration of 30 mins requires 6GB RAM to process, with the processing time ratio: 1 second of orignal duration : 10 seconds to process. This script is an adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn so see there if something be amiss.")
    print(f"Using moviepy version: {moviepy.__version__}")
    print(f"Using torchaudio version (better be pinned at version 2.8.0 for a while...): {torchaudio.__version__}")

    print(f"")

    parser = argparse.ArgumentParser(description='Audio tagging and Sound event detection.')
    parser.add_argument('audio_path', type=str, help='Path to audio or video file')
    parser.add_argument('--mode', choices=['audio_tagging', 'sound_event_detection'],
                        default='sound_event_detection', help='Select processing mode')
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--translucency', type=float, default=0.7,
                        help='Overlay translucency (0 to 1)')
    parser.add_argument('--overlay_size', type=float, default=0.2,
                        help='Overlay size as fraction of video height')
    parser.add_argument('--dynamic_eventogram', action='store_true', default=False,
                        help='Generate dynamic eventogram with scrolling window')
    parser.add_argument('--crf', type=int, default=23, help='FFmpeg CRF value (0-51, lower is higher quality)')
    parser.add_argument('--bitrate', type=str, default=None, help='FFmpeg video bitrate (e.g., "2000k" for 2 Mbps)')
    parser.add_argument('--window_duration', type=float, default=30.0,
                        help='Duration of sliding window for dynamic eventogram (seconds)')
    parser.add_argument('--use_adaptive_window', action='store_true', default=False,
                        help='Use adaptive window size based on event boundaries')

    args = parser.parse_args()

    if args.mode == 'audio_tagging':
        audio_tagging(args)
    else:
        sound_event_detection(args)
