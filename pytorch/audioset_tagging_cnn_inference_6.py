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
#Torch and torchaudio and coverage are version sensitive. Careful with: python -m pip install torch torchaudio--upgrade --extra-index-url https://download.pytorch.org/whl/cpu

import torch
import torchaudio
import torchcodec
# AI ARCHITECTURAL NOTE: torchaudio is essential for:
# 1. High-level Decoding/Normalization/Resampling to 32kHz (fixed model rate).
# 2. GPU/CUDA efficiency on large files.
# 3. Direct Tensor-land processing to minimize NumPy/Memory overhead in Termux.
# If torchaudio.load fails on Android, prefer the 'android=linux' torchcodec hack
# over full library replacement to preserve these architectural benefits.
import csv
import datetime
import time
import subprocess
import shutil
import moviepy
import warnings
import soundfile as sf
import psutil
#import coverage 

from moviepy import AudioFileClip, VideoClip
import json
import collections
import plotly.offline as pyo
import tempfile # Import tempfile for temporary file handling

# Suppress torchaudio deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# Add utils directory to sys.path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../utils'))
from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config

    



def get_media_metadata(input_media_path):
    """
    Extract duration, FPS, resolution, and is_video flag from a media file using FFprobe.
    Handles audio-only and video files universally, prioritizing format duration.
    Returns: (duration, avg_fps, width, height, is_video, r_fps)
    """
    try:
        # Run ffprobe to get format and stream info
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-print_format', 'json', input_media_path],
            capture_output=True, text=True
        )
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            print(f"\033[1;31mError: Failed to parse FFprobe JSON output for {input_media_path}: {e}\033[0m")
            data = {}

        streams = data.get('streams', [])
        format_info = data.get('format', {})
        
        duration, avg_fps, width, height, r_fps = None, None, None, None, None
        is_video = any(s['codec_type'] == 'video' and s.get('codec_name') not in ['mjpeg', 'png'] for s in streams)

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
            
            # Average FPS
            avg_fr = video_stream.get('avg_frame_rate')
            if avg_fr and '/' in avg_fr:
                try:
                    num, den = map(int, avg_fr.split('/'))
                    avg_fps = num / den if den else None
                except (ValueError, TypeError): pass
            
            # Real FPS (for VFR check)
            r_fr = video_stream.get('r_frame_rate')
            if r_fr and '/' in r_fr:
                try:
                    num, den = map(int, r_fr.split('/'))
                    r_fps = num / den if den else None
                except (ValueError, TypeError): pass
            
            width = int(video_stream.get('width', 0)) if video_stream.get('width') else None
            height = int(video_stream.get('height', 0)) if video_stream.get('height') else None
            
            if duration is None and video_stream.get('nb_frames') and avg_fps:
                try:
                    duration = int(video_stream['nb_frames']) / avg_fps
                except (ValueError, TypeError):
                    print(f"\033[1;33mWarning: Invalid nb_frames or fps for duration calculation in {input_media_path}\033[0m")

        # Check audio stream fallback
        if duration is None:
            audio_stream = next((s for s in streams if s['codec_type'] == 'audio'), None)
            if audio_stream and audio_stream.get('duration'):
                try:
                    duration = float(audio_stream['duration'])
                except (ValueError, TypeError):
                    print(f"\033[1;33mWarning: Invalid audio stream duration in {input_media_path}\033[0m")

        # Final Fallback: Direct duration probe
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
            return None, None, None, None, False, None

        # --- USER INTERFACE & LOGGING ---
        duration_str = str(datetime.timedelta(seconds=int(duration))) if duration else "?"
        print(f"⏲  🗃️  File duration: \033[1;34m{duration_str}\033[0m")
        if avg_fps:
            print(f"🮲  🗃️  Video FPS (avg): \033[1;34m{avg_fps:.3f}\033[0m")
        if width and height:
            print(f"📽  🗃️  Video resolution: \033[1;34m{width}x{height}\033[0m")
            if height > width:
                print("\n\033[1;33m--- WARNING: Portrait Video Detected ---\033[0m")
                print("\nProcessing a portrait video may result in a narrow or distorted overlay.")
                print("For best results, it is recommended to rotate the video to landscape before processing.")
                print("\nYou can use the following ffmpeg command to rotate the video:\n")
                print(f"\033[1;32mffmpeg -i \"{input_media_path}\" -vf \"transpose=1\" \"{os.path.splitext(input_media_path)[0]}_rotated.mp4\"\033[0m\n")
                print("To do so, break the script now (Ctrl+C) and run the command above.")
                print("Otherwise, we are proceeding with the portrait video in 2 seconds...")
                time.sleep(2)
                print("\033[1;33m------------------------------------------\033[0m\n")

        return duration, avg_fps, width, height, is_video, r_fps

    except Exception as e:
        print(f"\033[1;31mError: Metadata probe failed for {input_media_path}: {e}\033[0m")
        return None, None, None, None, False, None

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

    waveform, sr = torchaudio.load(audio_path)
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
    # --- PHASE 0: Setup & Environment ---
    output_fps = args.output_fps
    viz_fps = args.vis_fps  # Fixed constant for internal resolution
    adaptive_lookahead = args.adaptive_lookahead
    
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    inference_fps = sample_rate // hop_size # Typically 100
    
    top_k = 10  # Number of top events to track/visualize
    fig_width_px, fig_height_px, dpi = 1280, 480, 100
    
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    
    # Path Consolidation: Clearly define the roles of each media path
    source_media = args.audio_path    # The original file (never changed)
    inference_media = source_media   # The file used for duration/audio loading (may be updated to recovered/sanitized)
    overlay_media = source_media     # The file used for the final video overlay
    
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    else:
        print('Using CPU.')

    classes_num = config.classes_num
    labels = config.labels
    
    audio_dir = os.path.dirname(source_media)
    base_name = get_filename(source_media)
    checkpoint_name = os.path.basename(checkpoint_path)
    output_dir = os.path.join(audio_dir, f'{base_name}_{checkpoint_name}_audioset_tagging_cnn')
    create_folder(output_dir)

    # --- PHASE 1: Dependency Injection (AI Guide) ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        guide_src_path = os.path.normpath(os.path.join(script_dir, '..', 'docs', 'auditory_cognition_guide_template.md'))
        guide_dest_path = os.path.join(output_dir, 'auditory_cognition_guide_template.md')
        
        if os.path.exists(guide_src_path):
            shutil.copy(guide_src_path, guide_dest_path)
            print(f'Copied AI analysis guide to: \033[1;34m{guide_dest_path}\033[1;0m')
        else:
            print(f'\033[1;33mWarning: AI analysis guide not found at {guide_src_path}\033[0m')
    except Exception as e:
        print(f'\033[1;33mWarning: Failed to copy AI analysis guide: {e}\033[0m')

    tag_suffix = "_audioset_tagging_cnn"
    fig_path = os.path.join(output_dir, 'eventogram.png')

    # --- PHASE 2: Idempotency Check ---
    csv_path = os.path.join(output_dir, 'full_event_log.csv')
    
    # Efficiently probe all metadata once
    duration, video_fps, video_width, video_height, is_video, r_fps = get_media_metadata(source_media)
    if duration is None: return # Error already printed in probe

    # Define what files we strictly expect to see before skipping
    required_files = [csv_path]
    if args.static_eventogram:
        vid_path = os.path.join(output_dir, f'{base_name}{tag_suffix}_eventogram_static.mp4')
        required_files.append(f"{os.path.splitext(vid_path)[0]}_overlay.mp4" if is_video else vid_path)
            
    if args.dynamic_eventogram:
        vid_path = os.path.join(output_dir, f"{base_name}{tag_suffix}_eventogram_dynamic.mp4")
        required_files.append(f"{os.path.splitext(vid_path)[0]}_overlay.mp4" if is_video else vid_path)
            
    if all(os.path.exists(f) for f in required_files):
        print(f"✅ Skipping {source_media}, all requested outputs already exist in: \033[1;34m{output_dir}\033[1;0m")
        return

    # Check for sufficient disk space
    disk_usage = shutil.disk_usage(audio_dir)
    if disk_usage.free < 1e9:
        print(f"\033[1;31mError: Insufficient disk space ({disk_usage.free / 1e9:.2f} GB free). Exiting.\033[0m")
        return
    
    # --- PHASE 3: Model Loading ---
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

    # --- PHASE 4: Media Integrity & Recovery ---
    # Recovery: Attempt conversion to MP3 if duration detection returned 0
    if duration == 0:
        print(f"\033[1;33mWarning: Duration probe failed for {inference_media}. Attempting MP3 recovery...\033[0m")
        mp3_path = os.path.join(tempfile.gettempdir(), f"{base_name}_recovered.mp3")

        try:
            subprocess.run(['ffmpeg', '-i', inference_media, '-y', mp3_path], check=True, capture_output=True)
            inference_media = mp3_path # Rest of the script uses this recovered file
            duration, video_fps, video_width, video_height, is_video, r_fps = get_media_metadata(inference_media)
            
            if duration is None or duration == 0:
                print(f"\033[1;31mError: Could not determine duration even after recovery. Exiting.\033[0m")
                return
        except Exception as e:
            print(f"\033[1;31mRecovery conversion failed: {e}\033[0m")
            return

    if is_video and (video_width is None or video_height is None):
        video_width, video_height = 1280, 720
        print(f"\033[1;33mWarning: Video dimensions not detected, using default 1280x720.\033[0m")
    
    # --- PHASE 5: VFR Check & Constant Frame Rate Sanitization ---
    temp_video_path = None
    if is_video and r_fps and video_fps:
        if abs(r_fps - video_fps) > 0.01:
            print(f"\033[1;33mDetected VFR ({r_fps:.2f}/{video_fps:.2f}). Re-encoding to CFR for sync...\033[0m")
            temp_video_path = os.path.join(tempfile.gettempdir(), f'temp_cfr_{base_name}.mp4')
            target_fps = video_fps if video_fps > 0 else output_fps
            subprocess.run([
                'ffmpeg', '-loglevel', 'warning', '-i', inference_media, '-r', str(target_fps), '-fps_mode', 'cfr', '-c:a', 'aac', temp_video_path, '-y'
            ], check=True)
            inference_media = temp_video_path

    # --- PHASE 6: Waveform Loading (Sanitary Gate) ---
    try:
        waveform, sr = torchaudio.load(inference_media)
    except Exception as e:
        print(f"\033[1;33mWarning: Direct torchaudio load failed ({e}). Attempting PCM sanitization...\033[0m")
        sanitized_temp_path = os.path.join(tempfile.gettempdir(), f'sanitized_{base_name}.wav')
        try:
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', inference_media, '-vn', '-acodec', 'pcm_s16le', sanitized_temp_path, '-y'], check=True)
            try:
                waveform, sr = torchaudio.load(sanitized_temp_path)
            except:
                waveform_np, sr = sf.read(sanitized_temp_path)
                waveform = torch.from_numpy(waveform_np).float().T if waveform_np.ndim > 1 else torch.from_numpy(waveform_np).float().unsqueeze(0)
            inference_media = sanitized_temp_path
        except Exception as generic_err:
            print(f"\033[1;31mError: Unexpected failure during audio loading: {generic_err}\033[0m")
            return

    import gc
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
        gc.collect()
        
    waveform = waveform.mean(dim=0, keepdim=True)  # Downmix to mono
    waveform_np = waveform.squeeze(0).numpy() # Move to NumPy for memory-safe chunking
    
    # CRITICAL: Drop torch tensors to free significant RAM immediately (essential for Termux)
    del waveform
    gc.collect()
    waveform = waveform_np 

    # --- PHASE 7: Chunked Model Inference (Memory-Safe) ---
    chunk_duration = 180  # 3 minutes
    chunk_samples = int(chunk_duration * sample_rate)
    vis_downsample = inference_fps // viz_fps
    
    framewise_vis_list = []
    stft_vis_list = []
    
    avail_ram = psutil.virtual_memory().available / (1024 * 1024)
    print(f"📊  Starting inference in {chunk_duration/60}m chunks. (RAM Avail: {avail_ram:.0f} MB)")
    print(f"    Resolution: Disk {inference_fps} Hz (Data Frames) | Visualization {viz_fps} Hz (UI/RAM)")

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time'] + list(labels))
        
        for start in range(0, len(waveform), chunk_samples):
            chunk = waveform[start:start + chunk_samples]
            if len(chunk) < sample_rate // 10: continue
            
            # Step A: Inference
            chunk_tensor = move_data_to_device(torch.from_numpy(chunk[None, :]).float(), device)
            with torch.no_grad():
                model.eval()
                try:
                    batch_output_dict = model(chunk_tensor, None)
                    chunk_out = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
                except Exception as e:
                    print(f"\033[1;31mInference error in chunk: {e}\033[0m"); continue
            
            # Step B: Write high-res results to disk immediately (Lean memory)
            chunk_start_time = start / sample_rate
            csv_downsample = max(1, inference_fps // args.csv_fps)
            for i in range(0, len(chunk_out), csv_downsample):
                timestamp = chunk_start_time + (i / inference_fps)
                writer.writerow([round(timestamp, 3)] + chunk_out[i].tolist())
            
            # Step C: Downsample for RAM-based visualization (Max-pooling preserves short events)
            for i in range(0, len(chunk_out), vis_downsample):
                vis_slice = chunk_out[i : i + vis_downsample]
                if len(vis_slice) > 0:
                    framewise_vis_list.append(np.max(vis_slice, axis=0))

            # Step D: Chunked STFT for visualization background
            chunk_tensor_stft = torch.from_numpy(chunk).to(device)
            chunk_stft = torch.stft(
                chunk_tensor_stft, n_fft=window_size, hop_length=hop_size,
                window=torch.hann_window(window_size).to(device),
                center=True, return_complex=True
            ).abs().cpu().numpy()
            stft_vis_list.append(chunk_stft[:, ::vis_downsample])
            
            avail_ram = psutil.virtual_memory().available / (1024 * 1024)
            print(f"Chunk at {int(chunk_start_time/60)}m finished. (RAM Avail: {avail_ram:.0f} MB)")

    # --- PHASE 8: Result Aggregation & Metadata Consolidation ---
    framewise_output = np.array(framewise_vis_list)
    stft = np.concatenate(stft_vis_list, axis=1)
    del framewise_vis_list, stft_vis_list
    
    # DATA-DRIVEN DURATION: Use real data length as truth (fixes VBR/probe guesses)
    frames_num = len(framewise_output)
    duration = frames_num / viz_fps
    
    print(f'Aggregation complete. Internal Viz resolution: \033[1;34m{viz_fps} Hz (Data Frames)\033[1;0m')
    print(f'Final analysis duration: \033[1;34m{duration:.2f}s\033[1;0m')

    # --- PHASE 9: Static Eventogram Generation (PNG) ---
    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]
    top_result_mat = framewise_output[:, sorted_indexes[0:top_k]]
    top_labels = np.array(labels)[sorted_indexes[0:top_k]]

    fig = plt.figure(figsize=(fig_width_px / dpi, fig_height_px / dpi), dpi=dpi)
    
    # Pass 1: Create dummy plot to measure Y-axis label pixel width (Dynamic Margin)
    gs_init = fig.add_gridspec(2, 1, height_ratios=[1, 1], left=0.1, right=1.0, top=0.95, bottom=0.08, hspace=0.05)
    axs_init = [fig.add_subplot(gs_init[0]), fig.add_subplot(gs_init[1])]
    axs_init[1].set_yticks(np.arange(0, top_k)); axs_init[1].set_yticklabels(top_labels, fontsize=14)
    fig.canvas.draw()
    
    max_label_width_px = max(lbl.get_window_extent(renderer=fig.canvas.get_renderer()).width for lbl in axs_init[1].yaxis.get_majorticklabels())
    left_frac = min(0.45, (max_label_width_px + 14) / fig_width_px)
    
    # Pass 2: Final Render with corrected margin
    fig.clear()
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], left=left_frac, right=1.0, top=0.95, bottom=0.08, hspace=0.05)
    axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]

    axs[0].matshow(np.log(stft + 1e-10), origin='lower', aspect='auto', cmap='jet')
    axs[0].set_ylabel('Freq bins', fontsize=14); axs[0].set_title('Spectrogram and Eventogram', fontsize=14)
    axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)

    tick_interval = max(5, int(duration / 20))
    x_ticks = np.arange(0, frames_num, viz_fps * tick_interval)
    axs[1].xaxis.set_ticks(x_ticks)
    axs[1].xaxis.set_ticklabels([int(t / viz_fps) for t in x_ticks], rotation=45, ha='right', fontsize=10)
    axs[1].set_xlim(0, frames_num); axs[1].set_yticks(np.arange(0, top_k)); axs[1].set_yticklabels(top_labels, fontsize=14)
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds', fontsize=14); axs[1].xaxis.set_ticks_position('bottom')
    
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0); plt.close(fig)
    print(f'Saved eventogram PNG to: \033[1;34m{fig_path}\033[1;0m')




    # --- PHASE 10: AI-Friendly Summary Generation (Event Block Detection) ---
    print("📊  Generating AI-friendly event summary files…")
    summary_csv_path = os.path.join(output_dir, 'summary_events.csv')
    
    # Event detection heuristics
    onset_threshold, offset_threshold = 0.01, 0.01
    min_event_duration_seconds = 0.5 
    top_n_per_class = 3 # Only export the strongest N events per class

    events_by_class = {label: [] for label in top_labels}

    # Identify continuous blocks of sound for the top classes shown in the eventogram
    for label_idx, label in enumerate(labels):
        if label not in top_labels: continue
        
        in_event, event_start_frame = False, 0
        for frame_index, prob in enumerate(framewise_output[:, label_idx]):
            if not in_event and prob > onset_threshold:
                in_event, event_start_frame = True, frame_index
            elif in_event and prob < offset_threshold:
                in_event = False
                duration_frames = frame_index - event_start_frame
                duration_secs = duration_frames / viz_fps
                
                if duration_secs >= min_event_duration_seconds:
                    event_block_probs = framewise_output[event_start_frame:frame_index, label_idx]
                    events_by_class[label].append({
                        'sound_class': label,
                        'start_time_seconds': round(event_start_frame / viz_fps, 3),
                        'end_time_seconds': round(frame_index / viz_fps, 3),
                        'duration_seconds': round(duration_secs, 3),
                        'peak_probability': float(np.max(event_block_probs)),
                        'average_probability': float(np.mean(event_block_probs))
                    })

    # Consolidate and sort top events chronologically
    top_events = []
    for label, events in events_by_class.items():
        if events:
            top_events.extend(sorted(events, key=lambda x: x['peak_probability'], reverse=True)[:top_n_per_class])
    
    final_sorted_events = sorted(top_events, key=lambda x: x['start_time_seconds'])

    with open(summary_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['sound_class', 'start_time_seconds', 'end_time_seconds', 'duration_seconds', 'peak_probability', 'average_probability']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(final_sorted_events)
    print(f'Saved summary events CSV to: \033[1;34m{summary_csv_path}\033[1;0m')

    # --- PHASE 11: Detailed AI-Friendly Event Delta Map (RLE Compression) ---
    # This format is optimized for AI attention mechanisms: it uses Run-Length Encoding (RLE)
    # and probability deltas to focus on "attacks" and "decays".
    print("📊  Generating detailed AI-friendly event delta JSON (RLE optimized)…")
    json_ai_path = os.path.join(output_dir, 'detailed_events_delta_ai_attention_friendly.json')
    ai_threshold = 0.05
    events_derivative = collections.defaultdict(list)

    def zip_trace(trace):
        """Compress trace by replacing consecutive zeros with a skip marker."""
        zipped = []
        zero_count = 0
        for val in trace:
            if val == 0.0:
                zero_count += 1
            else:
                if zero_count > 0:
                    zipped.append({"skip": zero_count}); zero_count = 0
                zipped.append(val)
        if zero_count > 0: zipped.append({"skip": zero_count})
        return zipped

    for label_idx, label in enumerate(labels):
        current_event = None
        probs_stream = framewise_output[:, label_idx]
        
        for frame_index, prob in enumerate(probs_stream):
            if prob > ai_threshold:
                if current_event is None:
                    current_event = {
                        "start": round(frame_index / viz_fps, 3),
                        "peak": float(prob), "trace": [float(prob)]
                    }
                else:
                    current_event["peak"] = max(current_event["peak"], float(prob))
                    current_event["trace"].append(float(prob))
            elif current_event:
                # Finalize event and calculate momentum (deltas)
                tr = current_event["trace"]
                deltas = [tr[0]] # Anchor point
                for i in range(1, len(tr)): deltas.append(round(tr[i] - tr[i-1], 6))
                
                events_derivative[label].append({
                    "start_time": current_event["start"],
                    "end_time": round(frame_index / viz_fps, 3),
                    "peak_prob": current_event["peak"],
                    "delta_trace": zip_trace(deltas)
                })
                current_event = None
        
        if current_event: # Cleanup trailing event
            tr = current_event["trace"]
            deltas = [tr[0]]
            for i in range(1, len(tr)): deltas.append(round(tr[i] - tr[i-1], 6))
            events_derivative[label].append({
                "start_time": current_event["start"],
                "end_time": round(len(probs_stream) / viz_fps, 3),
                "peak_prob": current_event["peak"],
                "delta_trace": zip_trace(deltas)
            })

    with open(json_ai_path, 'w') as f:
        json.dump({k: v for k, v in events_derivative.items() if v}, f, indent=2)
    print(f'Saved AI-friendly delta JSON to: \033[1;34m{json_ai_path}\033[1;0m')

    # --- PHASE 12: Interactive Plotly Dashboard (Top 50 Events) ---
    print("📊  Generating interactive Plotly dashboard…")
    html_dashboard_path = os.path.join(output_dir, 'interactive_dashboard.html')
    
    # Identify top 50 classes by total popularity
    popularity = np.sum(framewise_output, axis=0)
    top_50_indices = np.argsort(popularity)[::-1][:50]
    
    times = [round(i / viz_fps, 2) for i in range(frames_num)]
    traces_data = []
    for idx in top_50_indices:
        traces_data.append({"name": labels[idx], "y": np.round(framewise_output[:frames_num, idx], 4).tolist()})
    
    json_payload = json.dumps({"times": times, "traces": traces_data})
    plotly_js = pyo.get_plotlyjs()
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Audio Analysis - {base_name}</title>
    <script type="text/javascript">{plotly_js}</script>
    <style>
        body {{ font-family: sans-serif; margin: 20px; background: #fafafa; color: #333; }}
        #plot {{ width: 100%; height: 85vh; background: white; border-radius: 8px; border: 1px solid #ddd; }}
        .header {{ margin-bottom: 15px; border-left: 5px solid #2563eb; padding-left: 15px; }}
        code {{ background: #eee; padding: 2px 5px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Sound Event Analysis: {base_name}</h1>
        <p>Interactive Plotly Dashboard | Top 50 Classes | Source: <code>full_event_log.csv</code></p>
    </div>
    <div id="plot"></div>
    <script type="text/javascript">
        const data = {json_payload};
        const traces = data.traces.map((t, index) => ({{
            x: data.times, y: t.y, name: t.name, mode: 'lines',
            visible: index < 10 ? true : 'legendonly', line: {{ width: 2 }}
        }}));
        Plotly.newPlot('plot', traces, {{
            title: 'Top 50 Sound Events Momentum',
            xaxis: {{ title: 'Seconds', gridcolor: '#eee' }},
            yaxis: {{ title: 'Probability', range: [0, 1], gridcolor: '#eee' }},
            hovermode: 'x unified', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            margin: {{ t: 50, b: 50, l: 60, r: 20 }}
        }}, {{ responsive: true }});
    </script>
</body>
</html>
"""
    with open(html_dashboard_path, 'w', encoding='utf-8') as f: f.write(html_content)
    print(f'Saved interactive dashboard to: \033[1;34m{html_dashboard_path}\033[1;0m')

    """
    # Auto-open the dashboard using the system 'open' command
    try:
        subprocess.run(['open', html_dashboard_path], check=False)
    except Exception as e:
        print(f"Warning: Could not auto-open dashboard: {e}")
    """

    


    # --- PHASE 13: Video Rendering Pipeline (Data-Synchronous & Cached) ---
    if args.dynamic_eventogram or args.static_eventogram:
        print(f"🎞  Initializing video rendering pipeline ({output_fps} FPS)…")
        
        # Rendering Strategy: Dynamic (Scrolling) vs Static (Marker only)
        if args.dynamic_eventogram:
            output_video_path = os.path.join(output_dir, f"{base_name}{tag_suffix}_eventogram_dynamic.mp4")
            window_frames = int(args.window_duration * viz_fps)
            half_window = window_frames // 2 

            # PRECOMPUTE: Map every data point to its local acoustic window (once per run)
            print(f"📊  Precomputing {frames_num} windows (Adaptive={args.use_adaptive_window})…")
            precomputed_data = []
            for i in range(frames_num):
                start_f, end_f = max(0, i - half_window), min(frames_num, i + half_window)
                
                # Adaptive logic: Try to center the window on acoustic "peaks" rather than strict time
                if args.use_adaptive_window:
                    kl_threshold = 0.5
                    lookahead_f = int(adaptive_lookahead * viz_fps)
                    for offset in range(half_window, half_window + lookahead_f, int(viz_fps)):
                        if start_f - offset >= 0:
                            prev_p = np.mean(framewise_output[start_f-offset:start_f], axis=0)
                            curr_p = np.mean(framewise_output[start_f:start_f+offset], axis=0)
                            if compute_kl_divergence(prev_p, curr_p) > kl_threshold:
                                start_f = max(0, start_f - offset // 2); break
                        if end_f + offset < frames_num:
                            curr_p = np.mean(framewise_output[end_f-offset:end_f], axis=0)
                            next_p = np.mean(framewise_output[end_f:end_f+offset], axis=0)
                            if compute_kl_divergence(curr_p, next_p) > kl_threshold:
                                end_f = min(frames_num, end_f + offset // 2); break
                
                win_out, win_idxs = get_dynamic_top_events(framewise_output, start_f, end_f, top_k)
                if win_out.size == 0:
                    win_out, win_idxs = np.zeros((end_f - start_f, top_k)), sorted_indexes[:top_k]
                precomputed_data.append({'start': start_f, 'end': end_f, 'out': win_out, 'idxs': win_idxs})

            # Setup Persistent Figure for Sprint Speed (Reuses same canvas objects to avoid GC spikes)
            fig_fr = plt.figure(figsize=(fig_width_px/dpi, fig_height_px/dpi), dpi=dpi)
            gs_fr = fig_fr.add_gridspec(2, 1, height_ratios=[1, 1], left=left_frac, right=1.0, top=0.95, bottom=0.08, hspace=0.05)
            axs_fr = [fig_fr.add_subplot(gs_fr[0]), fig_fr.add_subplot(gs_fr[1])]
            
            stft_log = np.log(stft + 1e-10)
            v_min, v_max = np.percentile(stft_log, [1, 99]) # Fix contrast globally

            # Persistent Artists (The "Engine")
            im_spec = axs_fr[0].imshow(np.zeros((stft.shape[0], 2)), origin='lower', aspect='auto', cmap='jet', vmin=v_min, vmax=v_max)
            im_event = axs_fr[1].imshow(np.zeros((top_k, 2)), origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
            marker_lines = [ax.axvline(x=0, color='red', linewidth=2, alpha=0.8) for ax in axs_fr]
            
            axs_fr[0].set_ylabel('Freq bins', fontsize=14); axs_fr[1].set_xlabel('Seconds', fontsize=14)
            axs_fr[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
            axs_fr[0].tick_params(labelbottom=False, labeltop=False, bottom=False, top=False)
            
            last_idxs = {"val": None}

            def draw_strategy(i):
                """High-speed redraw using set_data instead of creating new plots."""
                data = precomputed_data[i]
                s_f, e_f = data['start'], data['end']
                t_start, t_end, t_curr = s_f / viz_fps, e_f / viz_fps, i / viz_fps
                
                im_spec.set_data(stft_log[:, s_f:e_f]); im_spec.set_extent([t_start, t_end, 0, stft.shape[0]])
                im_event.set_data(data['out'].T); im_event.set_extent([t_start, t_end, 0, top_k])
                
                # Only update labels if the "local context" changed (saves CPU)
                if last_idxs["val"] is None or not np.array_equal(last_idxs["val"], data['idxs']):
                    axs_fr[1].set_yticks(np.arange(0.5, top_k + 0.5))
                    axs_fr[1].set_yticklabels(np.array(labels)[data['idxs']][::-1], fontsize=14)
                    last_idxs["val"] = data['idxs']
                
                axs_fr[0].set_title(f'Spectrogram and Eventogram (t={t_curr:.1f}s)', fontsize=14)
                for ax, line in zip(axs_fr, marker_lines):
                    ax.set_xlim(t_start, t_end); line.set_xdata([t_curr, t_curr])
                
                fig_fr.canvas.draw()
                return np.frombuffer(fig_fr.canvas.buffer_rgba(), dtype=np.uint8).reshape((fig_height_px, fig_width_px, 4))[:,:,:3]

        else: # Static Eventogram (Just a moving red line over the PNG)
            output_video_path = os.path.join(output_dir, f'{base_name}{tag_suffix}_eventogram_static.mp4')
            base_img = plt.imread(fig_path)
            if base_img.dtype == np.float32: base_img = (base_img * 255).astype(np.uint8)
            base_img = base_img[:, :, :3]
            h, w, _ = base_img.shape
            x_start, x_end = int(left_frac * w), w

            def draw_strategy(i):
                img = base_img.copy()
                marker_x = int(x_start + (x_end - x_start) * (i / max(frames_num - 1, 1)))
                img[:, max(0, marker_x-1):min(w, marker_x+1)] = [255, 0, 0]
                return img

        # --- SMART CACHE LAYER ---
        # Since video FPS (30) >> data Hz (5), we reuse the last rendered frame
        # for 6 consecutive video frames to save ~80% of CPU rendering time.
        frame_cache = {"last_i": -1, "last_img": None}

        def make_frame(t):
            i = min(int(t * viz_fps), frames_num - 1)
            if i == frame_cache["last_i"]: return frame_cache["last_img"]
            
            img = draw_strategy(i)
            frame_cache.update({"last_i": i, "last_img": img})
            return img

        # Final Compositing & Export
        eventogram_clip = VideoClip(make_frame, duration=duration)
        audio_clip = AudioFileClip(inference_media)
        final_clip = eventogram_clip.with_audio(audio_clip)
        final_clip.fps = output_fps
        
        final_clip.write_videofile(
            output_video_path, codec="libx264", fps=output_fps, 
            threads=os.cpu_count(),
            temp_audiofile=os.path.join(output_dir, "temp_render_audio.mp3")
        )
        print(f"✅ Saved eventogram video to: \033[1;34m{output_video_path}\033[1;0m")
        if args.dynamic_eventogram: plt.close(fig_fr)

        # --- PHASE 14: FFmpeg Overlay (Video-on-Video) ---
        if is_video:
            print("🎬  Overlaying the source media with the created eventogram…")
            final_overlay_path = f"{os.path.splitext(output_video_path)[0]}_overlay.mp4"
            
            # Use pre-probed dimensions (no redundant FFprobe call)
            b_w, b_h = video_width, video_height
            
            # Leaner Scaling: Match eventogram width (1280px) if source is smaller; force even height
            t_w = max(b_w, fig_width_px)
            t_h = int(b_h * t_w / b_w) // 2 * 2 
            
            overlay_cmd = [
                "ffmpeg", "-y", "-i", overlay_media, "-i", output_video_path, "-loglevel", "warning",
                "-filter_complex", (
                    f"[0:v]scale={t_w}:{t_h}[main];" # Scale source video
                    f"[1:v]scale={t_w}:{int(t_h * args.overlay_size)}[ovr];" # Scale eventogram
                    f"[ovr]format=rgba,colorchannelmixer=aa={args.translucency}[ovr_t];" # Set alpha
                    "[main][ovr_t]overlay=x=0:y=H-h[v]" # Stack them at the bottom
                ),
                "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest"
            ]
            if args.bitrate: overlay_cmd.extend(["-b:v", args.bitrate])
            else: overlay_cmd.extend(["-crf", str(args.crf)])
            
            overlay_cmd.append(final_overlay_path)
            
            try:
                subprocess.run(overlay_cmd, check=True)
                print(f"✅ 🎥 Final overlay saved to: \033[1;34m{final_overlay_path}\033[1;0m")
            except subprocess.CalledProcessError as e:
                print(f"\033[1;31mError during FFmpeg overlay: {e}\033[0m")
            
            # Cleanup temporary CFR file if created
            if temp_video_path and os.path.exists(temp_video_path): os.remove(temp_video_path)
        else:
            print("🎧 Source is audio-only, skipping video overlay.")

    print(f"⏲  🗃️  Analysis finished. Input duration: \033[1;34m{duration:.2f}s\033[0m")

if __name__ == '__main__':



    

    parser = argparse.ArgumentParser(
        description='Audio tagging and Sound event detection.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=""" """
    )
    parser.add_argument('audio_path', type=str, help='Path to the media file')
    parser.add_argument('--mode', choices=['audio_tagging', 'sound_event_detection'],
                        default='sound_event_detection', help='Select the processing mode')
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000)
    parser.add_argument('--model_type', type=str, required=True, help='Model architecture type')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to pretrained .pth file')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA for inference')
    parser.add_argument('--translucency', type=float, default=0.7,
                        help='Overlay translucency (0 to 1)')
    parser.add_argument('--overlay_size', type=float, default=0.2,
                        help='Overlay size as fraction of video height')
    parser.add_argument('--dynamic_eventogram', action='store_true', default=False,
                        help='Generate dynamic eventogram with scrolling window (slower to generate)')
    parser.add_argument('--static_eventogram', action='store_true', default=False,
                        help='Generate static eventogram with a scrolling marker (faster to generate)')                        
    parser.add_argument('--crf', type=int, default=23, help='FFmpeg CRF value (0-51, lower is higher quality)')
    parser.add_argument('--bitrate', type=str, default=None, help='FFmpeg video bitrate (e.g., "2000k" for 2 Mbps)')
    parser.add_argument('--window_duration', type=float, default=30.0,
                        help='Duration of sliding window for the dynamic eventogram (in seconds)')
    parser.add_argument('--use_adaptive_window', action='store_true', default=False,
                        help='Use adaptive window size based on the event boundaries')
    parser.add_argument('--csv_fps', type=int, default=5,
                        help='Data frames per second (Hz) to write to full_event_log.csv. '
                             'Use 100 for full resolution (very large file). '
                             '5 is enough for Shapash + outlier detection.')
    parser.add_argument('--vis_fps', type=int, default=5,
                        help='Data frames per second (Hz) for internal RAM-based visualization data (RAM guard)')
    parser.add_argument('--output_fps', type=int, default=30,
                        help='FPS for the final rendered video output')
    parser.add_argument('--adaptive_lookahead', type=float, default=30.0,
                        help='Max seconds to look ahead/back for acoustic boundaries in adaptive mode')

    # Heuristic to find audio_path for the help/info block
    audio_path_hint = "[audio_path]"
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if not arg.startswith('-'):
                audio_path_hint = arg
                break                        
                        
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
 
    print(f"Eventogrammer, version 6.8.9") 
    print(f"Adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn")
    print()

    print(f"Recent Material Changes:")
    print(f"* Constants Promoted: vis_fps, output_fps, and adaptive_lookahead are now CLI arguments.")
    print(f"* Speed Hack: Persistent Matplotlib figures with Artist Updates.")
    print(f"* Visual Fix: Proper window centering for scrolling eventograms.")
    print(f"* Refactor: Gemini AI rationalized path management and code structure.")
    print()

    print(f"Note on the models:")   
    print(f"* Cnn14_DecisionLevelMax (Sound Event Detection): Uses Decision-level pooling to maintain")
    print(f"  time resolution. Essential for generating Eventograms and high-res CSV logs.")
    print(f"* Other models: Best for global audio tagging (use the '--audio_tagging' mode).")
    print()

    print(f"Performance & Stability:") 
    print(f"* Processing ratio: ~15s audio per 1s CPU time (300 GFLOPs, 4-core, no viz).") 
    print(f"* Platform Gap: works ~1.7x faster in Prooted Debian than in Termux (Eigen BLAS).")  
    print(f"* OOM Safety: Close browsers or restart whole phone if crashes occur in Termux.")  
    print()

    print(f"Split Suggestion:") 
    print(f"If the file is too long, use FFmpeg to segment it first:") 
    print(f"mkdir split_input_media && cd split_input_media && \\")
    print(f"ffmpeg -i {audio_path_hint} -c copy -f segment -segment_time 1200 output_%03d.mp4")
    print()

    print("Tips & Environment Hacks:")
    print("* For 'undefined symbol: torch_library_impl' or 'NotImplementedError':")
    print("  Run: pip install -U torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu")
    print("* If you see 'NotImplementedError: sys.platform = android' after an update:")
    print(f"  Edit: /data/data/com.termux/files/usr/lib/python{py_ver}/site-packages/torchaudio/_internally_replaced_utils.py")
    print("  Change 'if sys.platform == \"linux\":' to 'if sys.platform == \"android\":'")
    print()

    print(f"Dependency Versions:")
    print(f"MoviePy: {moviepy.__version__}")
    print(f"Torchaudio: {torchaudio.__version__}")
    # May need to be disabled as it errors if installed some weird version 0 dev. 
    print(f"Torchcodec: {torchcodec.__version__}") 
    print()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    audio_path = args.audio_path


    if args.mode == 'audio_tagging':
        audio_tagging(args)
    else:
        sound_event_detection(args)
