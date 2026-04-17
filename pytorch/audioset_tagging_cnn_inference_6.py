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
import platform
import soundfile as sf
#import coverage 

from moviepy import ImageClip, CompositeVideoClip, AudioFileClip, ColorClip, VideoClip
import json
import collections
import plotly.offline as pyo
from scipy.stats import entropy
import tempfile # Import tempfile for temporary file handling

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
        print(f"⏲  🗃️  File duration: \033[1;34m{duration_str}\033[0m")
        if fps:
            print(f"🮲  🗃️  Video FPS (avg): \033[1;34m{fps:.3f}\033[0m")
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
    checkpoint_name = os.path.basename(checkpoint_path)
    output_dir = os.path.join(audio_dir, f'{base_filename_for_dir}_{checkpoint_name}_audioset_tagging_cnn')
    create_folder(output_dir)

    # --- Copy AI analysis guide ---
    try:
        # Using a relative path from the script's location
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



        

    base_filename = f'{base_filename_for_dir}_audioset_tagging_cnn'
    fig_path = os.path.join(output_dir, 'eventogram.png')
    # --- Idempotency Check ---
    csv_path = os.path.join(output_dir, 'full_event_log.csv')
    if os.path.exists(csv_path):
        print(f"✅ Skipping {audio_path}, because {csv_path} already exists. Remove that file to reprocess that folder.")
        return

    # '''    
    disk_usage = shutil.disk_usage(audio_dir)
    if disk_usage.free < 1e9:
        print(f"\033[1;31mError: Insufficient disk space ({disk_usage.free / 1e9:.2f} GB free). Exiting.\033[0m")
        return
    #'''
    
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
    
    # Check if duration is problematic (None or 0)
    if duration is None or duration == 0:
        print(f"\033[1;33mWarning: Initial duration detection for {audio_path} returned {duration}. Attempting conversion to MP3.\033[0m")
        original_path_for_conversion = audio_path # Keep original path for ffmpeg input
        audio_dir = os.path.dirname(original_path_for_conversion)
        base_filename = get_filename(original_path_for_conversion)
        
        # New MP3 path in temporary directory
        mp3_path = os.path.join(tempfile.gettempdir(), f"{base_filename}_converted.mp3")

        try:
            print(f"\033[1;36mConverting {original_path_for_conversion} to {mp3_path} using ffmpeg...\033[0m")
            subprocess.run(
                ['ffmpeg', '-i', original_path_for_conversion, '-y', mp3_path],
                check=True, capture_output=True, text=True
            )
            print(f"\033[1;32mSuccessfully converted to {mp3_path}. Re-probing duration from the new MP3.\033[0m")
            
            # Update audio_path to point to the newly converted MP3 for all subsequent operations
            audio_path = mp3_path
            
            # Re-run duration detection on the converted file
            duration, video_fps, video_width, video_height = get_duration_and_fps(audio_path)
            
            if duration is None or duration == 0:
                print(f"\033[1;31mError: Could not determine duration for {audio_path}, even after conversion. Exiting.\033[0m")
                return
            else:
                print(f"\033[1;32mConversion successful, new duration detected: {duration} seconds.\033[0m")

        except subprocess.CalledProcessError as e:
            print(f"\033[1;31mError: FFmpeg conversion failed for {original_path_for_conversion}. Stderr: {e.stderr}\033[0m")
            return
        except Exception as e:
            print(f"\033[1;31mAn unexpected error occurred during conversion: {e}\033[0m")
            return

    # If we reach here, 'duration' is valid (either initially or after successful conversion).
    # Proceed with the rest of the script.
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
                    temp_video_path = os.path.join(tempfile.gettempdir(), f'temp_cfr_{get_filename(audio_path)}.mp4')
                    try:
                        target_fps = video_fps
                        if not target_fps or target_fps <= 0:
                            if r_fps > 0:
                                target_fps = r_fps
                                print(f"\033[1;33mWarning: avg_frame_rate is invalid, falling back to r_frame_rate: {target_fps:.3f}\033[0m")
                            else:
                                target_fps = output_fps
                                print(f"\033[1;33mWarning: Both avg_frame_rate and r_frame_rate are invalid, falling back to default FPS: {target_fps}\033[0m")
                        
                        subprocess.run([
                            'ffmpeg', '-loglevel', 'warning', '-i', audio_path, '-r', str(target_fps), '-fps_mode', 'cfr', '-c:a', 'aac', temp_video_path, '-y'
                        ], check=True)
                        video_input_path = temp_video_path
                        print(f"Re-encoded to: \033[1;34m{temp_video_path}\033[1;0m")
                    except subprocess.CalledProcessError as e:
                        print(f"\033[1;31mError during VFR-to-CFR conversion: {e}\033[0m")
                        return

    try:
        waveform, sr = torchaudio.load(video_input_path)
    except Exception as e:
        print(f"\033[1;33mWarning: Failed to load audio directly ({e}). Attempting FFmpeg sanitization...\033[0m")
        sanitized_temp_path = os.path.join(tempfile.gettempdir(), f'sanitized_{get_filename(audio_path)}.wav')
        try:
            subprocess.run([
                'ffmpeg', '-loglevel', 'error', '-i', video_input_path, '-vn', '-acodec', 'pcm_s16le', sanitized_temp_path, '-y'
            ], check=True)
            
            # Try torchaudio again on the sanitized file
            try:
                waveform, sr = torchaudio.load(sanitized_temp_path)
            except Exception as e2:
                # If torchaudio still fails (common in Proot/Termux without backends), use soundfile
                print(f"\033[1;33mWarning: torchaudio failed on sanitized file ({e2}). Falling back to soundfile...\033[0m")
                waveform_np, sr = sf.read(sanitized_temp_path)
                # Convert soundfile's [samples, channels] to torch's [channels, samples]
                waveform = torch.from_numpy(waveform_np).float()
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.T
            
            print(f"✅ Sanitization and loading successful. Loaded audio from: \033[1;34m{sanitized_temp_path}\033[1;0m")
            video_input_path = sanitized_temp_path
        except subprocess.CalledProcessError as f_err:
            print(f"\033[1;31mError: FFmpeg sanitization failed: {f_err}\033[0m")
            return
        except Exception as generic_err:
            print(f"\033[1;31mError: Unexpected failure during audio loading: {generic_err}\033[0m")
            return

    import gc
    print(f"Loaded waveform shape: {waveform.shape}, sample rate: {sr}")
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
        gc.collect() # Clear resampling workspace
        
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
    waveform_np = waveform.squeeze(0).numpy()  # Convert to numpy for STFT
    
    # --- CRITICAL: Delete torch tensors to free ~1-2GB of RAM immediately ---
    del waveform
    gc.collect()
    waveform = waveform_np 
    print(f"Processed waveform shape: {waveform.shape}")

    chunk_duration = 180  # 3 minutes
    chunk_samples = int(chunk_duration * sample_rate)
    
    # --- MEMORY OPTIMIZATION: Prepare for streaming and downsampling ---
    vis_fps = 2  # Target 2 FPS for visualization as suggested
    frames_per_second = sample_rate // hop_size
    vis_downsample = frames_per_second // vis_fps
    
    framewise_vis_list = []
    stft_vis_list = []
    
    print(f"📊  Processing in {chunk_duration/60}m chunks. CSV: {frames_per_second} FPS | RAM/Vis: {vis_fps} FPS")

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time'] + list(labels))
        
        for start in range(0, len(waveform), chunk_samples):
            chunk = waveform[start:start + chunk_samples]
            if len(chunk) < sample_rate // 10:
                continue
            
            # 1. Inference
            chunk_tensor = move_data_to_device(torch.from_numpy(chunk[None, :]).float(), device)
            with torch.no_grad():
                model.eval()
                try:
                    batch_output_dict = model(chunk_tensor, None)
                    chunk_out = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
                except Exception as e:
                    print(f"\033[1;31mError in chunk at {start}: {e}\033[0m")
                    continue
            
            # 2. Stream downsampled CSV - keeps ALL classes but makes file ~20x smaller
            chunk_start_time = start / sample_rate
            csv_downsample = max(1, frames_per_second // args.csv_fps)  # e.g. 100//5 = 20
            for i in range(0, len(chunk_out), csv_downsample):
                timestamp = chunk_start_time + (i / frames_per_second)
                writer.writerow([round(timestamp, 3)] + chunk_out[i].tolist())
            
            # 3. Downsample for RAM-based Visualization (Max-Pool to preserve peaks)
            # We use max-pooling so we don't miss short sound events in the graph
            for i in range(0, len(chunk_out), vis_downsample):
                vis_slice = chunk_out[i : i + vis_downsample]
                if len(vis_slice) > 0:
                    framewise_vis_list.append(np.max(vis_slice, axis=0))

            # 4. Chunked STFT for Visualization
            chunk_tensor_stft = torch.from_numpy(chunk).to(device)
            chunk_stft = torch.stft(
                chunk_tensor_stft, n_fft=window_size, hop_length=hop_size,
                window=torch.hann_window(window_size).to(device),
                center=True, return_complex=True
            ).abs().cpu().numpy()
            
            # Downsample STFT columns
            stft_vis_list.append(chunk_stft[:, ::vis_downsample])
            
            print(f"Chunk at {int(chunk_start_time/60)}m done. RAM usage: {len(framewise_vis_list)} vis-frames cached.")

    # 5. Global Aggregation (Lean)
    framewise_output = np.array(framewise_vis_list) # Rename for compatibility with subsequent blocks
    stft = np.concatenate(stft_vis_list, axis=1) # Correctly concatenate STFT chunks
    del framewise_vis_list
    del stft_vis_list
    del waveform
    
    # Update frame metadata for downsampled resolution
    frames_per_second = vis_fps 
    frames_num = len(framewise_output)
    
    print(f'Aggregation complete. Internal RAM resolution: \033[1;34m{frames_per_second} FPS\033[1;0m')
    # Static PNG visualization
    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]
    top_k = 10
    top_result_mat = framewise_output[:, sorted_indexes[0:top_k]]
    top_labels = np.array(labels)[sorted_indexes[0:top_k]]

    fig_width_px = 1280
    fig_height_px = 480
    dpi = 100
    fig = plt.figure(figsize=(fig_width_px / dpi, fig_height_px / dpi), dpi=dpi)
    
    # 1. First Pass: Create plot to measure Y-axis label widths
    gs_init = fig.add_gridspec(2, 1, height_ratios=[1, 1], left=0.1, right=1.0, top=0.95, bottom=0.08, hspace=0.05)
    axs_init = [fig.add_subplot(gs_init[0]), fig.add_subplot(gs_init[1])]
    axs_init[1].set_yticks(np.arange(0, top_k))
    axs_init[1].set_yticklabels(top_labels, fontsize=14)
    fig.canvas.draw()
    
    renderer = fig.canvas.get_renderer()
    max_label_width_px = 0
    for lbl in axs_init[1].yaxis.get_majorticklabels():
        bbox = lbl.get_window_extent(renderer=renderer)
        w = bbox.width
        if w > max_label_width_px:
            max_label_width_px = w

    pad_px = 8
    left_margin_px = int(max_label_width_px + pad_px + 6)
    fig_w_in = fig.get_size_inches()[0]
    fig_w_px = fig_w_in * dpi
    left_frac = left_margin_px / fig_w_px
    if left_frac < 0: left_frac = 0.0
    if left_frac > 0.45: left_frac = 0.45
    
    # 2. Second Pass: Final Render with correct dynamic margin
    fig.clear()
    print(f'Computed left margin: \033[1;34m{left_margin_px}px\033[1;0m (frac: {left_frac:.3f})')
    
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], left=left_frac, right=1.0, top=0.95, bottom=0.08, hspace=0.05)
    axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]

    axs[0].matshow(np.log(stft + 1e-10), origin='lower', aspect='auto', cmap='jet')
    axs[0].set_ylabel('Frequency bins', fontsize=14)
    axs[0].set_title('Spectrogram and Eventogram', fontsize=14)
    axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)

    tick_interval = max(5, int(duration / 20))
    x_ticks = np.arange(0, frames_num, frames_per_second * tick_interval)
    x_labels = np.arange(0, int(duration) + 1, tick_interval)
    axs[1].xaxis.set_ticks(x_ticks)
    axs[1].xaxis.set_ticklabels(x_labels[:len(x_ticks)], rotation=45, ha='right', fontsize=10)
    axs[1].set_xlim(0, frames_num)
    
    axs[1].set_yticks(np.arange(0, top_k))
    axs[1].set_yticklabels(top_labels, fontsize=14)
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds', fontsize=14)
    axs[1].xaxis.set_ticks_position('bottom')
    
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f'Saved visualization to: \033[1;34m{fig_path}\033[1;0m')




    # --- AI-Friendly Summary Generation (Event Block Detection) ---
    print("📊  Generating AI-friendly event summary files…")

    # 1. Generate summary_events.csv
    summary_csv_path = os.path.join(output_dir, 'summary_events.csv')
    
    # Set thresholds to a very low value to capture all activity for the top classes
    onset_threshold = 0.01  # Probability to start an event
    offset_threshold = 0.01 # Probability to end an event
    min_event_duration_seconds = 0.5 # Minimum duration to be considered a valid event
    top_n_per_class = 3 # Get the top N most intense events for each sound class

    # Only track events for the global top_labels (from eventogram)
    events_by_class = {label: [] for label in top_labels}

    # Find event blocks for each sound class, focusing only on top_labels
    for label_idx, label in enumerate(labels):
        if label not in top_labels: # Skip classes not in the global top_labels for the eventogram
            continue
        
        in_event = False
        event_start_frame = 0
        
        for frame_index, prob in enumerate(framewise_output[:, label_idx]):
            if not in_event and prob > onset_threshold:
                in_event = True
                event_start_frame = frame_index
            elif in_event and prob < offset_threshold:
                in_event = False
                event_end_frame = frame_index
                
                duration_frames = event_end_frame - event_start_frame
                duration_seconds = duration_frames / frames_per_second
                
                if duration_seconds >= min_event_duration_seconds:
                    event_block_probs = framewise_output[event_start_frame:event_end_frame, label_idx]
                    
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

    # --- Detailed AI-Friendly Event Delta Map (optimized for Attention Mechanisms) ---
    print("📊  Generating detailed AI-friendly event delta JSON (with RLE compression)…")
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
                    zipped.append({"skip": zero_count})
                    zero_count = 0
                zipped.append(val)
        if zero_count > 0:
            zipped.append({"skip": zero_count})
        return zipped

    for label_idx, label in enumerate(labels):
        current_event = None
        probs_stream = framewise_output[:, label_idx]
        
        for frame_index, prob in enumerate(probs_stream):
            if prob > ai_threshold:
                if current_event is None:
                    current_event = {
                        "start": round(frame_index / frames_per_second, 3),
                        "end": round(frame_index / frames_per_second, 3),
                        "peak": float(prob),
                        "trace": [float(prob)]
                    }
                else:
                    current_event["end"] = round(frame_index / frames_per_second, 3)
                    current_event["peak"] = max(current_event["peak"], float(prob))
                    current_event["trace"].append(float(prob))
            else:
                if current_event:
                    # Finalize event and calculate deltas
                    tr = current_event["trace"]
                    deltas = [tr[0]] # Anchor
                    for i in range(1, len(tr)):
                        deltas.append(round(tr[i] - tr[i-1], 6))
                    
                    events_derivative[label].append({
                        "start_time": current_event["start"],
                        "end_time": current_event["end"],
                        "peak_prob": current_event["peak"],
                        "delta_trace": zip_trace(deltas)
                    })
                    current_event = None
        
        if current_event: # Finalize if trailing
            tr = current_event["trace"]
            deltas = [tr[0]]
            for i in range(1, len(tr)):
                deltas.append(round(tr[i] - tr[i-1], 6))
            events_derivative[label].append({
                "start_time": current_event["start"],
                "end_time": current_event["end"],
                "peak_prob": current_event["peak"],
                "delta_trace": zip_trace(deltas)
            })

    with open(json_ai_path, 'w') as f:
        json.dump({k: v for k, v in events_derivative.items() if v}, f, indent=2)
    print(f'Saved AI-friendly delta JSON (compressed) to: \033[1;34m{json_ai_path}\033[1;0m')

    # --- Interactive Plotly Dashboard (Top 50 Events) ---
    print("📊  Generating interactive Plotly dashboard…")
    html_dashboard_path = os.path.join(output_dir, 'interactive_dashboard.html')
    
    # 1. Identify top 50 classes by popularity (sum of probabilities)
    popularity = np.sum(framewise_output, axis=0)
    top_50_indices = np.argsort(popularity)[::-1][:50]
    
    # 2. Extract and optimize data (round to 4 decimals to save space)
    times = [round(i / frames_per_second, 2) for i in range(frames_num)]
    traces_data = []
    for idx in top_50_indices:
        probs = np.round(framewise_output[:frames_num, idx], 4).tolist()
        traces_data.append({
            "name": labels[idx],
            "y": probs
        })
    
    json_data = json.dumps({"times": times, "traces": traces_data})
    plotly_js = pyo.get_plotlyjs()
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Audio Analysis Dashboard - {base_filename_for_dir}</title>
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
        <h1>Sound Event Analysis: {base_filename_for_dir}</h1>
        <p>Interactive Plotly Dashboard | Top 50 Classes | Source: <code>full_event_log.csv</code></p>
    </div>
    <div id="plot"></div>
    <script type="text/javascript">
        const dataPayload = {json_data};
        const traces = dataPayload.traces.map((t, index) => ({{
            x: dataPayload.times,
            y: t.y,
            name: t.name,
            mode: 'lines',
            visible: index < 10 ? true : 'legendonly',
            line: {{ width: 2 }}
        }}));
        const layout = {{
            title: 'Top 50 Sound Events Momentum',
            xaxis: {{ title: 'Seconds', gridcolor: '#eee' }},
            yaxis: {{ title: 'Probability', range: [0, 1], gridcolor: '#eee' }},
            hovermode: 'x unified',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: {{ t: 50, b: 50, l: 60, r: 20 }}
        }};
        Plotly.newPlot('plot', traces, layout, {{ responsive: true }});
    </script>
</body>
</html>
"""
    with open(html_dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f'Saved interactive dashboard to: \033[1;34m{html_dashboard_path}\033[1;0m')

    """
    # Auto-open the dashboard using the system 'open' command
    try:
        subprocess.run(['open', html_dashboard_path], check=False)
    except Exception as e:
        print(f"Warning: Could not auto-open dashboard: {e}")
    """

    


    # Video rendering
    # Force output FPS to 30 for faster rendering and smaller files
    output_fps = 30

    if args.dynamic_eventogram:
        output_video_path = os.path.join(output_dir, f"{base_filename}_eventogram_dynamic.mp4")
        print(f"🎞  Rendering dynamic eventogram video at {output_fps} FPS…")
        
        window_duration = args.window_duration
        window_frames = int(window_duration * frames_per_second)
        half_window_frames = window_frames // 2

        # 1. Precompute ALL unique data windows at the data's native resolution (frames_per_second, e.g., 2 FPS)
        # This eliminates redundant KL divergence calculations and top-event searches.
        print(f"📊  Precomputing data windows for {frames_num} unique time points…")
        precomputed_data = []
        for i in range(frames_num):
            current_frame = i
            start_f = max(0, current_frame - half_window_frames)
            end_f = min(frames_num, current_frame + half_window_frames)
            
            # Adaptive window size (if enabled)
            if args.use_adaptive_window:
                kl_threshold = 0.5
                # Look back and ahead to find acoustic boundaries
                for offset in range(half_window_frames, half_window_frames + int(30 * frames_per_second), int(frames_per_second)):
                    if start_f - offset >= 0:
                        prev_prob = np.mean(framewise_output[start_f - offset:start_f], axis=0)
                        curr_prob = np.mean(framewise_output[start_f:start_f + offset], axis=0)
                        kl_div = compute_kl_divergence(prev_prob, curr_prob)
                        if kl_div > kl_threshold:
                            start_f = max(0, start_f - offset // 2)
                            break
                    if end_f + offset < frames_num:
                        curr_prob = np.mean(framewise_output[end_f - offset:end_f], axis=0)
                        next_prob = np.mean(framewise_output[end_f:end_f + offset], axis=0)
                        kl_div = compute_kl_divergence(curr_prob, next_prob)
                        if kl_div > kl_threshold:
                            end_f = min(frames_num, end_f + offset // 2)
                            break
            
            window_out, window_idxs = get_dynamic_top_events(framewise_output, start_f, end_f, top_k)
            if window_out.size == 0:
                window_out = np.zeros((end_f - start_f, top_k))
                window_idxs = sorted_indexes[:top_k]
                
            precomputed_data.append({
                'start_f': start_f,
                'end_f': end_f,
                'window_out': window_out,
                'window_idxs': window_idxs
            })

        # 2. Optimized Frame Generation with Caching
        # We cache the last rendered image. Since the video FPS (30) is much higher than 
        # the data FPS (2), most calls to make_frame will return the cached image.
        frame_cache = {"last_i": -1, "last_img": None}

        def make_frame(t):
            # Map video time to our data frame index
            i = int(t * frames_per_second)
            i = min(i, frames_num - 1)
            
            # If the data frame hasn't changed, return the cached image immediately
            if i == frame_cache["last_i"]:
                return frame_cache["last_img"]
            
            # Retrieve precomputed window data
            data = precomputed_data[i]
            start_f, end_f = data['start_f'], data['end_f']
            window_output, window_indexes = data['window_out'], data['window_idxs']
            
            # Create the figure using OO API for better performance/safety
            fig_fr = plt.figure(figsize=(fig_width_px / dpi, fig_height_px / dpi), dpi=dpi)
            gs_fr = fig_fr.add_gridspec(2, 1, height_ratios=[1, 1], left=left_frac, right=1.0, 
                                        top=0.95, bottom=0.08, hspace=0.05)
            axs_fr = [fig_fr.add_subplot(gs_fr[0]), fig_fr.add_subplot(gs_fr[1])]

            # Calculate a stable timestamp for the title
            t_stable = i / frames_per_second
            
            # Render Spectrogram for window
            stft_window = stft[:, start_f:end_f]
            axs_fr[0].matshow(np.log(stft_window + 1e-10), origin='lower', aspect='auto', cmap='jet')
            axs_fr[0].set_ylabel('Frequency bins', fontsize=14)
            axs_fr[0].set_title(f'Spectrogram and Eventogram (t={t_stable:.1f}s)', fontsize=14)

            # Render Eventogram for window
            axs_fr[1].matshow(window_output.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
            window_labels = np.array(labels)[window_indexes]
            axs_fr[1].yaxis.set_ticks(np.arange(0, top_k))
            axs_fr[1].yaxis.set_ticklabels(window_labels, fontsize=14)
            axs_fr[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
            axs_fr[1].set_xlabel('Seconds', fontsize=14)
            axs_fr[1].xaxis.set_ticks_position('bottom')

            # Adjust x-axis ticks to show absolute time in seconds
            window_seconds = (end_f - start_f) / frames_per_second
            tick_interval_window = max(1, int(window_seconds / 5))
            x_ticks_window = np.arange(0, end_f - start_f + 1, frames_per_second * tick_interval_window)
            x_labels_window = np.arange(int(start_f / frames_per_second), 
                                        int(end_f / frames_per_second) + 1, 
                                        tick_interval_window)
            
            for ax in axs_fr:
                ax.xaxis.set_ticks(x_ticks_window)
                ax.xaxis.set_ticklabels(x_labels_window[:len(x_ticks_window)], rotation=45, ha='right', fontsize=10)
                ax.set_xlim(0, end_f - start_f)

            # Add marker (at the current frame's position relative to the window)
            marker_x = i - start_f
            for ax in axs_fr:
                ax.axvline(x=marker_x, color='red', linewidth=2, alpha=0.8)

            # Draw and convert to image
            fig_fr.canvas.draw()
            img = np.frombuffer(fig_fr.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape((fig_height_px, fig_width_px, 4))[:, :, :3]  # Drop alpha
            plt.close(fig_fr)
            
            # Update cache
            frame_cache["last_i"] = i
            frame_cache["last_img"] = img
            return img

        # Use VideoClip to generate the dynamic video
        eventogram_clip = VideoClip(make_frame, duration=duration)
        audio_clip = AudioFileClip(video_input_path)
        eventogram_with_audio_clip = eventogram_clip.with_audio(audio_clip)
        eventogram_with_audio_clip.fps = output_fps

        eventogram_with_audio_clip.write_videofile(
            output_video_path,
            codec="libx264",
            fps=output_fps,
            threads=os.cpu_count()
        )
        print(f"🎹 Saved the dynamic eventogram video to: \033[1;34m{output_video_path}\033[1;0m")

    if args.static_eventogram:    
        print(f"🎞  Rendering static eventogram video …")
        output_video_path = os.path.join(output_dir, f'{base_filename}_eventogram_static.mp4')
        
        # 1. Load the base static image as a NumPy array once
        # Using the actual generated PNG ensures we have the correct final dimensions and margins.
        base_img = plt.imread(fig_path)
        # Handle alpha if present and convert to uint8 (0-255)
        if base_img.dtype == np.float32:
            base_img = (base_img * 255).astype(np.uint8)
        if base_img.shape[2] == 4:
            base_img = base_img[:, :, :3]
            
        h, w, _ = base_img.shape
        x_start = int(left_frac * w)
        x_end = w
        
        # 2. Optimized Frame Generation with 2 FPS Marker Caching
        # Jerkiness is accepted for speed; marker only moves when data 'ticks'.
        frame_cache_static = {"last_i": -1, "last_img": None}

        def make_frame_static(t):
            # Map video time to our data frame index (2 FPS)
            i = int(t * frames_per_second)
            i = min(i, frames_num - 1)
            
            # Return cached image if marker hasn't moved
            if i == frame_cache_static["last_i"]:
                return frame_cache_static["last_img"]
            
            # Calculate marker position based on the data frame index
            frac = i / max(frames_num - 1, 1)
            marker_x = int(x_start + (x_end - x_start) * frac)
            
            # Create frame by drawing a 2px red line on a copy of the base image
            img = base_img.copy()
            # Draw marker (vertical red line)
            m_left = max(0, marker_x - 1)
            m_right = min(w, marker_x + 1)
            img[:, m_left:m_right] = [255, 0, 0] # Red marker
            
            # Update cache
            frame_cache_static["last_i"] = i
            frame_cache_static["last_img"] = img
            return img

        # Use VideoClip with the caching function
        static_eventogram_clip = VideoClip(make_frame_static, duration=duration)
        audio_clip = AudioFileClip(video_input_path)
        eventogram_with_audio_clip = static_eventogram_clip.with_audio(audio_clip)
        eventogram_with_audio_clip.fps = output_fps

        eventogram_with_audio_clip.write_videofile(
            output_video_path,
            codec="libx264",
            fps=output_fps,
            threads=os.cpu_count()
        )
        print(f"🎹 Saved the static eventogram video to: \033[1;34m{output_video_path}\033[1;0m")

    if (args.dynamic_eventogram or args.static_eventogram):
        if is_video:
            print("🎬  Overlaying the source media with the created eventogram…")
            root, ext = os.path.splitext(output_video_path)
            final_output_path = f"{root}_overlay{ext}"
            _, _, base_w, base_h = get_duration_and_fps(audio_path)
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
                "-c:a", "aac",
                "-shortest",
                final_output_path
            ])

            try:
                subprocess.run(overlay_cmd, check=True)
                print(f"✅ 🎥 The overlaid video has been saved to: \033[1;34m{final_output_path}\033[1;0m")
            except subprocess.CalledProcessError as e:
                print(f"\033[1;31mError during FFmpeg overlay: {e}\033[0m")
                return
            
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except:
                    pass
        else:
            print("🎧 Source is audio-only, no overlay video created.")


    
    print(f"⏲  🗃️  Reminder: input file duration: \033[1;34m{duration}\033[0m")

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Audio tagging and Sound event detection.')
    parser.add_argument('audio_path', type=str, help='Path to the media file')
    parser.add_argument('--mode', choices=['audio_tagging', 'sound_event_detection'],
                        default='sound_event_detection', help='Select the processing mode')
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
                        help='FPS to write to full_event_log.csv (default 5). '
                             'Use 100 for full resolution (very large file). '
                             '5 is enough for Shapash + outlier detection.')
                        

    args = parser.parse_args()
    
    audio_path = args.audio_path  # <-- extract it
    #Hard code the output's frequency:
    output_fps = 25

    print(f"Eventogrammer, version 6.7.2. Material Changes: \n * Broken the 'Aggregation Bottleneck': High-res data (100 FPS) is now streamed directly to disk (CSV) during inference.\n * 50x RAM Optimization: Internal RAM structures (Eventogram/Spectrogram) are now max-pooled to 2 FPS. ")
    
    # --- ECHO INFO SECTION: ANDROID PLATFORM HACK ---

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
 
    print(f"Notes: The processing time ratio now is: 15 second of the orignal duration takes 1 seconds to process on a regular 300 GFLOPs, 4 core CPU.") 
    print(f"If the file is too long, use e.g. this to split:") 
    print(f"mkdir split_input_media && cd split_input_media && ffmpeg -i {audio_path} -c copy -f segment -segment_time 1200 output_%03d.mp4")
    

 
    print(f"This script is an adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn so see there if something be amiss.")
    print(f"Note on the models:")   
    print(f"* `Cnn14_DecisionLevelMax_mAP=0.385.pth` (Sound Event Detection): This model uses Decision-level pooling. It calculates classification probabilities for every small segment of time first, and only then takes the maximum probability to represent the whole clip. Resolution: Cnn14_DecisionLevelMax is specifically designed for Sound Event Detection (SED). Because it maintains the time dimension through the classifier, it can output the framewise_output used by the inference script to generate the 'Eventograms' and the CSV logs.")
    print(f"* The other models are good for audio tagging: use the '--audio_tagging' switch for that mode.")
    print()
    print(f"Note on speed: works 1.7 times faster in Prooted Debian than in Termux, see the comments why so.")  
    print(f"Note on the out of memory crashes: close all other programs in Droid, especially the browser. Or just restart whole phone. Or do it in Recovery.")  
    # In Termux your PyTorch build is explicitly: USE_EIGEN_FOR_BLAS=ON. It means: PyTorch tensor ops that would normally dispatch to BLAS/LAPACK are not using an external high-performance BLAS backend at all. They are using Eigen’s generic CPU kernels compiled into PyTorch. That decision is compile-time, not runtime. In practice: same workload, same model family, same CPU class, different environments, and we observed a consistent ~1.7× gap in wall time (Termux ~14 min vs Debian ~8 min).
    
    
    

    print("Tips: 'undefined symbol: torch_library_impl' or 'NotImplementedError':")
    print("This is often a version mismatch between torch and torchaudio, simply run:")
    print("pip install -U torch torchaudio --extra-index-url https://download.pytorch.org/whl/cpu")
    print("pip install -U torchcodec --extra-index-url https://download.pytorch.org/whl/cpu")
    print("In e.g. Prooted Debian under Termux, torchcodec has dependencies on NVIDA .so files and the script errors, so you may need to git clone and pip install it from scratch (which works without a hitch) then.")

    print(f"If you see 'NotImplementedError: sys.platform = android' after an update:")
    print(f"1. Edit: /data/data/com.termux/files/usr/lib/python{py_ver}/site-packages/torchaudio/_internally_replaced_utils.py")
    print("2. Change 'if sys.platform == \"linux\":' to 'if sys.platform == \"android\":' - it works.")
    

    

    print(f"Using moviepy version: {moviepy.__version__}")
    print(f"Using torchaudio version: {torchaudio.__version__}")
    # May need to be disabled as it errors if installed some weird version 0 dev. 
    print(f"Using torchcodec version: {torchcodec.__version__}") 


    
    

    print(f"")

    if args.mode == 'audio_tagging':
        audio_tagging(args)
    else:
        sound_event_detection(args)
