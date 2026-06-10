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
#Torch and torchaudio and coverage are version sensitive.  Use apt for that if you can or just remove coverage.

import torch
# Handle version-sensitive imports: Torchaudio is essential for tensor-land processing and CUDA efficiency.
try:
    import torchaudio
except (OSError, ImportError) as e:
    print(f"\033[1;31mERROR: torchaudio and torch are not compatible ({e}). We stop.\033[0m")
    print("Please synchronize your versions to fix the 'undefined symbol' or import error:")
    print("\033[1;32mpip install -U torch torchaudio torchcodec --extra-index-url https://download.pytorch.org/whl/cpu\033[0m")
    sys.exit(1)

try:
    import torchcodec
except (OSError, ImportError, RuntimeError):
    torchcodec = None
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
import warnings
import soundfile as sf
import psutil
#import coverage 

import h5py
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
    Extract duration, FPS, resolution, is_video flag, and native sample rate from a media file using FFprobe.
    Returns: (duration, avg_fps, width, height, is_video, r_fps, native_sr)
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
        
        duration, avg_fps, width, height, r_fps, native_sr = None, None, None, None, None, None
        is_video = any(s['codec_type'] == 'video' and s.get('codec_name') not in ['mjpeg', 'png'] for s in streams)

        # Try format duration first (most reliable)
        if format_info.get('duration'):
            try:
                duration = float(format_info['duration'])
            except (ValueError, TypeError):
                print(f"\033[1;33mWarning: Invalid format duration in {input_media_path}\033[0m")

        # Get native sample rate from audio stream
        audio_stream = next((s for s in streams if s['codec_type'] == 'audio'), None)
        if audio_stream and audio_stream.get('sample_rate'):
            native_sr = int(audio_stream['sample_rate'])

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

        return duration, avg_fps, width, height, is_video, r_fps, native_sr

    except Exception as e:
        print(f"Error probing media: {e}")
        return None, None, None, None, False, None, None

    # --- USER INTERFACE & LOGGING ---

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
    h5_path = os.path.join(output_dir, 'full_event_log.h5')
    
    # Efficiently probe all metadata once
    duration, video_fps, video_width, video_height, is_video, r_fps, native_sr = get_media_metadata(source_media)
    if duration is None: return # Error already printed in probe

    # Define requested video outputs
    video_outputs = []
    if args.static_eventogram:
        vid_path = os.path.join(output_dir, f'{base_name}{tag_suffix}_eventogram_static.mp4')
        video_outputs.append(f"{os.path.splitext(vid_path)[0]}_overlay.mp4" if is_video else vid_path)
    if args.dynamic_eventogram:
        vid_path = os.path.join(output_dir, f'{base_name}{tag_suffix}_eventogram_dynamic.mp4')
        video_outputs.append(f"{os.path.splitext(vid_path)[0]}_overlay.mp4" if is_video else vid_path)

    # Partial Skip Logic: If HDF5 exists, we can skip inference
    skip_inference = os.path.exists(h5_path)
    
    # Full Skip Logic: If CSV AND all requested videos exist
    if skip_inference and all(os.path.exists(f) for f in video_outputs):
        print(f"✅ Skipping {source_media}, all requested outputs already exist in: \033[1;34m{output_dir}\033[1;0m")
        print(f"To start anew, execute e.g.: \033[1;31m rm -rf \"{output_dir}\"\033[1;0m")
        return

    # Check for sufficient disk space
    disk_usage = shutil.disk_usage(audio_dir)
    if disk_usage.free < 1e9:
        print(f"\033[1;31mError: Insufficient disk space ({disk_usage.free / 1e9:.2f} GB free). Exiting.\033[0m")
        return
    
    # --- PHASE 3: Model Loading (Only if inference needed) ---
    model = None
    if not skip_inference:
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

    # --- PHASE 4: Media Integrity & Recovery (Sanitary Gate) ---
    sf_formats = sf.available_formats()
    has_mp3_support = 'MP3' in sf_formats
    source_ext = os.path.splitext(source_media)[1][1:].upper()
    
    # Always re-encode to CBR MP3 to fix seekability/drift issues
    print(f"🎬  Re-encoding input to CBR MP3 for seeking stability...")
    temp_audio_path = os.path.join(tempfile.gettempdir(), f'temp_cbr_{base_name}.mp3')
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', source_media, '-c:a', 'libmp3lame', '-b:a', '41k', temp_audio_path, '-y'], check=True)
    inference_media = temp_audio_path
    # Refresh duration and metadata for the sanitized file
    duration, video_fps, video_width, video_height, is_video, r_fps, native_sr = get_media_metadata(inference_media)

    # Fallback Recovery: If duration is still 0 (e.g. corrupt header), attempt emergency recovery
    if duration == 0:
        print(f"\033[1;33mWarning: Duration probe failed for {inference_media}. Attempting emergency recovery...\033[0m")
        mp3_path = os.path.join(tempfile.gettempdir(), f"{base_name}_recovered.mp3")

        try:
            subprocess.run(['ffmpeg', '-i', inference_media, '-y', mp3_path], check=True, capture_output=True)
            inference_media = mp3_path # Rest of the script uses this recovered file
            duration, video_fps, video_width, video_height, is_video, r_fps, native_sr = get_media_metadata(inference_media)
            
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
            # --- FFmpeg Compatibility Note ---
            # We use '-vsync 1' (Constant Frame Rate) instead of the newer '-fps_mode cfr'.
            # Rationale: '-fps_mode' was introduced in FFmpeg 5.1 (late 2022). 
            # Many stable systems (like Ubuntu 20.04) still use FFmpeg 4.x, which only 
            # recognizes '-vsync'. While '-vsync' is deprecated in FFmpeg 5.1+, it 
            # remains a functional alias in FFmpeg 6.x/7.x. 
            # If a future FFmpeg version (e.g., v10+) removes '-vsync', switch this back 
            # to: '-fps_mode', 'cfr'
            subprocess.run([
                'ffmpeg', '-loglevel', 'warning', '-i', inference_media, '-r', str(target_fps), '-vsync', '1', '-c:a', 'aac', temp_video_path, '-y'
            ], check=True)
            inference_media = temp_video_path

    # --- PHASE 6: Metadata Probe (Surgical Load Preparation) ---
    # Get Native Sample Rate via ffprobe (O(1) memory safety)
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate", "-of", "default=noprint_wrappers=1:nokey=1", inference_media]
    try:
        native_sr = int(subprocess.check_output(cmd).decode().strip())
    except Exception:
        # Fallback if ffprobe fails
        _, native_sr = torchaudio.load(inference_media, frame_offset=0, num_frames=1)
        
    native_num_frames = int(duration * native_sr)
    
    # --- USER INTERFACE & LOGGING ---
    duration_str = str(datetime.timedelta(seconds=int(duration))) if duration else "?"
    print(f"⏲  🗃️  File duration: \033[1;34m{duration_str}\033[0m")
    
    # Calculate chunk size in native samples
    chunk_duration = 180  # 3 minutes
    native_chunk_samples = int(chunk_duration * native_sr)

    # Aesthetic Decoupling: Cap RAM aggregation arrays to ~2500 columns for O(1) memory
    max_vis_cols = 2500
    potential_vis_frames = int(duration * viz_fps)
    vis_agg_factor = max(1, int(np.ceil(potential_vis_frames / max_vis_cols)))
    
    # Store the original viz_fps for display, but update viz_fps for internal logic
    effective_viz_fps = viz_fps / vis_agg_factor
    vis_downsample = max(1, int(inference_fps / effective_viz_fps))

    # Pre-allocate fixed-size buffers for O(1) memory management
    num_viz_frames = int(duration * effective_viz_fps) + 1
    framewise_viz_buffer = np.zeros((num_viz_frames, len(labels)), dtype=np.float32)
    current_viz_row = 0
    stft_vis_list = []

    # --- PHASE 7: Chunked Model Inference (Memory-Safe & Surgical) ---
    avail_ram = psutil.virtual_memory().available / (1024 * 1024)
    print(f"📊  Starting decoupled inference in {chunk_duration/60}m chunks. (RAM Avail: {avail_ram:.0f} MB)")
    print(f"    Native SR: {native_sr} Hz | Inference SR: {sample_rate} Hz")
    if vis_agg_factor > 1:
        print(f"💡 Aesthetic Decoupling: Aggregating {vis_agg_factor}x into {max_vis_cols} RAM columns.")

    resampler = None
    if native_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=native_sr, new_freq=sample_rate).to(device)

    if not skip_inference:
        # Pre-calculate total rows to enable pre-allocation
        total_rows_est = int(np.ceil(duration * args.csv_fps))
        
        # Initialize HDF5 with NO compression during the loop to avoid single-threaded stalls.
        # Pre-allocating the shape prevents expensive resizing during inference.
        h5_file = h5py.File(h5_path, 'w')
        h5_dataset = h5_file.create_dataset('framewise_output', 
                                            shape=(total_rows_est, len(labels)), 
                                            maxshape=(None, len(labels)), 
                                            dtype='float32', compression='gzip')
        h5_file.create_dataset('labels', data=np.array(labels, dtype='S'))
        h5_stft_viz = h5_file.create_dataset('stft_viz',
                                             shape=(513, 0),
                                             maxshape=(513, None),
                                             dtype='float32', compression='gzip')
        h5_timestamps = h5_file.create_dataset('timestamps', 
                                               shape=(total_rows_est,), 
                                               maxshape=(None,), 
                                               dtype='float32', compression='gzip')

        from torchcodec.decoders import AudioDecoder
        decoder = AudioDecoder(inference_media, sample_rate=sample_rate, num_channels=1)
        
        current_row = 0
        chunk_seconds = native_chunk_samples / native_sr
        print(f"DEBUG: Native Chunk Samples={native_chunk_samples}, Native SR={native_sr}, Chunk Seconds={chunk_seconds}")
        for i in range(0, int(duration), int(chunk_seconds)):
            # Use torchcodec to get samples in range
            start_s = float(i)
            stop_s = float(i + chunk_seconds)
            
            # get_samples_played_in_range returns [channels, samples]
            samples = decoder.get_samples_played_in_range(start_seconds=start_s, stop_seconds=stop_s).data
            
            print(f"DEBUG: Range [{start_s}, {stop_s}], Raw Samples Shape: {samples.shape}")

            # Ensure samples is [1, samples] (Mono)
            # torchcodec returns [channels, samples]. Mean over channels, then unsqueeze(0) for batch dim.
            if samples.dim() > 1:
                # If [channels, samples], mean over dim 0 (channels) -> [samples]
                # If [samples, channels] (unlikely), mean over dim 1 -> [samples]
                # Based on torchcodec docs, shape is [C, T]
                samples = samples.mean(dim=0)
            
            # Now [samples], need [1, samples]
            chunk_waveform = samples.unsqueeze(0).to(device)
            print(f"DEBUG: Processed Chunk Shape: {chunk_waveform.shape}")

            # Step B: Inference
            with torch.no_grad():
                model.eval()
                batch_output_dict = model(chunk_waveform, None)
                chunk_out = batch_output_dict['framewise_output'].data.cpu().numpy()[0]

            # Step C: Write results to disk (OOM-safe, vectorized streaming to HDF5)
            chunk_start_time = float(i) # Use loop variable 'i'
            csv_downsample = max(1, inference_fps // args.csv_fps)

            downsampled_data = chunk_out[::csv_downsample]
            num_rows = downsampled_data.shape[0]
            downsampled_timestamps = np.array([chunk_start_time + (i / inference_fps) 
                                               for i in range(0, len(chunk_out), csv_downsample)], dtype='float32')

            # Ensure pre-allocation was enough (safety check)
            if current_row + num_rows > h5_dataset.shape[0]:
                h5_dataset.resize(current_row + num_rows + 1000, axis=0)
                h5_timestamps.resize(current_row + num_rows + 1000, axis=0)

            # Fast Vectorized Write (No compression / No flush = High CPU utilization)
            h5_dataset[current_row : current_row + num_rows] = downsampled_data.astype('float32')
            h5_timestamps[current_row : current_row + num_rows] = downsampled_timestamps

            current_row += num_rows
            
            # Step D: Downsample for RAM-based visualization (Max-pooling preserves short events)
            for i in range(0, len(chunk_out), vis_downsample):
                vis_slice = chunk_out[i : i + vis_downsample]
                if len(vis_slice) > 0:
                    if current_viz_row < len(framewise_viz_buffer):
                        framewise_viz_buffer[current_viz_row] = np.max(vis_slice, axis=0)
                        current_viz_row += 1

            # Step E: STFT for visualization background
            chunk_numpy = chunk_waveform.squeeze(0).cpu().numpy()
            chunk_tensor_stft = torch.from_numpy(chunk_numpy).to(device)
            chunk_stft = torch.stft(
                chunk_tensor_stft, n_fft=window_size, hop_length=hop_size,
                window=torch.hann_window(window_size).to(device),
                center=True, return_complex=True
            ).abs().cpu().numpy()
            
            # Save downsampled STFT to HDF5
            downsampled_stft = chunk_stft[:, ::vis_downsample].astype('float32')
            h5_stft_viz.resize(h5_stft_viz.shape[1] + downsampled_stft.shape[1], axis=1)
            h5_stft_viz[:, -downsampled_stft.shape[1]:] = downsampled_stft


            # Cleanup
            del chunk_waveform, chunk_numpy, chunk_tensor_stft, chunk_stft, chunk_out, batch_output_dict, downsampled_data
            import gc
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            avail_ram = psutil.virtual_memory().available / (1024 * 1024)
            print(f"Chunk at {int(chunk_start_time/60)}m finished. (RAM Avail: {avail_ram:.0f} MB)")
            
        # Trim dataset to final size
        h5_dataset.resize(current_row, axis=0)
        h5_timestamps.resize(current_row, axis=0)
        h5_file.close()

    # --- PHASE 8: Result Aggregation & Metadata Consolidation ---
    if not skip_inference:
        # Use only the filled portion of the pre-allocated buffer
        framewise_output = framewise_viz_buffer[:current_viz_row]
        stft = h5py.File(h5_path, 'r')['stft_viz'][:] # Load from HDF5 instead
        del framewise_viz_buffer
    elif skip_inference:
        # Load data from HDF5 if inference was skipped
        with h5py.File(h5_path, 'r') as hf:
            framewise_output = hf['framewise_output'][:]
            stft = hf['stft_viz'][:]
    else:
        print("Error: Inference returned no data. Check input file or model.")
        return

    if framewise_output.size == 0:
        print("Error: No event data detected.")
        return

    # DATA-DRIVEN DURATION: Use real data length as truth (fixes VBR/probe guesses)
    frames_num = len(framewise_output)
    duration = frames_num / effective_viz_fps
    
    print(f'Aggregation complete. Internal Viz resolution: \033[1;34m{effective_viz_fps:.4f} Hz (Data Frames)\033[1;0m')
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
    x_ticks = np.arange(0, frames_num, effective_viz_fps * tick_interval)
    axs[1].xaxis.set_ticks(x_ticks)
    axs[1].xaxis.set_ticklabels([int(t / effective_viz_fps) for t in x_ticks], rotation=45, ha='right', fontsize=10)
    axs[1].set_xlim(0, frames_num); axs[1].set_yticks(np.arange(0, top_k)); axs[1].set_yticklabels(top_labels, fontsize=14)
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds', fontsize=14); axs[1].xaxis.set_ticks_position('bottom')
    
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0); plt.close(fig)
    print(f'Saved eventogram PNG to: \033[1;34m{fig_path}\033[1;0m')




    # --- PHASE 10: AI-Friendly Summary Generation (Event Block Detection) ---
    print("📊  Generating AI-friendly event summary files…")
    summary_csv_path = os.path.join(output_dir, 'summary_events.csv')
    
    # Load data from HDF5
    with h5py.File(h5_path, 'r') as hf:
        framewise_output = hf['framewise_output'][:]
        timestamps = hf['timestamps'][:]
        
    # Event detection heuristics
    onset_threshold, offset_threshold = 0.01, 0.01
    min_event_duration_seconds = 0.5 
    top_n_per_class = 3 # Only export the strongest N events per class

    events_by_class = {label: [] for label in top_labels}

    # Identify continuous blocks of sound for the top classes shown in the eventogram
    for label_idx, label in enumerate(labels):
        if label not in top_labels: continue
        
        in_event, event_start_idx = False, 0
        for idx, prob in enumerate(framewise_output[:, label_idx]):
            if not in_event and prob > onset_threshold:
                in_event, event_start_idx = True, idx
            elif in_event and prob < offset_threshold:
                in_event = False
                
                start_time = timestamps[event_start_idx]
                end_time = timestamps[idx]
                duration_secs = end_time - start_time
                
                if duration_secs >= min_event_duration_seconds:
                    event_block_probs = framewise_output[event_start_idx:idx, label_idx]
                    events_by_class[label].append({
                        'sound_class': label,
                        'start_time_seconds': round(float(start_time), 3),
                        'end_time_seconds': round(float(end_time), 3),
                        'duration_seconds': round(float(duration_secs), 3),
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
    
    # We already loaded framewise_output in Phase 10
    
    # Compute derivative (momentum of change)
    delta_output = np.diff(framewise_output, axis=0, prepend=0)

    def zip_trace(trace):
        """Compress trace by replacing consecutive zeros with a skip marker."""
        zipped = []
        zero_count = 0
        for val in trace:
            if abs(val) < ai_threshold:
                zero_count += 1
            else:
                if zero_count > 0:
                    zipped.append({"skip": zero_count}); zero_count = 0
                zipped.append(round(float(val), 4))
        if zero_count > 0: zipped.append({"skip": zero_count})
        return zipped

    for label_idx, label in enumerate(labels):
        current_event = None
        probs_stream = framewise_output[:, label_idx]
        
        for frame_index, prob in enumerate(probs_stream):
            if prob > ai_threshold:
                if current_event is None:
                    current_event = {
                        "start": round(frame_index / effective_viz_fps, 3),
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
                    "end_time": round(frame_index / effective_viz_fps, 3),
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
                "end_time": round(len(probs_stream) / effective_viz_fps, 3),
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
    
    times = [round(i / effective_viz_fps, 2) for i in range(frames_num)]
    traces_data = []
    for idx in top_50_indices:
        traces_data.append({"name": labels[idx], "y": np.round(framewise_output[:frames_num, idx], 4).tolist()})
    
    # Build tick labels — timedelta gives "H:MM:SS" for <24h
    max_time = max(times) if times else 0
    tick_interval = max(30, int(max_time / 20))
    tick_vals = list(range(0, int(max_time) + tick_interval, tick_interval))
    tick_text = [str(datetime.timedelta(seconds=v)) for v in tick_vals]

    json_payload = json.dumps({
        "times": times,
        "traces": traces_data,
        "tick_vals": tick_vals,
        "tick_text": tick_text
    })
    plotly_js = pyo.get_plotlyjs()
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Audio Analysis - {base_name}</title>
    <script type="text/javascript">{plotly_js}</script>
    <style>
        body {{ font-family: sans-serif; margin: 20px; background: #fafafa; color: #333; box-sizing: border-box; }}
        #plot {{ width: 100%; height: 50vh; background: white; border-radius: 8px; border: 1px solid #ddd; margin-bottom: 20px; }}
        .header {{ margin-bottom: 15px; border-left: 5px solid #2563eb; padding-left: 15px; }}
        code {{ background: #eee; padding: 2px 5px; border-radius: 4px; }}
        #mediaPlayer {{ width: 100%; max-width: 100%; display: block; box-sizing: border-box; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Sound Event Analysis: {base_name}</h1>
        <p>Interactive Plotly Dashboard | Top 50 Classes | Source: <code>full_event_log.h5</code></p>
    </div>
    <video controls id="mediaPlayer" src="{audio_path}"></video>
    <div id="plot"></div>
    <script type="text/javascript">
        const data = {json_payload};
        const traces = data.traces.map((t, index) => ({{
            x: data.times, y: t.y, name: t.name, mode: 'lines',
            visible: index < 10 ? true : 'legendonly', line: {{ width: 2 }}
        }}));
        Plotly.newPlot('plot', traces, {{
            title: 'Top 50 Sound Events Momentum',
            xaxis: {{
                title: 'Time (MM:SS)',
                tickvals: data.tick_vals,
                ticktext: data.tick_text,
                hoverformat: '.1f',
                gridcolor: '#eee'
            }},
            yaxis: {{ title: 'Probability', range: [0, 1], gridcolor: '#eee' }},
            hovermode: 'x unified', paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            margin: {{ t: 50, b: 50, l: 60, r: 20 }}
        }}, {{ responsive: true }});

        const mediaPlayer = document.getElementById('mediaPlayer');
        if (mediaPlayer) {{
            document.getElementById('plot').on('plotly_click', function(data){{
                if(data.points.length > 0) {{
                    const clickedTime = data.points[0].xaxis.d2l(data.points[0].x);
                    mediaPlayer.currentTime = clickedTime;
                    mediaPlayer.play();
                }}
            }});
        }}
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
            window_frames = int(args.window_duration * effective_viz_fps)
            half_window = window_frames // 2 

            # PRECOMPUTE: Map every data point to its local acoustic window (once per run)
            print(f"📊  Precomputing {frames_num} windows (Adaptive={args.use_adaptive_window})…")
            precomputed_data = []
            for i in range(frames_num):
                start_f, end_f = max(0, i - half_window), min(frames_num, i + half_window)
                
                # Adaptive logic: Try to center the window on acoustic "peaks" rather than strict time
                if args.use_adaptive_window:
                    kl_threshold = 0.5
                    lookahead_f = int(adaptive_lookahead * effective_viz_fps)
                    for offset in range(half_window, half_window + lookahead_f, int(max(1, effective_viz_fps))):
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
                t_start, t_end, t_curr = s_f / effective_viz_fps, e_f / effective_viz_fps, i / effective_viz_fps
                
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
                c_w, c_h = fig_fr.canvas.get_width_height()
                return np.frombuffer(fig_fr.canvas.buffer_rgba(), dtype=np.uint8).reshape((c_h, c_w, 4))[:,:,:3]

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
            i = min(int(t * effective_viz_fps), frames_num - 1)
            if i == frame_cache["last_i"]: return frame_cache["last_img"]
            
            img = draw_strategy(i)
            frame_cache.update({"last_i": i, "last_img": img})
            return img

        # --- DYNAMIC DIMENSION PROBE ---
        # Claude's Fix: Since 'bbox_inches=tight' changes the PNG size, we probe the 
        # first frame to get the 'Visual Truth' before opening the pipe.
        first_frame = make_frame(0)
        v_h, v_w, _ = first_frame.shape
        
        # Ensure even dimensions for yuv420p compatibility (FFmpeg requirement)
        v_w, v_h = v_w // 2 * 2, v_h // 2 * 2
        
        # Final Compositing & Export (Direct FFmpeg Pipe - Zero Temp Files)
        total_output_frames = int(duration * output_fps)
        
        # FFmpeg setup: Pipe 0 is raw video frames; Input 1 is the original audio
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{v_w}x{v_h}",
            "-pix_fmt", "rgb24", "-r", str(output_fps),
            "-i", "pipe:0",          # Raw frames from Python
            "-i", inference_media,   # Audio from source
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "copy",          # 1:1 Audio Stream Copy
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", output_video_path, "-loglevel", "warning"
        ]
        
        print(f"🎬  Encoding video via direct pipe ({v_w}x{v_h}, {total_output_frames} frames)...")
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        try:
            for frame_idx in range(total_output_frames):
                t = frame_idx / output_fps
                frame = make_frame(t)
                
                # Surgical Crop: Ensure buffer matches the 'even' dimensions promised to FFmpeg
                if frame.shape[0] > v_h or frame.shape[1] > v_w:
                    frame = frame[:v_h, :v_w, :]
                
                ffmpeg_proc.stdin.write(frame.astype(np.uint8).tobytes())
                
                if frame_idx % (output_fps * 10) == 0:
                    percent = (frame_idx / total_output_frames) * 100
                    print(f"   Progress: {percent:.1f}% ({t:.0f}s / {duration:.0f}s)", end="\r")
            
            print(f"   Progress: 100.0% ({duration:.0f}s / {duration:.0f}s)")
        finally:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()

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
                "-c:a", "copy", "-shortest"
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
    parser.add_argument('--no-shapash', action='store_true', default=False,
                        help='Skip Shapash dashboard launch')
                        
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
 
    print(f"Eventogrammer, version 6.12.2") 
    print(f"Adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn")
    print(f"Recent Material Changes:")
    print(f"* Media player added to plotly graph. ")   
    print(f"* H5py used instead of CSV to save disk space.")
    print(f"* We convert all files to MP3")
    print(f"* Load: Memory-safe chunked decoding (OOM Fix for 10h+ files).")
    print(f"* Constants Promoted: vis_fps, output_fps, and adaptive_lookahead are now CLI arguments.")
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
    print(f"time bash -c 'for file in ""*; do bash audio_me.sh \"$file\" --dynamic_eventogram; done'")
    print()

    print("Tips & Environment Hacks:")
    print("* For 'undefined symbol' (e.g. torch_from_blob), 'Invalid argument' crashes, or SIGABRT:")
    print("  Root Cause: ABI Collision. You likely have a mix of 'apt' (Debian) and 'pip' (Official) Torch packages.")
    print("  The C++ backend becomes a 'hollow shell'—importing the Python part works, but touching the C++ core kills the process.")
    print("  Fix: Uninstall ALL apt versions and 'pip install -U' the whole stack together to ensure binary alignment:")
    print("  Run: pip install -U torch torchaudio torchcodec --extra-index-url https://download.pytorch.org/whl/cpu")
    print("* If Torchcodec errors (e.g., 'libnvrtc.so.13 not found'):")
    print("  Root Cause: CUDA Ghosting. Standard wheels link to NVIDIA libs even on CPU-only setups.")
    print("  Fix: Use the --extra-index-url above to force CPU-only binaries, or: pip uninstall torchcodec")
    print("* If you see 'NotImplementedError: sys.platform = android' after an update:")
    print(f"  Edit: /data/data/com.termux/files/usr/lib/python{py_ver}/site-packages/torchaudio/_internally_replaced_utils.py")
    print("  Change 'if sys.platform == \"linux\":' to 'if sys.platform == \"android\":'")
    print("* If some coverage numba error: do 'apt remove python3-coverage'. Be careful with the below python modules if they have parallel apt based install versions, use one or the other then: 'python -m pip install torch torchaudio torchcodec --upgrade --extra-index-url https://download.pytorch.org/whl/cpu ' : prefer their apt versions")
    
    print("* If: 'LibsndfileError: File contains data in an unimplemented format', then 'git clone https://github.com/libsndfile/libsndfile.git' and install it in e.g. Termux. Or run in Proot.")  

    print()

    print(f"Dependency Versions:")
    print(f"Torch: {torch.__version__ if torch else 'Not Available'}")
    print(f"Torchaudio: {torchaudio.__version__ if torchaudio else 'Not Available'}")
    # May need to be disabled as it errors if installed some weird version 0 dev.
    print(f"Torchcodec: {torchcodec.__version__ if torchcodec else 'Not Available'}")
 
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

 
    print()

