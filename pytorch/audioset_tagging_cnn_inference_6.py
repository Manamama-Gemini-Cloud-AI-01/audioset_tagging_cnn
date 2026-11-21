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
import atexit
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
    print("Please synchronize your versions to fix the 'undefined symbol' or import error, torchcodec : 'pip index versions torchcodec' or:")
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
import gc
import time
import subprocess
import shutil
import warnings
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

# ---------------------------------------------------------------------------
# STAGES — named, ordered pipeline stages, used only for log banners.
#
# Numbers were dropped on purpose: they encoded history (a reordered dashboard
# left "PHASE 12" between "PHASE 9" and "PHASE 10"), so renumbering was a
# manual chore every time a stage moved. Keys are stable identifiers; the
# banner number is computed from this list, so reordering is a one-line edit.
# The label is what you quote when reporting a broken run.
# ---------------------------------------------------------------------------
STAGES = [
    ("setup",       "Setup & Environment"),
    ("idempotency", "Idempotency Check"),
    ("model",       "Model Loading"),
    ("sanitize",    "Media Integrity & Recovery (Sanitary Gate)"),
    ("cfr",         "VFR Check & Constant Frame Rate"),
    ("probe",       "Metadata Probe"),
    ("inference",   "Chunked Model Inference"),
    ("aggregate",   "Result Aggregation"),
    ("eventogram",  "Static Eventogram (PNG)"),
    ("dashboard",   "Interactive Plotly Dashboard"),
    ("summary",     "AI-Friendly Event Summary"),
    ("delta",       "Detailed Event Delta Map"),
    ("render",      "Video Rendering"),
]
_STAGE_INDEX = {key: i for i, (key, _) in enumerate(STAGES)}


def stage(key):
    """Print the banner for a named stage. See STAGES above."""
    i = _STAGE_INDEX[key]
    print(f"\n\033[1;36m[{i + 1}/{len(STAGES)}] {STAGES[i][1]}\033[0m")


def stage_skip(key, reason="skipped"):
    """Print a named stage's banner with a skip note, so the log still shows the slot."""
    i = _STAGE_INDEX[key]
    print(f"\n\033[1;36m[{i + 1}/{len(STAGES)}] {STAGES[i][1]} — {reason}\033[0m")


# ---------------------------------------------------------------------------
# THRESHOLDS — the single home for "what counts as a real event".
#
# These numbers used to be scattered across five places (PNG, CSV, delta JSON,
# dashboard payload, hover JS) with no shared definition, so "significance"
# meant something slightly different at each consumer. One dict, one place to
# audit. Keys are named by ROLE, not by value: two consumers that happen to
# share 0.05 today are still separate decisions and can drift apart on purpose.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    # Static eventogram PNG
    "eventogram_top_k": 10,          # rows drawn in the PNG heatmap
    # summary_events.csv (block segmentation)
    "csv_include_peak": 0.02,        # class enters the CSV if its peak reaches this
    "csv_onset": 0.01,               # block starts when probability reaches this
    "csv_offset": 0.01,              # block ends when probability drops below this
    "csv_min_seconds": 0.5,          # blocks shorter than this are dropped
    "csv_top_per_class": 3,          # strongest blocks exported per class
    # detailed_events_delta_ai_attention_friendly.json
    "delta_include_prob": 0.05,      # frame is recorded above this; below is skipped
    # interactive_dashboard.html
    "dashboard_visible_peak": 0.05,  # trace drawn by default if its peak reaches this
    "hover_floor": 0.005,            # hover rows below this are treated as silence
    "hover_rows": 12,                # hover rows shown before the "… N more" line
}

# Sanitized audio is transient. The CBR MP3 is only needed while the model or a
# video render reads it, so it is written to the system temp directory and deleted
# at the end of the run. No cache: the transcode takes seconds, and a cached copy
# would just waste space in the artifact folder. A dashboard-only rerun skips the
# transcode entirely instead of reusing a cache.
def build_sanitized_audio(source_path, base_name):
    """Transcode the source to a transient CBR MP3 and return its path."""
    audio_path = os.path.join(tempfile.gettempdir(), f'temp_cbr_{base_name}.mp3')
    register_scratch(audio_path)
    # `-f mp3` is explicit: ffmpeg infers the muxer from the extension, and a
    # suffix without a known format fails with "Unable to find a suitable output
    # format" (the reason the first version of this cache builder crashed).
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', source_path,
                    '-c:a', 'libmp3lame', '-b:a', '41k', '-f', 'mp3', audio_path, '-y'], check=True)
    return audio_path


# Scratch files are intermediates, never artifacts. They are registered as they
# are created and removed at exit as well as at the end of the happy path, so a
# crash mid-run cannot leave a multi-GB CFR video, CBR MP3 or recovery file
# behind. Nothing here is cached between runs.
_SCRATCH_FILES = []


def register_scratch(path):
    """Track an intermediate file for removal at exit; returns the path."""
    if path:
        _SCRATCH_FILES.append(path)
    return path


def cleanup_scratch():
    """Remove every registered intermediate that still exists."""
    for path in _SCRATCH_FILES:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError as e:
            print(f"\033[1;33mWarning: could not remove {path}: {e}\033[0m")


atexit.register(cleanup_scratch)


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

def h5_master_key(hf):
    """Name of the master probability grid ('framewise_ui' is the legacy spelling)."""
    return 'framewise_ui' if 'framewise_ui' in hf else 'framewise_output'

def h5_has_payload(h5_path):
    """True only for a *finished* event log.

    A run that dies between ``h5py.File(..., 'w')`` and the first successful
    ``create_dataset()`` leaves a valid but empty HDF5 file (superblock + empty
    root group, ~800 bytes, no datasets). Treating that as "inference already
    done" turned the *first* crash into a murky
    ``KeyError: "object 'framewise_output' doesn't exist"`` on the next run, so
    only skip inference when every dataset we are going to read is there and
    non-empty. Anything else is treated as a stale artifact and recomputed.
    """
    try:
        with h5py.File(h5_path, 'r') as hf:
            key = h5_master_key(hf)
            if any(k not in hf for k in (key, 'stft_viz', 'timestamps')):
                return False
            return hf[key].shape[0] > 0 and hf['stft_viz'].shape[1] > 0
    except OSError:
        return False

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
    stage('setup')
    video_fps = args.video_fps
    ui_fps = args.ui_fps  # Fixed constant for internal resolution
    adaptive_lookahead = args.adaptive_lookahead
    
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    inference_fps = sample_rate // hop_size # Typically 100
    
    top_k = THRESHOLDS['eventogram_top_k']  # Number of top events to track/visualize
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

    tag_suffix = "_audioset_tagging_cnn"

    # Every artifact a run produces, video or not. Declared here because the idempotency
    # check below has to know about all of them, not only the videos.
    h5_path = os.path.join(output_dir, 'full_event_log.h5')
    fig_path = os.path.join(output_dir, 'eventogram.png')
    html_dashboard_path = os.path.join(output_dir, 'interactive_dashboard.html')
    summary_csv_path = os.path.join(output_dir, 'summary_events.csv')
    json_ai_path = os.path.join(output_dir, 'detailed_events_delta_ai_attention_friendly.json')

    stage('idempotency')
    
    # Efficiently probe all metadata once
    duration, probed_video_fps, video_width, video_height, is_video, r_fps, native_sr = get_media_metadata(source_media)
    if duration is None: return # Error already printed in probe

    # Define requested video outputs
    video_outputs = []
    if args.static_eventogram:
        vid_path = os.path.join(output_dir, f'{base_name}{tag_suffix}_eventogram_static.mp4')
        video_outputs.append(f"{os.path.splitext(vid_path)[0]}_overlay.mp4" if is_video else vid_path)
    if args.dynamic_eventogram:
        vid_path = os.path.join(output_dir, f'{base_name}{tag_suffix}_eventogram_dynamic.mp4')
        video_outputs.append(f"{os.path.splitext(vid_path)[0]}_overlay.mp4" if is_video else vid_path)

    # Partial Skip Logic: only a finished HDF5 lets us skip the model pass.
    h5_exists = os.path.exists(h5_path)
    skip_inference = h5_exists and h5_has_payload(h5_path)
    if h5_exists and not skip_inference:
        print(f"\033[1;33mWarning: existing {h5_path} is empty or incomplete "
              f"(a previous run died before finishing it). Re-running inference.\033[0m")

    # Full Skip Logic: only when everything this run would produce already exists.
    # Checking the videos alone was a footgun: with no video flags `all([])` is True, so any
    # rerun against an existing HDF5 exited silently without regenerating the dashboard,
    # the summary CSV, the PNG or the JSON.
    expected_outputs = video_outputs + [h5_path, fig_path, html_dashboard_path,
                                        summary_csv_path, json_ai_path]
    if skip_inference and all(os.path.exists(f) for f in expected_outputs):
        print(f"✅ Skipping {source_media}, all requested outputs already exist in: \033[1;34m{output_dir}\033[1;0m")
        print(f"To start anew, execute e.g.: \033[1;31m rm -rf \"{output_dir}\"\033[1;0m")
        return

    # Check for sufficient disk space
    disk_usage = shutil.disk_usage(audio_dir)
    if disk_usage.free < 1e9:
        print(f"\033[1;31mWarning: Insufficient disk space ({disk_usage.free / 1e9:.2f} GB free). It may fail...\033[0m")
        #return
    
    model = None
    if not skip_inference:
        stage('model')
        Model = eval(model_type)
        model = Model(sample_rate=sample_rate, window_size=window_size, 
                      hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                      classes_num=classes_num)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            model.to(device)
        except Exception as e:
            print(f"\033[1;31mError loading model checkpoint: {e}\033[0m")
            return
    else:
        stage_skip('model', 'skipped; probabilities come from the existing HDF5')

    stage('sanitize')
    duration, probed_video_fps, video_width, video_height, is_video, r_fps, native_sr = get_media_metadata(inference_media)

    # Audio is only needed when the model runs or when a video is rendered. A
    # dashboard/CSV-only rerun needs neither, so it must not pay for a full
    # transcode (that was the "a dashboard tweak costs six minutes" bug).
    needs_audio = (not skip_inference) or bool(video_outputs)

    # CBR MP3: the chunked decoder needs consistent seekability, so the source is
    # normalised to a 41 kbps CBR MP3 for the duration of this run. The file is
    # transient (system temp, deleted at exit), not a cache: the transcode takes
    # seconds, and a stored copy would waste space in the artifact folder. A run
    # that skips inference and requests no video skips the transcode entirely.
    if not needs_audio:
        print("⏭  No inference and no video requested; skipping the audio transcode.")
    else:
        print(f"🎬  Re-encoding input to CBR MP3 for stability of the chunk seeking...")
        inference_media = build_sanitized_audio(source_media, base_name)

    # Fallback Recovery: If duration is still unknown/0 (e.g. corrupt header),
    # attempt emergency recovery. `not duration` also catches None, which would
    # otherwise blow up later as int(None).
    recovered_path = None
    if not duration:
        print(f"\033[1;33mWarning: Duration probe failed for {inference_media}. Attempting emergency recovery...\033[0m")
        recovered_path = register_scratch(os.path.join(tempfile.gettempdir(), f'{base_name}_recovered.mp3'))

        try:
            subprocess.run(['ffmpeg', '-i', inference_media, '-y', recovered_path], check=True, capture_output=True)
            inference_media = recovered_path # Rest of the script uses this recovered file
            duration, probed_video_fps, video_width, video_height, is_video, r_fps, native_sr = get_media_metadata(inference_media)
            
            if duration is None or duration == 0:
                print(f"\033[1;31mError: Could not determine duration even after recovery. Exiting.\033[0m")
                return
        except Exception as e:
            print(f"\033[1;31mRecovery conversion failed: {e}\033[0m")
            return

    if is_video and (video_width is None or video_height is None):
        video_width, video_height = 1280, 720
        print(f"\033[1;33mWarning: Video dimensions not detected, using default 1280x720.\033[0m")
    
    if needs_audio:
        stage('cfr')
        temp_video_path = None
        if is_video and r_fps and probed_video_fps:
            if abs(r_fps - probed_video_fps) > 0.01:
                print(f"\033[1;33mDetected VFR ({r_fps:.2f}/{probed_video_fps:.2f}). Re-encoding to CFR for sync...\033[0m")
                temp_video_path = register_scratch(os.path.join(tempfile.gettempdir(), f'temp_cfr_{base_name}.mp4'))
                target_fps = probed_video_fps if probed_video_fps > 0 else video_fps
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
    else:
        stage_skip('cfr', 'skipped; no audio needed')

    stage('probe')
    # Get Native Sample Rate via ffprobe (O(1) memory safety)
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate", "-of", "default=noprint_wrappers=1:nokey=1", inference_media]
    try:
        native_sr = int(subprocess.check_output(cmd).decode().strip())
    except Exception:
        # Fallback if ffprobe fails
        _, native_sr = torchaudio.load(inference_media, frame_offset=0, num_frames=1)

    # --- USER INTERFACE & LOGGING ---
    duration_str = str(datetime.timedelta(seconds=int(duration))) if duration else "?"
    print(f"⏲  🗃️  File duration: \033[1;34m{duration_str}\033[0m")
    
    # Calculate chunk size in native samples
    chunk_duration = 180  # 3 minutes
    native_chunk_samples = int(chunk_duration * native_sr)

    # One resolution for the whole pipeline: the max-pooled UI/analysis grid.
    # Derive the *actual* rate from the integer pooling factor, so a --ui_fps that
    # does not divide inference_fps (e.g. 3 Hz) cannot silently drift: 100/3 -> 33
    # frames = 3.0303 Hz, and every timestamp/x-axis/attr uses that same number.
    ui_downsample = max(1, round(inference_fps / ui_fps))
    effective_ui_fps = inference_fps / ui_downsample
    if abs(effective_ui_fps - ui_fps) > 1e-9:
        print(f"\033[1;33mNote: --ui_fps {ui_fps} Hz pools {ui_downsample} model frames "
              f"({inference_fps} Hz) -> effective {effective_ui_fps:.4f} Hz.\033[0m")

    # Pre-allocate fixed-size buffers for O(number of rows) memory management.
    # One extra chunk of headroom absorbs last-chunk padding from the model, so the
    # H5 master and this buffer stay row-for-row identical.
    num_ui_frames = int(duration * effective_ui_fps) + int(chunk_duration * effective_ui_fps) + 2
    framewise_ui_buffer = np.zeros((num_ui_frames, len(labels)), dtype=np.float32)
    current_ui_row = 0

    if not skip_inference:
        stage('inference')
        avail_ram = psutil.virtual_memory().available / (1024 * 1024)
        print(f"📊  Starting decoupled inference in {chunk_duration/60}m chunks. (RAM Avail: {avail_ram:.0f} MB)")
        print(f"    Native SR: {native_sr} Hz | Inference SR: {sample_rate} Hz")

        # Pre-calculate total rows to enable pre-allocation.
        # ONE resolution for the whole pipeline: the max-pooled UI/analysis grid.
        total_rows_est = int(np.ceil(duration * effective_ui_fps))

        # Chunked + LZF. h5py docs: lzf is "available with every installation of
        # h5py" and "very fast", while gzip/deflate needs a libhdf5 built with
        # zlib (Termux's is not, so `compression='gzip'` raises
        # ValueError: Compression filter "gzip" is unavailable).
        # Pre-allocating the shape prevents expensive resizing during inference.
        h5_file = h5py.File(h5_path, 'w')
        # UNIFIED MASTER DATASET: the single source of truth for both analysis and the dashboard.
        # float32, max-pooled, no threshold => faint/transient signals are preserved.
        h5_dataset = h5_file.create_dataset('framewise_output', 
                                            shape=(total_rows_est, len(labels)), 
                                            maxshape=(None, len(labels)), 
                                            dtype='float32', compression='lzf',
                                            chunks=(256, len(labels)))
        h5_file.create_dataset('labels', data=np.array(labels, dtype='S'))
        # STFT bins come from --window_size (n_fft), not from a hard-coded 513.
        stft_bins = window_size // 2 + 1
        h5_stft_viz = h5_file.create_dataset('stft_viz',
                                             shape=(stft_bins, 0),
                                             maxshape=(stft_bins, None),
                                             dtype='float32', compression='lzf')
        h5_timestamps = h5_file.create_dataset('timestamps', 
                                               shape=(total_rows_est,), 
                                               maxshape=(None,), 
                                               dtype='float32', compression='lzf')

        from torchcodec.decoders import AudioDecoder
        decoder = AudioDecoder(inference_media, sample_rate=sample_rate, num_channels=1)
        
        current_row = 0
        chunk_seconds = native_chunk_samples / native_sr
        print(f"DEBUG: Native Chunk Samples={native_chunk_samples}, Native SR={native_sr}, Chunk Seconds={chunk_seconds}")
        for chunk_offset in range(0, int(duration), int(chunk_seconds)):
            # Use torchcodec to get samples in range
            start_s = float(chunk_offset)
            stop_s = float(chunk_offset + chunk_seconds)
            
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
                model.to(device)
                model.eval()
                batch_output_dict = model(chunk_waveform, None)
                chunk_out = batch_output_dict['framewise_output'].data.cpu().numpy()[0]

            # Step C: Reduce raw model frames to the single master grid.
            # MAX-POOLING (not striding): a strided read can step over a short transient,
            # max-pooling keeps the peak of every window. No threshold, so faint signals survive.
            chunk_start_time = float(chunk_offset)
            pooled_rows = []
            pooled_timestamps = []
            for v_row_idx in range(0, len(chunk_out), ui_downsample):
                vis_slice = chunk_out[v_row_idx : v_row_idx + ui_downsample]
                if len(vis_slice) > 0:
                    pooled_rows.append(np.max(vis_slice, axis=0))
                    pooled_timestamps.append(chunk_start_time + (v_row_idx / inference_fps))

            if pooled_rows:
                downsampled_data = np.asarray(pooled_rows, dtype=np.float32)
            else:
                downsampled_data = np.empty((0, len(labels)), dtype=np.float32)
            num_rows = downsampled_data.shape[0]
            downsampled_timestamps = np.asarray(pooled_timestamps, dtype='float32')

            # Ensure pre-allocation was enough (safety check)
            if current_row + num_rows > h5_dataset.shape[0]:
                h5_dataset.resize(current_row + num_rows + 1000, axis=0)
                h5_timestamps.resize(current_row + num_rows + 1000, axis=0)

            # Write the SAME rows to disk and to the RAM buffer so they cannot diverge.
            # `n_write` is bounded by the buffer; with the headroom above this only
            # bites if the media duration was badly underestimated by the probe.
            n_write = min(num_rows, len(framewise_ui_buffer) - current_ui_row)
            if n_write < num_rows:
                print(f"\033[1;33mWarning: RAM buffer full; dropping {num_rows - n_write} "
                      f"trailing rows (probed duration was too short).\033[0m")
            if n_write > 0:
                h5_dataset[current_row : current_row + n_write] = downsampled_data[:n_write]
                h5_timestamps[current_row : current_row + n_write] = downsampled_timestamps[:n_write]
                framewise_ui_buffer[current_ui_row : current_ui_row + n_write] = downsampled_data[:n_write]
                current_row += n_write
                current_ui_row += n_write

            # Step E: STFT for visualization background
            chunk_numpy = chunk_waveform.squeeze(0).cpu().numpy()
            chunk_tensor_stft = torch.from_numpy(chunk_numpy).to(device)
            chunk_stft = torch.stft(
                chunk_tensor_stft, n_fft=window_size, hop_length=hop_size,
                window=torch.hann_window(window_size).to(device),
                center=True, return_complex=True
            ).abs().cpu().numpy()
            
            # Save downsampled STFT to HDF5
            downsampled_stft = chunk_stft[:, ::ui_downsample].astype('float32')
            h5_stft_viz.resize(h5_stft_viz.shape[1] + downsampled_stft.shape[1], axis=1)
            h5_stft_viz[:, -downsampled_stft.shape[1]:] = downsampled_stft


            # Cleanup
            del chunk_waveform, chunk_numpy, chunk_tensor_stft, chunk_stft, chunk_out, batch_output_dict, downsampled_data
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            avail_ram = psutil.virtual_memory().available / (1024 * 1024)
            print(f"Chunk at {int(chunk_start_time/60)}m finished. (RAM Avail: {avail_ram:.0f} MB)")
            
        # Trim dataset to final size
        h5_dataset.resize(current_row, axis=0)
        h5_timestamps.resize(current_row, axis=0)
        
        # Save visualization-specific data to HDF5 for persistent fidelity
        h5_file.attrs['effective_ui_fps'] = effective_ui_fps
        
        h5_file.close()

        # --- RE-POSITIONED: AI Guide Copy (Post-Inference) ---
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            guide_src_path = os.path.normpath(os.path.join(script_dir, '..', 'docs', 'auditory_cognition_guide_template.md'))
            guide_dest_path = os.path.join(output_dir, 'auditory_cognition_guide_template.md')
            
            if os.path.exists(guide_src_path):
                shutil.copy(guide_src_path, guide_dest_path)
                print(f'Copied AI analysis guide to: \033[1;34m{guide_dest_path}\033[1;0m')
        except Exception as e:
            print(f'\033[1;33mWarning: Failed to copy AI analysis guide: {e}\033[0m')
    else:
        stage_skip('inference', 'skipped; the HDF5 already holds the probabilities')

    stage('aggregate')
    if not skip_inference:
        # The in-RAM buffer holds the exact rows already written to the H5 master,
        # so use it directly instead of re-reading gigabytes from disk.
        framewise_output = framewise_ui_buffer[:current_ui_row]
        del framewise_ui_buffer
        with h5py.File(h5_path, 'r') as hf:
            stft = hf['stft_viz'][:]
    else:
        # Load data from HDF5 if inference was skipped (h5_has_payload() already
        # guaranteed these datasets exist and carry rows).
        with h5py.File(h5_path, 'r') as hf:
            # Restore the exact visualization parameters used during the original run
            if 'effective_ui_fps' in hf.attrs:
                effective_ui_fps = hf.attrs['effective_ui_fps']

            framewise_output = hf[h5_master_key(hf)][:]
            stft = hf['stft_viz'][:]

    if framewise_output.size == 0:
        print("Error: No event data detected.")
        return

    # DATA-DRIVEN DURATION: Use real data length as truth (fixes VBR/probe guesses)
    frames_num = len(framewise_output)
    duration = frames_num / effective_ui_fps
    
    print(f'Aggregation complete. Internal UI resolution: \033[1;34m{effective_ui_fps:.4f} Hz (Data Frames)\033[1;0m')
    print(f'Final analysis duration: \033[1;34m{duration:.2f}s\033[1;0m')

    stage('eventogram')
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
    x_ticks = np.arange(0, frames_num, effective_ui_fps * tick_interval)
    axs[1].xaxis.set_ticks(x_ticks)
    axs[1].xaxis.set_ticklabels([int(t / effective_ui_fps) for t in x_ticks], rotation=45, ha='right', fontsize=10)
    axs[1].set_xlim(0, frames_num); axs[1].set_yticks(np.arange(0, top_k)); axs[1].set_yticklabels(top_labels, fontsize=14)
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds', fontsize=14); axs[1].xaxis.set_ticks_position('bottom')
    
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0); plt.close(fig)
    print(f'Saved eventogram PNG to: \033[1;34m{fig_path}\033[1;0m')

    stage('dashboard')
    print("📊  Generating interactive Plotly dashboard (All Classes)…")

    # Pass all classes to allow forensic search, not just top 50
    times = [round(i / effective_ui_fps, 2) for i in range(frames_num)]
    traces_data = []
    for idx in range(len(labels)):
        traces_data.append({"name": labels[idx], "y": np.round(framewise_output[:frames_num, idx], 4).tolist()})

    # Build tick labels
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
    rel_audio_path = os.path.relpath(audio_path, output_dir)

    # Kept out of the f-string below on purpose: JS braces would otherwise all need
    # doubling. This is a plain string, injected as a single `{hover_tooltip_js}` slot.
    hover_tooltip_js = r"""
        // --- Custom hover tooltip: top-heavy, never clipped ---------------------------
        // The native 'x unified' legend renders one row per drawn trace. With 144 traces
        // it is ~3,000 px tall, far taller than the plot, and Plotly clips it to the plot
        // area: the visible slice is the *tail* of the list — the near-zero classes —
        // while the strongest events sit above the top edge. The native legend is hidden
        // in CSS and this handler draws a compact tooltip instead: the strongest classes
        // at the hovered frame, ranked descending (the "top-heavy" reading order).
        const HOVER_ROWS = TH.hover_rows;        // rows shown in the tooltip
        const HOVER_FLOOR = TH.hover_floor;      // ignore probabilities below this (noise floor)

        const hoverTip = document.createElement('div');
        hoverTip.id = 'hoverTip';
        hoverTip.style.cssText = [
            'min-width: 180px', 'max-width: 340px', 'padding: 6px 8px',
            'background: rgba(255,255,255,0.97)', 'border: 1px solid #cbd5e1',
            'border-radius: 6px', 'box-shadow: 0 2px 8px rgba(0,0,0,0.18)',
            'font: 11px/1.35 sans-serif', 'color: #333'
        ].join(';');
        plotElement.appendChild(hoverTip);

        const FRAME_STEP = data.times.length > 1 ? (data.times[1] - data.times[0]) : 1;
        const FRAME_LAST = data.times.length - 1;

        function frameIndex(t) {
            const i = Math.round((t - data.times[0]) / FRAME_STEP);
            return i < 0 ? 0 : (i > FRAME_LAST ? FRAME_LAST : i);
        }

        function clockLabel(t) {
            const s = Math.max(0, Math.round(t));
            const h = Math.floor(s / 3600);
            const m = Math.floor((s % 3600) / 60);
            const ss = String(s % 60).padStart(2, '0');
            return h > 0
                ? h + ':' + String(m).padStart(2, '0') + ':' + ss
                : String(m).padStart(2, '0') + ':' + ss;
        }

        function traceColour(k) {
            const fd = plotElement._fullData && plotElement._fullData[k];
            return (fd && fd.line && fd.line.color) || '#94a3b8';
        }

        plotElement.on('plotly_hover', function (ev) {
            const pt = ev.points && ev.points[0];
            if (!pt) return;
            const i = frameIndex(pt.x);

            // Rank every currently drawn trace by its probability at this frame.
            const rows = [];
            for (let k = 0; k < traces.length; k++) {
                const fd = plotElement._fullData && plotElement._fullData[k];
                if (fd && fd.visible !== true) continue;   // skip legendonly traces
                const v = traces[k].y[i];
                if (v >= HOVER_FLOOR) rows.push([v, k]);
            }
            rows.sort(function (a, b) { return b[0] - a[0] || a[1] - b[1]; });

            let html = '<div style="font-weight:600;color:#111;margin-bottom:4px">'
                     + clockLabel(pt.x) + '</div>';
            if (rows.length === 0) {
                html += '<div style="color:#94a3b8">nothing above ' + HOVER_FLOOR + '</div>';
            }
            for (let j = 0; j < rows.length && j < HOVER_ROWS; j++) {
                const v = rows[j][0], k = rows[j][1];
                html += '<div style="display:flex;align-items:center;gap:6px;white-space:nowrap">'
                     +  '<span style="flex:0 0 10px;height:2px;background:' + traceColour(k) + '"></span>'
                     +  '<span style="flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis">'
                     +  traces[k].name + '</span>'
                     +  '<span style="flex:0 0 auto;font-variant-numeric:tabular-nums;color:#475569">'
                     +  v.toFixed(3) + '</span>'
                     +  '</div>';
            }
            if (rows.length > HOVER_ROWS) {
                html += '<div style="margin-top:3px;color:#94a3b8">… '
                     +  (rows.length - HOVER_ROWS) + ' more ≥ ' + HOVER_FLOOR + '</div>';
            }
            hoverTip.innerHTML = html;
            hoverTip.style.display = 'block';

            // Park the tooltip beside the cursor, flipping/clamping inside the plot box.
            const box = plotElement.getBoundingClientRect();
            const e = ev.event || {};
            const cx = typeof e.clientX === 'number' ? e.clientX : box.left + box.width / 2;
            const cy = typeof e.clientY === 'number' ? e.clientY : box.top + box.height / 2;
            let left = cx - box.left + 16;
            let top = cy - box.top + 16;
            if (left + hoverTip.offsetWidth > box.width - 4) {
                left = cx - box.left - hoverTip.offsetWidth - 16;
            }
            if (top + hoverTip.offsetHeight > box.height - 4) {
                top = box.height - hoverTip.offsetHeight - 4;
            }
            hoverTip.style.left = Math.max(4, left) + 'px';
            hoverTip.style.top = Math.max(4, top) + 'px';
        });

        function hideHoverTip() { hoverTip.style.display = 'none'; }
        plotElement.on('plotly_unhover', hideHoverTip);
        plotElement.addEventListener('mouseleave', hideHoverTip);
    """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8" />
    <title>Audio Analysis - {base_name}</title>
    <script type="text/javascript">{plotly_js}</script>
    <style>
        body {{ font-family: sans-serif; margin: 20px; background: #fafafa; color: #333; }}
        #plot {{ position: relative; width: 100%; height: 60vh; background: white; border-radius: 8px; border: 1px solid #ddd; margin-bottom: 20px; }}
        /* The native "x unified" hover legend draws one row per drawn trace. With 100+
           traces it is several thousand pixels tall, far taller than the plot, and Plotly
           clips it to the plot area: what stays visible is the *tail* of the list — the
           near-zero classes — while the strongest events sit above the top edge. Hide it
           and draw a compact, top-heavy tooltip in JS instead (see the hover handler). */
        #plot .hoverlayer g.legend {{ display: none; }}
        #hoverTip {{ position: absolute; display: none; pointer-events: none; z-index: 20; }}
        #controls {{ margin-bottom: 15px; padding: 10px; background: #eee; border-radius: 8px; }}
        .header {{ margin-bottom: 15px; border-left: 5px solid #2563eb; padding-left: 15px; }}
        #mediaPlayer {{ width: 100%; max-width: 100%; display: block; }}
    </style>
    </head>
    <body>
    <div class="header">
        <h1>Sound Event Analysis: {base_name}</h1>
        <p>Interactive Plotly Dashboard | All Classes | Source: <code>full_event_log.h5</code></p>
    </div>
    <video controls id="mediaPlayer" src="{rel_audio_path}"></video>
    
    <div id="controls">
        <label>Search/Toggle Class: </label>
        <input type="text" id="classSearch" placeholder="Type to filter classes..." style="width: 200px;">
        <button id="showAll">Show All</button>
        <button id="hideAll">Hide All</button>
    </div>
    
    <div id="plot"></div>
    <script type="text/javascript">
        const data = {json_payload};
        // Thresholds live in one place in the Python source (THRESHOLDS) and are
        // injected here, so the dashboard can never drift from the CSV/JSON gates.
        const TH = {json.dumps(THRESHOLDS)};
        // Default visibility must be data-driven. An ontology index (index < 10) selects the
        // first ten AudioSet classes (all speech-related), NOT the ten strongest detections.
        const peaks = data.traces.map(t => t.y.reduce((m, v) => (v > m ? v : m), 0));
        const traces = data.traces.map((t, index) => ({{
            x: data.times, y: t.y, name: t.name, mode: 'lines',
            visible: peaks[index] >= TH.dashboard_visible_peak ? true : 'legendonly',
            line: {{ width: 2 }}
        }}));
        
        const plotElement = document.getElementById('plot');
        
        Plotly.newPlot('plot', traces, {{
            title: 'Sound Events Momentum',
            xaxis: {{
                title: 'Time (MM:SS)',
                tickvals: data.tick_vals,
                ticktext: data.tick_text,
                gridcolor: '#eee'
            }},
            yaxis: {{ title: 'Probability', range: [0, 1], gridcolor: '#eee' }},
            legend: {{ orientation: 'h', y: -0.2 }},
            // Unified hover supplies the spike line and the hover events; its own legend
            // is hidden by CSS (see the style block) because it cannot fit 100+ traces.
            // hoversort is kept so rows stay value-ranked if the legend is ever shown.
            hovermode: 'x unified', hoversort: 'value descending',
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            margin: {{ t: 50, b: 100, l: 60, r: 20 }}
        }}, {{ responsive: true }});

        // Search/Filter Logic
        document.getElementById('classSearch').addEventListener('input', (e) => {{
            const search = e.target.value.toLowerCase();
            const visibility = traces.map(t => t.name.toLowerCase().includes(search) ? true : 'legendonly');
            Plotly.restyle(plotElement, {{ visible: visibility }});
        }});

        document.getElementById('showAll').addEventListener('click', () => {{
            Plotly.restyle(plotElement, {{ visible: true }});
        }});

        document.getElementById('hideAll').addEventListener('click', () => {{
            Plotly.restyle(plotElement, {{ visible: 'legendonly' }});
        }});

        const mediaPlayer = document.getElementById('mediaPlayer');
        plotElement.on('plotly_click', function(data){{
            if(data.points.length > 0) {{
                mediaPlayer.currentTime = data.points[0].xaxis.d2l(data.points[0].x);
                mediaPlayer.play();
            }}
        }});

        {hover_tooltip_js}
    </script>
    </body>
    </html>
    """
    with open(html_dashboard_path, 'w', encoding='utf-8') as f: f.write(html_content)
    print(f'Saved interactive dashboard to: \033[1;34m{html_dashboard_path}\033[1;0m')

    stage('summary')
    print("📊  Generating AI-friendly event summary files…")
    
    # `framewise_output` is already in memory from the aggregate stage (RAM buffer on a fresh
    # run, HDF5 on a skipped one); only the timestamps still need reading.
    with h5py.File(h5_path, 'r') as hf:
        timestamps = hf['timestamps'][:]
        
    # Event detection heuristics — see THRESHOLDS at the top of this file.
    onset_threshold = THRESHOLDS['csv_onset']
    offset_threshold = THRESHOLDS['csv_offset']
    min_event_duration_seconds = THRESHOLDS['csv_min_seconds']
    top_n_per_class = THRESHOLDS['csv_top_per_class']

    # Inclusion gate: an absolute floor, not the top-10-by-peak list.
    # Ranking and inclusion are separate decisions. Gating on top_labels silently deleted
    # every event outside the eventogram's top rows (e.g. the Explosion in this clip).
    event_floor = THRESHOLDS['csv_include_peak']
    peak_by_label = framewise_output.max(axis=0)
    event_labels = [labels[i] for i in np.where(peak_by_label >= event_floor)[0]]

    events_by_class = {label: [] for label in event_labels}
    frame_dt = 1.0 / effective_ui_fps if effective_ui_fps else 0.0

    def emit_block(label, label_idx, start_idx, end_idx):
        """Record one contiguous block. end_idx is exclusive."""
        start_time = timestamps[start_idx]
        end_time = timestamps[end_idx] if end_idx < len(timestamps) else timestamps[-1] + frame_dt
        duration_secs = end_time - start_time
        if duration_secs < min_event_duration_seconds:
            return
        block_probs = framewise_output[start_idx:end_idx, label_idx]
        events_by_class[label].append({
            'sound_class': label,
            'start_time_seconds': round(float(start_time), 3),
            'end_time_seconds': round(float(end_time), 3),
            'duration_seconds': round(float(duration_secs), 3),
            'peak_probability': float(np.max(block_probs)),
            'average_probability': float(np.mean(block_probs)),
        })

    # Identify continuous blocks of sound for every class above the floor.
    # A block still open at the last frame must be closed too: it used to be
    # silently dropped, so an event running to end-of-file never reached the CSV.
    for label_idx, label in enumerate(labels):
        if label not in events_by_class: continue

        in_event, event_start_idx = False, 0
        for idx, prob in enumerate(framewise_output[:, label_idx]):
            if not in_event and prob > onset_threshold:
                in_event, event_start_idx = True, idx
            elif in_event and prob < offset_threshold:
                in_event = False
                emit_block(label, label_idx, event_start_idx, idx)
        if in_event:
            emit_block(label, label_idx, event_start_idx, len(framewise_output))

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

    stage('delta')
    # This format is optimized for AI attention mechanisms: it uses Run-Length Encoding (RLE)
    # and probability deltas to focus on "attacks" and "decays".
    print("📊  Generating detailed AI-friendly event delta JSON (RLE optimized)…")
    ai_threshold = THRESHOLDS['delta_include_prob']
    events_derivative = collections.defaultdict(list)
    
    # We already loaded framewise_output in the summary stage

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
                        "start": round(float(timestamps[frame_index]), 3),
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
                    "end_time": round(float(timestamps[frame_index]), 3),
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
                "end_time": round(float(timestamps[-1]), 3),
                "peak_prob": current_event["peak"],
                "delta_trace": zip_trace(deltas)
            })

    with open(json_ai_path, 'w') as f:
        json.dump({k: v for k, v in events_derivative.items() if v}, f, indent=2)
    print(f'Saved AI-friendly delta JSON to: \033[1;34m{json_ai_path}\033[1;0m')

    stage('render')
    if args.dynamic_eventogram or args.static_eventogram:
        print(f"🎞  Initializing video rendering pipeline ({video_fps} FPS)…")
        
        # Rendering Strategy: Dynamic (Scrolling) vs Static (Marker only)
        if args.dynamic_eventogram:
            output_video_path = os.path.join(output_dir, f"{base_name}{tag_suffix}_eventogram_dynamic.mp4")
            window_frames = int(args.window_duration * effective_ui_fps)
            half_window = window_frames // 2 

            # PRECOMPUTE: Map every data point to its local acoustic window (once per run)
            print(f"📊  Precomputing {frames_num} windows (Adaptive={args.use_adaptive_window})…")
            precomputed_data = []
            for i in range(frames_num):
                start_f, end_f = max(0, i - half_window), min(frames_num, i + half_window)
                
                # Adaptive logic: Try to center the window on acoustic "peaks" rather than strict time
                if args.use_adaptive_window:
                    kl_threshold = 0.5
                    lookahead_f = int(adaptive_lookahead * effective_ui_fps)
                    for offset in range(half_window, half_window + lookahead_f, int(max(1, effective_ui_fps))):
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
                t_start, t_end, t_curr = s_f / effective_ui_fps, e_f / effective_ui_fps, i / effective_ui_fps
                
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
            i = min(int(t * effective_ui_fps), frames_num - 1)
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
        total_output_frames = int(duration * video_fps)
        
        # FFmpeg setup: Pipe 0 is raw video frames; Input 1 is the original audio
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{v_w}x{v_h}",
            "-pix_fmt", "rgb24", "-r", str(video_fps),
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
                t = frame_idx / video_fps
                frame = make_frame(t)
                
                # Surgical Crop: Ensure buffer matches the 'even' dimensions promised to FFmpeg
                if frame.shape[0] > v_h or frame.shape[1] > v_w:
                    frame = frame[:v_h, :v_w, :]
                
                ffmpeg_proc.stdin.write(frame.astype(np.uint8).tobytes())
                
                if frame_idx % (video_fps * 10) == 0:
                    percent = (frame_idx / total_output_frames) * 100
                    print(f"   Progress: {percent:.1f}% ({t:.0f}s / {duration:.0f}s)", end="\r")
            
            print(f"   Progress: 100.0% ({duration:.0f}s / {duration:.0f}s)")
        finally:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()

        print(f"✅ Saved eventogram video to: \033[1;34m{output_video_path}\033[1;0m")
        if args.dynamic_eventogram: plt.close(fig_fr)

        # --- FFmpeg Overlay (Video-on-Video) ---
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
        else:
            print("🎧 Source is audio-only, skipping video overlay.")

    # Intermediates are never artifacts: drop the temporary CFR video, CBR MP3 and
    # recovery file now. cleanup_scratch() is also registered at exit, so a crash
    # mid-run cannot leave a multi-GB intermediate behind. Nothing is cached
    # between runs.
    cleanup_scratch()

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
    parser.add_argument('--ui_fps', type=int, default=5,
                        help='Data frames per second (Hz) for the single master grid used by the '
                             'HDF5, the dashboard, the PNG and the CSV/JSON. '
                             'Note: Values > 5 are DECEPTIVE/GIGO as the model (Cnn14_DecisionLevelMax) '
                             'only "thinks" at ~3.125 Hz native resolution due to internal pooling layers; '
                             'high values only inflate the artifacts with redundant data.')
    parser.add_argument('--video_fps', type=int, default=30,
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
 
    print(f"Eventogrammer, version 6.14.0") 
    print(f"Adaptation of: https://github.com/qiuqiangkong/audioset_tagging_cnn")
    print(f"Recent Material Changes:")
    print(f"* One master grid (--ui_fps) feeds the HDF5, PNG, dashboard, CSV and JSON; the")
    print(f"  former --log_fps duplicate stream is gone. --log_fps/--no-shapash were removed.")
    print(f"* Significance thresholds live in one THRESHOLDS dict, injected into the dashboard")
    print(f"  so the HTML, CSV and JSON gates cannot drift apart.")
    print(f"* The sanitized CBR MP3 is written to the system temp dir and deleted after the")
    print(f"  run; a rerun that skips inference and video skips the transcode entirely.")
    print(f"* A sound event still open at the last frame is now closed and written to the CSV")
    print(f"  (it used to be dropped silently).")
    print(f"* Dashboard hover is a compact top-N tooltip, strongest first; the native unified")
    print(f"  legend was clipped on long vocabularies and showed only the least probable tail.")
    print(f"* Pipeline stages are named (see STAGES) and logged as [n/N] banners.")
    print()

    print(f"Note on the models:")   
    print(f"* Cnn14_DecisionLevelMax (Sound Event Detection): Uses Decision-level pooling to maintain")
    print(f"  time resolution. Essential for generating Eventograms and high-res HDF5 logs.")
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
    print(f"time bash -c 'for file in *; do bash audio_me.sh \"$file\" --dynamic_eventogram; done'")
    print()

    print("Tips & Environment Hacks:")
    print("* For 'undefined symbol' (e.g. torch_from_blob), 'Invalid argument' crashes, or SIGABRT:")
    print("  Root Cause: ABI Collision. You likely have a mix of 'apt' (Debian) and 'pip' (Official) Torch packages.")
    print("  The C++ backend becomes a 'hollow shell'—importing the Python part works, but touching the C++ core kills the process.")
    print("  Fix: Uninstall ALL apt versions and 'pip install -U' the whole stack together to ensure binary alignment:")
    print("  Run: pip install -U torch torchaudio torchcodec --extra-index-url https://download.pytorch.org/whl/cpu")
    print("* If Torchcodec errors (e.g., 'libnvrtc.so.13 not found'):")
    print("  Root Cause: CUDA Ghosting. Standard wheels link to NVIDIA libraries.")
    print("  Fix: 'pip uninstall torchcodec' . Use the --extra-index-url above to force CPU-only binaries: 'pip install --upgrade torchcodec --extra-index-url https://download.pytorch.org/whl/cpu'. Or downgrade to match versions, use: 'pip index versions torchcodec' to match. ")
    print("* If you see 'NotImplementedError: sys.platform = android' after an update:")
    print(f"  Edit: /data/data/com.termux/files/usr/lib/python{py_ver}/site-packages/torchaudio/_internally_replaced_utils.py")
    print("  Change 'if sys.platform == \"linux\":' to 'if sys.platform == \"android\":'")
    print("* If some coverage numba error: do 'apt remove python3-coverage'. Be careful with the below python modules if they have parallel apt based install versions, use one or the other then: 'python -m pip install torch torchaudio torchcodec --upgrade --extra-index-url https://download.pytorch.org/whl/cpu ' : do check these against their apt versions. For anything torch or torchcodec related: sudo apt install libpython3.* to avoid .so file linkage problems")
    
    print("* If: 'LibsndfileError: File contains data in an unimplemented format', then 'git clone https://github.com/libsndfile/libsndfile.git' and install it in e.g. Termux. Or run in Proot.")  

    print()

    print(f"Dependency Versions:")
    print(f"Torch: {torch.__version__ if torch else 'Not Available'}")
    print(f"Torchaudio: {torchaudio.__version__ if torchaudio else 'Not Available'}")
    # Often errors, unless  sudo apt install libpython3.* to avoid .so file linkage problems:
    print(f"Torchcodec: {getattr(torchcodec, '__version__', 'Unknown') if torchcodec else 'Error checking version, so do read the tips above!'}")
 
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

