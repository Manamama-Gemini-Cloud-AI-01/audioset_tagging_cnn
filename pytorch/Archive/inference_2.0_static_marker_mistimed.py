import os
import sys
import numpy as np
import argparse
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import csv
import datetime
import subprocess
import platform
import soundfile as sf
import json
import moviepy
print(f"Using moviepy version: {moviepy.__version__}")
from moviepy import VideoFileClip, ImageSequenceClip, CompositeVideoClip, TextClip, concatenate_videoclips, ImageClip, VideoClip, AudioFileClip, ColorClip

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
        return any(stream['codec_type'] == 'video' for stream in streams)
    except subprocess.CalledProcessError:
        return False

def get_duration_and_fps(audio_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=avg_frame_rate,width,height', '-of', 'json', audio_path],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        duration_str = str(datetime.timedelta(seconds=int(duration)))
        print(f"\033[1;34m🗃️  Input file duration: {duration_str}\033[0m")
        
        fps = None
        width = None
        height = None
        if 'streams' in data:
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    if 'avg_frame_rate' in stream:
                        num, denom = map(int, stream['avg_frame_rate'].split('/'))
                        if denom != 0:
                            fps = num / denom
                            print(f"\033[1;34m🗃️  Input video frame rate: {fps:.2f} fps\033[0m")
                    if 'width' in stream and 'height' in stream:
                        width = stream['width']
                        height = stream['height']
                        print(f"\033[1;34m🗃️  Input video resolution: {width}x{height}\033[0m")
                    break
        return duration, fps, width, height
    except (subprocess.CalledProcessError, KeyError, ValueError) as e:
        print(f"\033[1;31mFailed to get duration, fps, width, or height with ffprobe: {e}\033[0m")
        return None, None, None, None

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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    if device.type == 'cuda':
        model.to(device)
        print(f'GPU number: {torch.cuda.device_count()}')
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')

    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    waveform = waveform[None, :]
    waveform = move_data_to_device(waveform, device)

    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

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

    classes_num = config.classes_num
    labels = config.labels
    
    # Paths
    audio_dir = os.path.dirname(audio_path)
    create_folder(audio_dir)
    base_filename = get_filename(audio_path) + '_audioset_tagging_cnn'
    fig_path = os.path.join(audio_dir, f'{base_filename}.png')
    csv_path = os.path.join(audio_dir, f'{base_filename}.csv')
    srt_path = os.path.join(audio_dir, f'{base_filename}.srt')
    video_path = os.path.join(audio_dir, f'{base_filename}_eventogram.mp4')
    
    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                  classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    if device.type == 'cuda':
        model.to(device)
        print(f'GPU number: {torch.cuda.device_count()}')
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')

    # Audio loading
    duration, video_fps, video_width, video_height = get_duration_and_fps(audio_path)
    if duration is None:
        print("\033[1;31mError: Could not determine audio duration. Exiting.\033[0m")
        return
    is_video = is_video_file(audio_path)
    
    # Load audio directly from original path, assuming librosa can handle it
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
    
    waveform = waveform[None, :]
    waveform = move_data_to_device(waveform, device)

    # Inference
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    print(f'Sound event detection result (time_steps x classes_num): {framewise_output.shape}')

    # --- Png visualization ---
    frames_per_second = sample_rate // hop_size
    stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=window_size, 
                             hop_length=hop_size, window='hann', center=True)
    frames_num = min(stft.shape[-1], framewise_output.shape[0])
    
    sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]
    top_k = 10
    top_result_mat = framewise_output[:, sorted_indexes[0:top_k]]
    
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    axs[0].matshow(np.log(np.abs(stft[:, 0:frames_num]) + 1e-10), origin='lower', aspect='auto', cmap='jet')
    axs[0].set_ylabel('Frequency bins')
    axs[0].set_title('Log spectrogram')
    axs[1].matshow(top_result_mat[0:frames_num].T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    
    tick_interval = 5
    x_ticks = np.arange(0, frames_num + 1, frames_per_second * tick_interval)
    x_labels = np.arange(0, duration + 1, tick_interval)
    axs[1].xaxis.set_ticks(x_ticks)
    axs[1].xaxis.set_ticklabels(x_labels[:len(x_ticks)])

    axs[1].yaxis.set_ticks(np.arange(0, top_k))
    axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0:top_k]])
    axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    axs[1].set_xlabel('Seconds')
    axs[1].xaxis.set_ticks_position('bottom')
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight', pad_inches=0)
    print(f'Saved sound event detection visualization to: {fig_path}')

    # --- CSV and SRT output ---
    threshold = 0.5
    all_events = []
    for j in range(framewise_output.shape[1]):
        is_active = False
        start_frame = 0
        for i in range(frames_num):
            if framewise_output[i, j] > threshold and not is_active:
                is_active = True
                start_frame = i
            elif framewise_output[i, j] <= threshold and is_active:
                is_active = False
                end_frame = i
                start_seconds = start_frame / frames_per_second
                end_seconds = end_frame / frames_per_second
                all_events.append({
                    'start_time': datetime.timedelta(seconds=start_seconds),
                    'end_time': datetime.timedelta(seconds=end_seconds),
                    'label': labels[j],
                    'probability': np.max(framewise_output[start_frame:end_frame, j])
                })
        if is_active:
            end_frame = frames_num
            start_seconds = start_frame / frames_per_second
            end_seconds = end_frame / frames_per_second
            all_events.append({
                'start_time': datetime.timedelta(seconds=start_seconds),
                'end_time': datetime.timedelta(seconds=end_seconds),
                'label': labels[j],
                'probability': np.max(framewise_output[start_frame:end_frame, j])
            })

    all_events.sort(key=lambda x: x['start_time'])

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['start_time', 'end_time', 'sound', 'probability'])
        for event in all_events:
            writer.writerow([event['start_time'].total_seconds(), event['end_time'].total_seconds(), event['label'], event['probability']])
    print(f'Saved CSV results to {csv_path}')

    with open(srt_path, 'w') as srtfile:
        for i, event in enumerate(all_events):
            start_time_str = str(event['start_time']).split('.')[0] + ',' + str(event['start_time'].microseconds // 1000).zfill(3)
            end_time_str = str(event['end_time']).split('.')[0] + ',' + str(event['end_time'].microseconds // 1000).zfill(3)
            srtfile.write(f"{i+1}\n{start_time_str} --> {end_time_str}\n{event['label']} ({event['probability']:.2f})\n\n")
    print(f'Saved SRT results to {srt_path}')

    # --- Video output ---
    # Create a single eventogram image for ffmpeg
    eventogram_fig_path = os.path.join(audio_dir, f'{base_filename}_eventogram_full.png')
    
    # Cap the width of the eventogram PNG to prevent performance issues
    max_png_width_inches = 1920 / 100  # Corresponds to 1920 pixels at 100 dpi
    fig_width_inches = min(frames_num / 100, max_png_width_inches)
    
    fig, ax = plt.subplots(figsize=(fig_width_inches, 4), dpi=100)
    ax.matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    ax.yaxis.set_ticks(np.arange(0, top_k))
    ax.yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0:top_k]])
    ax.yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    ax.xaxis.set_visible(False) # No time axis on the image itself
    plt.tight_layout(pad=0)
    plt.savefig(eventogram_fig_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # --- MoviePy Video output ---
    fps = video_fps if video_fps else 24

    # Create a static image clip of the eventogram
    static_eventogram_clip = ImageClip(eventogram_fig_path, duration=duration)

    # Create the moving time marker
    marker = ColorClip(size=(2, static_eventogram_clip.h), color=(255, 0, 0)).with_duration(duration)
    
    def marker_position(t):
        return ((static_eventogram_clip.w / duration) * t, 0)
        
    marker = marker.with_position(marker_position)

    # Create the eventogram with the moving marker
    eventogram_visual_clip = CompositeVideoClip([static_eventogram_clip, marker])

    # Add the original audio
    audio_clip = AudioFileClip(audio_path)
    eventogram_with_audio_clip = eventogram_visual_clip.with_audio(audio_clip)

    if is_video:
        # Overlay on the original video
        original_video_clip = VideoFileClip(audio_path)
        
        # Resize overlay
        overlay_height = int(original_video_clip.h * args.overlay_size)
        overlay_width = int(overlay_height * (static_eventogram_clip.w / static_eventogram_clip.h))
        
        eventogram_with_audio_clip = eventogram_with_audio_clip.resize((overlay_width, overlay_height))
        
        # Set opacity
        eventogram_with_audio_clip = eventogram_with_audio_clip.set_opacity(args.translucency)
        
        # Position in bottom right
        final_clip = CompositeVideoClip([
            original_video_clip,
            eventogram_with_audio_clip.set_position(("right", "bottom"))
        ])
        
        # Use the audio from the original video
        final_clip.audio = original_video_clip.audio
        
    else:
        # Full screen eventogram video
        final_clip = eventogram_with_audio_clip

    # Write the video file
    try:
        final_clip.write_videofile(
            video_path, 
            codec='libx264', 
            audio_codec='aac', 
            temp_audiofile='temp-audio.m4a', 
            remove_temp=True,
            fps=fps
        )
        print(f'Saved video to: {video_path}')
    except Exception as e:
        print(f"\033[1;31mError during moviepy video processing: {e}\033[0m")

    # Keep the eventogram_fig_path for inspection
    # os.remove(eventogram_fig_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio tagging and Sound event detection.')
    subparsers = parser.add_subparsers(dest='mode')

    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument('--audio_path', type=str, required=True, help='Audio file path')
    parser_at.add_argument('--sample_rate', type=int, default=32000)
    parser_at.add_argument('--window_size', type=int, default=1024)
    parser_at.add_argument('--hop_size', type=int, default=320)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=14000)
    parser_at.add_argument('--model_type', type=str, required=True)
    parser_at.add_argument('--checkpoint_path', type=str, required=True)
    parser_at.add_argument('--cuda', action='store_true', default=False)

    parser_sed = subparsers.add_parser('sound_event_detection')
    parser_sed.add_argument('--audio_path', type=str, required=True, help='Audio file path')
    parser_sed.add_argument('--sample_rate', type=int, default=32000)
    parser_sed.add_argument('--window_size', type=int, default=1024)
    parser_sed.add_argument('--hop_size', type=int, default=320)
    parser_sed.add_argument('--mel_bins', type=int, default=64)
    parser_sed.add_argument('--fmin', type=int, default=50)
    parser_sed.add_argument('--fmax', type=int, default=14000)
    parser_sed.add_argument('--model_type', type=str, required=True)
    parser_sed.add_argument('--checkpoint_path', type=str, required=True)
    parser_sed.add_argument('--cuda', action='store_true', default=False)
    parser_sed.add_argument('--translucency', type=float, default=0.7, help='Overlay translucency (0 to 1)')
    parser_sed.add_argument('--overlay_size', type=float, default=0.2, help='Overlay size as fraction of video height')
    parser_sed.add_argument('--eventogram_mode', action='store_true', default=False, help='Use eventogram instead of text-based video')

    args = parser.parse_args()

    if args.mode == 'audio_tagging':
        audio_tagging(args)
    elif args.mode == 'sound_event_detection':
        sound_event_detection(args)
    else:
        parser.print_help()