#!/usr/bin/env python3
#import essentia.tensorflow as es
import essentia.standard as es

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless
import matplotlib.pyplot as plt
import subprocess
import tempfile
import pandas as pd

def get_sanitized_audio(input_file):
    """
    Uses FFmpeg to convert any media (including .webm) to a high-quality PCM WAV
    that Essentia's loaders can handle reliably.
    """
    temp_wav = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + ".wav")
    print(f"Sanitizing audio codec for Essentia (Transcoding to WAV)...")
    try:
        # Resample to 44100Hz, mono, for high-res forensics
        subprocess.run([
            'ffmpeg', '-loglevel', 'error', '-i', input_file, 
            '-ar', '44100', '-ac', '1', '-y', temp_wav
        ], check=True)
        return temp_wav
    except Exception as e:
        print(f"Sanitization failed: {e}")
        return None

def analyze_music_full(audio_path):
    # 0. Path Management (PANNs style)
    audio_path = os.path.abspath(audio_path)
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        return

    input_dir = os.path.dirname(audio_path)
    base_name = os.path.basename(audio_path)
    output_dir = os.path.join(input_dir, f"{base_name}_essentia_forensics")
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Essentia Forensic Analysis & Visualization ---")
    print(f"Input:  {audio_path}")
    print(f"Output: {output_dir}")

    # 1. Sanitization Gate
    temp_wav = get_sanitized_audio(audio_path)
    if not temp_wav: return

    try:
        # 2. Load Audio
        print("\nLoading audio signal...")
        loader = es.MonoLoader(filename=temp_wav, sampleRate=44100)
        audio = loader()
        duration = len(audio) / 44100.0

        # 3. Extract Melody (Melodia)
        print("Extracting Melody (PredominantPitchMelodia)...")
        pitch_extractor = es.PredominantPitchMelodia(frameSize=2048, hopSize=128)
        pitch_values, pitch_confidence = pitch_extractor(audio)
        pitch_times = np.linspace(0.0, duration, len(pitch_values))
        
        # Save Melody Artifact
        melody_df = pd.DataFrame({
            'time': pitch_times,
            'pitch_hz': pitch_values,
            'confidence': pitch_confidence
        })
        melody_csv = os.path.join(output_dir, "artifact_melody.csv")
        melody_df.to_csv(melody_csv, index=False)

        # 4. Extract High-level Stats and Frames (for HPCP and Chords)
        print("Extracting Tonal/Spectral Frames (MusicExtractor)...")
        extractor = es.MusicExtractor(lowlevelHopSize=1024, tonalHopSize=1024)
        pool_stats, pool_frames = extractor(temp_wav)

        # 5. Chord Detection
        print("Detecting Chords...")
        hpcp = pool_frames['tonal.hpcp']
        chords_extractor = es.ChordsDetection()
        chords, strength = chords_extractor(hpcp)
        chord_times = np.linspace(0.0, duration, len(chords))
        
        # Save Chords Artifact
        chords_df = pd.DataFrame({
            'time': chord_times,
            'chord': chords,
            'strength': strength
        })
        chords_csv = os.path.join(output_dir, "artifact_chords.csv")
        chords_df.to_csv(chords_csv, index=False)
        
        # Save HPCP (Chromagram) Artifact
        if hpcp.shape[1] == 12:
            hpcp_cols = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        else:
            hpcp_cols = [f'bin_{i}' for i in range(hpcp.shape[1])]
            
        hpcp_df = pd.DataFrame(hpcp, columns=hpcp_cols)
        hpcp_df.insert(0, 'time', np.linspace(0.0, duration, len(hpcp)))
        hpcp_csv = os.path.join(output_dir, "artifact_hpcp.csv")
        hpcp_df.to_csv(hpcp_csv, index=False)

        # 6. Loudness Envelope
        print("Calculating Loudness Envelope...")
        envelope = es.Envelope()(audio)
        env_times = np.linspace(0.0, duration, len(envelope))
        
        # Save Loudness Artifact (Downsampled for AI efficiency)
        ds_factor = 100
        loudness_df = pd.DataFrame({
            'time': env_times[::ds_factor],
            'amplitude': envelope[::ds_factor]
        })
        loudness_csv = os.path.join(output_dir, "artifact_loudness.csv")
        loudness_df.to_csv(loudness_csv, index=False)

        # --- VISUALIZATION ---
        print("\nGenerating Forensic Graphs...")
        fig = plt.figure(figsize=(15, 12))
        
        ax1 = fig.add_subplot(4, 1, 1)
        ax1.plot(pitch_times, pitch_values, color='tab:blue', linewidth=0.5)
        ax1.set_ylabel('Pitch (Hz)'); ax1.set_title(f'Melody Detection - {base_name}'); ax1.set_ylim(0, 1000); ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(4, 1, 2)
        ax2.imshow(hpcp.T, aspect='auto', origin='lower', cmap='magma', interpolation='none')
        ax2.set_ylabel('Semitone'); ax2.set_title('HPCP (Chromagram)')
        ax2.set_yticks(np.arange(12)); ax2.set_yticklabels(['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'])

        ax3 = fig.add_subplot(4, 1, 3)
        ax3.plot(chord_times, strength, color='tab:green')
        ax3.set_ylabel('Strength'); ax3.set_title('Chord Detection Strength')
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(4, 1, 4)
        ax4.plot(env_times[::ds_factor], envelope[::ds_factor], color='tab:red', alpha=0.7)
        ax4.set_ylabel('Amplitude'); ax4.set_xlabel('Time (seconds)'); ax4.set_title('Loudness Envelope'); ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "forensic_plot.png")
        plt.savefig(plot_path, dpi=150); plt.close(fig)

        # --- SUMMARY ---
        print(f"\n[ FORENSIC SUMMARY ]")
        print(f"  - Key/Scale:      {pool_stats['tonal.key_krumhansl.key']} {pool_stats['tonal.key_krumhansl.scale']}")
        print(f"  - BPM:            {pool_stats['rhythm.bpm']:.2f}")
        
        unique_chords = []
        for c in chords:
            if not unique_chords or c != unique_chords[-1]: unique_chords.append(c)
        print(f"  - Chord Flow:     {' -> '.join(unique_chords[:12])}...")

        print(f"\n✅ Analysis complete.")
        print(f"📊 Graphs:    {plot_path}")
        print(f"📄 Artifacts: {output_dir}/artifact_*.csv (AI-Readable)")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if os.path.exists(temp_wav): os.remove(temp_wav)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional Essentia Analysis with AI-readable artifacts.")
    parser.add_argument("audio_path", type=str, help="Path to the media file.")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    analyze_music_full(args.audio_path)
