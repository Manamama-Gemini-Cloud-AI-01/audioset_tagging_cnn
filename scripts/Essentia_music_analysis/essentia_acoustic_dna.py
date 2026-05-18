#!/usr/bin/env python3
"""
Essentia Acoustic DNA (Neural Inference)
Demonstrates the unique value of essentia-tensorflow by using deep learning
models to identify high-level semantic features (Genre and Mood).
"""
import os
import sys
import json
import argparse
import numpy as np
import subprocess
import tempfile
import pandas as pd

# 1. Import Check
try:
    import essentia.standard as es
except ImportError:
    print("\n[!] Error: Essentia module not found.")
    print("Please run: pip install --force-reinstall essentia-tensorflow\n")
    sys.exit(1)

def get_sanitized_audio(input_file, target_sr=16000):
    """
    Discogs-EffNet models require 16000Hz mono.
    """
    temp_wav = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + f"_{target_sr}.wav")
    print(f"Sanitizing audio for Neural Inference (Resampling to {target_sr}Hz mono)...")
    try:
        subprocess.run([
            'ffmpeg', '-loglevel', 'error', '-i', input_file, 
            '-ar', str(target_sr), '-ac', '1', '-y', temp_wav
        ], check=True)
        return temp_wav
    except Exception as e:
        print(f"Sanitization failed: {e}")
        return None

def analyze_acoustic_dna(audio_path):
    audio_path = os.path.abspath(audio_path)
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        return

    # Path setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    model_dir = os.path.join(project_root, "models")
    
    input_dir = os.path.dirname(audio_path)
    base_name = os.path.basename(audio_path)
    output_dir = os.path.join(input_dir, f"{base_name}_acoustic_dna")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Essentia Acoustic DNA (TensorFlow Powered) ---")
    print(f"Input:  {audio_path}")
    print(f"Output: {output_dir}")

    # 2. Model Path Check
    models = {
        "embedding": "discogs-effnet-bs64-1.pb",
        "genre": "genre_discogs400-discogs-effnet-1.pb",
        "genre_labels": "genre_discogs400-discogs-effnet-1.json",
        "mood": "mood_acoustic-discogs-effnet-1.pb",
        "mood_labels": "mood_acoustic-discogs-effnet-1.json",
        "voice": "voice_instrumental-discogs-effnet-1.pb",
        "voice_labels": "voice_instrumental-discogs-effnet-1.json"
    }
    
    for k, v in models.items():
        full_path = os.path.join(model_dir, v)
        if not os.path.exists(full_path):
            print(f"Critical Error: Model file missing at {full_path}")
            print("Please ensure the models folder is populated.")
            return
        models[k] = full_path

    # 3. Process Audio
    temp_wav = get_sanitized_audio(audio_path, target_sr=16000)
    if not temp_wav: return

    try:
        print("\n[Step 1] Loading audio at 16kHz...")
        audio = es.MonoLoader(filename=temp_wav, sampleRate=16000)()
        
        # [Step 2] Compute Embeddings (The Neural Feature Set)
        # These models break the audio into patches automatically.
        print("[Step 2] Extracting Neural Embeddings (Discogs-EffNet)...")
        embedding_model = es.TensorflowPredictEffnetDiscogs(
            graphFilename=models["embedding"],
            output='PartitionedCall:1'
        )
        embeddings = embedding_model(audio)
        print(f"        Generated {embeddings.shape[0]} feature patches.")

        # [Step 3] Genre Classification (400 Classes)
        print("[Step 3] Predicting Genres (Discogs400 Taxonomy)...")
        genre_model = es.TensorflowPredict2D(
            graphFilename=models["genre"],
            input='serving_default_model_Placeholder',
            output='PartitionedCall'
        )
        genre_predictions = genre_model(embeddings)
        
        # [Step 4] Mood Classification (Acousticness/Electronic)
        print("[Step 4] Predicting Mood/Atmosphere (Acoustic vs Electronic)...")
        mood_model = es.TensorflowPredict2D(
            graphFilename=models["mood"],
            input='model/Placeholder',
            output='model/Softmax'
        )
        mood_predictions = mood_model(embeddings)

        # [Step 5] Voice/Instrumental Classification
        print("[Step 5] Predicting Voice vs Instrumental...")
        voice_model = es.TensorflowPredict2D(
            graphFilename=models["voice"],
            input='model/Placeholder',
            output='model/Softmax'
        )
        voice_predictions = voice_model(embeddings)

        # 4. Result Processing
        def get_top_results(preds, labels_json, top_n=5):
            with open(labels_json, 'r') as f:
                labels = json.load(f)['classes']
            
            mean_preds = np.mean(preds, axis=0)
            top_indices = np.argsort(mean_preds)[::-1][:top_n]
            return [(labels[i], float(mean_preds[i])) for i in top_indices]

        top_genres = get_top_results(genre_predictions, models["genre_labels"], 10)
        top_moods = get_top_results(mood_predictions, models["mood_labels"], 2)
        top_voice = get_top_results(voice_predictions, models["voice_labels"], 2)

        # 5. Output Summary
        print(f"\n[ NEURAL ANALYSIS SUMMARY ]")
        
        print(f"\n  Top Detected Genres (Discogs Taxonomy):")
        for label, score in top_genres:
            print(f"    - {label:<25} ({score*100:5.1f}%)")

        print(f"\n  Acoustic Signature:")
        for label, score in top_moods:
            print(f"    - {label:<25} ({score*100:5.1f}%)")

        print(f"\n  Vocal Signature:")
        for label, score in top_voice:
            print(f"    - {label:<25} ({score*100:5.1f}%)")

        # 6. Save Artifact
        results_json = os.path.join(output_dir, "acoustic_dna.json")
        with open(results_json, 'w') as f:
            json.dump({
                "genres": dict(top_genres),
                "moods": dict(top_moods),
                "voice": dict(top_voice)
            }, f, indent=4)
        
        print(f"\n✅ Analysis complete.")
        print(f"📄 Neural Profile: {results_json}")

    except Exception as e:
        print(f"An error occurred during neural inference: {e}")
    finally:
        if os.path.exists(temp_wav): os.remove(temp_wav)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Essentia Acoustic DNA (Neural Inference)")
    parser.add_argument("audio_path", type=str, help="Path to the media file.")
    args = parser.parse_args()
    analyze_acoustic_dna(args.audio_path)
