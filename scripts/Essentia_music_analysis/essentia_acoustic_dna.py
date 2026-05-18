#!/usr/bin/env python3
"""
Essentia Acoustic DNA (Neural Inference) - Termux/Debian Hybrid Edition
Demonstrates the unique value of essentia-tensorflow with a robust fallback
to standard TensorFlow for environments like Debian-under-Termux (PRoot).
"""
import os
import sys
import json
import argparse
import numpy as np
import subprocess
import tempfile
import pandas as pd

# 1. Import Check & Dynamic Fallback Setup
try:
    import essentia.standard as es
    HAS_ESSENTIA = True
except ImportError:
    print("\n[!] Error: Essentia module not found.")
    print("Please install essentia or essentia-tensorflow.\n")
    sys.exit(1)

# Check if we have the "All-in-one" build
HAS_ESSENTIA_TF = hasattr(es, 'TensorflowPredictEffnetDiscogs')

# 2. Native TensorFlow Fallback Class
class NativeTFPredictor:
    """
    A 'Universal Bridge' that runs Essentia .pb models using the standard 
    tensorflow library. This allows the script to work in environments 
    where Essentia wasn't compiled with TF support (like Termux/PRoot).
    """
    def __init__(self, graph_file, input_node, output_node):
        try:
            import tensorflow as tf
            self.tf = tf
        except ImportError:
            print("\n[!] Error: Standard 'tensorflow' package not found.")
            print("Try: pip install tensorflow\n")
            sys.exit(1)

        print(f"        [Bridge] Loading {os.path.basename(graph_file)} via native TF...")
        self.input_node = input_node
        self.output_node = output_node
        self.sess = self._load_graph(graph_file)

    def _load_graph(self, frozen_graph_filename):
        with self.tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = self.tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        
        with self.tf.compat.v1.Graph().as_default() as graph:
            self.tf.import_graph_def(graph_def, name="")
        
        # Optimization: CPU-only and limit threads for stability in Proot
        config = self.tf.compat.v1.ConfigProto(
            intra_op_parallelism_threads=2,
            inter_op_parallelism_threads=2,
            device_count={'GPU': 0}
        )
        return self.tf.compat.v1.Session(graph=graph, config=config)

    def __call__(self, input_data):
        input_tensor = self.sess.graph.get_tensor_by_name(f"{self.input_node}:0")
        output_tensor = self.sess.graph.get_tensor_by_name(f"{self.output_node}:0")
        return self.sess.run(output_tensor, feed_dict={input_tensor: input_data})

# 3. Model Pre-processing Fallback (The C++ Emulation)
def get_embeddings_native(audio, model_path):
    """
    Emulates es.TensorflowPredictEffnetDiscogs using standard Essentia
    algorithms and native TensorFlow.
    """
    # Discogs-EffNet parameters
    patch_size = 12480 # ~0.78s at 16kHz
    
    # Pre-processing Chain (Mel Spectrogram)
    # Note: For simplicity in this bridge, we attempt to pass raw patches 
    # if the model has the mel-stage built-in. 
    # Discogs-EffNet .pb files from MTG often expect 12480 raw samples.
    
    # Slice audio into patches
    n_patches = len(audio) // patch_size
    if n_patches == 0:
        return np.zeros((1, 1280))
        
    patches = audio[:n_patches * patch_size].reshape(n_patches, patch_size)
    
    # Run Native TF
    predictor = NativeTFPredictor(model_path, 'model/Placeholder', 'PartitionedCall:1')
    return predictor(patches)


def get_sanitized_audio(input_file, target_sr=16000):
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

    print(f"\n--- Essentia Acoustic DNA (Termux/PC Hybrid) ---")
    print(f"Mode: {'Standard (Built-in TF)' if HAS_ESSENTIA_TF else 'Bridge (Native TF Fallback)'}")
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
            return
        models[k] = full_path

    # 3. Process Audio
    temp_wav = get_sanitized_audio(audio_path, target_sr=16000)
    if not temp_wav: return

    try:
        print("\n[Step 1] Loading audio at 16kHz...")
        audio = es.MonoLoader(filename=temp_wav, sampleRate=16000)()
        
        # [Step 2] Compute Embeddings
        print("[Step 2] Extracting Neural Embeddings (Discogs-EffNet)...")
        if HAS_ESSENTIA_TF:
            embedding_model = es.TensorflowPredictEffnetDiscogs(
                graphFilename=models["embedding"],
                output='PartitionedCall:1'
            )
            embeddings = embedding_model(audio)
        else:
            embeddings = get_embeddings_native(audio, models["embedding"])
        
        print(f"        Generated {embeddings.shape[0]} feature patches.")

        # [Steps 3-5] Classification
        print("[Step 3] Predicting Genres, Mood, and Vocals...")
        
        if HAS_ESSENTIA_TF:
            # High-speed built-in path
            genre_model = es.TensorflowPredict2D(graphFilename=models["genre"], input='serving_default_model_Placeholder', output='PartitionedCall')
            mood_model = es.TensorflowPredict2D(graphFilename=models["mood"], input='model/Placeholder', output='model/Softmax')
            voice_model = es.TensorflowPredict2D(graphFilename=models["voice"], input='model/Placeholder', output='model/Softmax')
            
            genre_predictions = genre_model(embeddings)
            mood_predictions = mood_model(embeddings)
            voice_predictions = voice_model(embeddings)
        else:
            # Native TF Bridge path
            genre_model = NativeTFPredictor(models["genre"], 'serving_default_model_Placeholder', 'PartitionedCall')
            mood_model = NativeTFPredictor(models["mood"], 'model/Placeholder', 'model/Softmax')
            voice_model = NativeTFPredictor(models["voice"], 'model/Placeholder', 'model/Softmax')
            
            genre_predictions = genre_model(embeddings)
            mood_predictions = mood_model(embeddings)
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
                "voice": dict(top_voice),
                "inference_engine": "essentia-tf" if HAS_ESSENTIA_TF else "native-tf-bridge"
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
