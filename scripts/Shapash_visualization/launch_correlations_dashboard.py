import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from shapash import SmartExplainer
import os
import argparse
import subprocess

def main():
    # 1. Setup Command Line Arguments
    parser = argparse.ArgumentParser(description="Universal Shapash Acoustic Explainer")
    parser.add_argument("csv_path", help="Path to the full_event_log.csv file")
    parser.add_argument("--target", help="Specific sound class to explain (default: automatic top sound)", default=None)
    parser.add_argument("--estimators", type=int, help="Number of trees in Random Forest", default=20)
    parser.add_argument("--samples", type=int, help="Number of background samples to include", default=200)
    
    args = parser.parse_args()

    # 2. Load the data
    if not os.path.exists(args.csv_path):
        print(f"Error: File not found: {args.csv_path}")
        return

    # Basic extension check to prevent binary file reading errors
    if not args.csv_path.lower().endswith('.csv'):
        print(f"Error: This script expects a CSV file (e.g., full_event_log.csv).")
        if args.csv_path.lower().endswith(('.mp3', '.wav', '.mp4', '.m4a', '.webm')):
            print(f"Hint: You provided a media file. You must run inference FIRST to generate the CSV.")
            print(f"Example: python3 pytorch/audioset_tagging_cnn_inference_6.py \"{args.csv_path}\" ...")
        return

    print("loading " + args.csv_path)
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 3. Target Selection
    sounds_only = df.drop(columns=["time"])
    
    if args.target:
        target = args.target
        if target not in sounds_only.columns:
            print(f"Error: Target '{target}' not found in CSV columns.")
            return
        avg_p = sounds_only[target].mean()
    else:
        # Automatic top sound selection
        top_sounds = sounds_only.mean().sort_values(ascending=False)
        target = top_sounds.index[0]
        avg_p = top_sounds.iloc[0]

    print("explaining target: " + str(target) + " (avg prob: " + str(round(avg_p, 4)) + ")")

    # 4. Prepare Features
    X = sounds_only.drop(columns=[target])
    y = df[target]

    # Filter features: only keep those with mean probability > 0.001 to keep it fast
    active_mask = X.mean() > 0.001
    X = X.loc[:, active_mask]
    print("analyzing " + str(len(X.columns)) + " features for " + str(target))

    # 5. Train the "Explainer" Model
    print(f"training model with {args.estimators} estimators...")
    model = RandomForestRegressor(n_estimators=args.estimators, random_state=42)
    model.fit(X, y)

    # 6. Strategic Sampling for the WebApp
    peaks_indices = df.sort_values(by=target, ascending=False).head(20).index.tolist()
    bg_indices = X.sample(n=min(args.samples, len(X)), random_state=42).index.tolist()
    sample_indices = sorted(list(set(peaks_indices + bg_indices)))

    X_sample = X.loc[sample_indices]

    print("compiling shapash on " + str(len(X_sample)) + " samples...")
    xpl = SmartExplainer(model=model)
    xpl.compile(x=X_sample)

    # 7. COMPUTE AND DISPLAY SUMMARY (The "Peek" Phase)
    print("computing global importance...")
    xpl.compute_features_import()
    
    print("\n--- TOP ACOUSTIC WEIGHTS (PREDICTORS FOR " + str(target).upper() + ") ---")
    if isinstance(xpl.features_imp, pd.Series):
        print(xpl.features_imp.sort_values(ascending=False).head(10))
    else:
        # In case of list/dict (multi-class logic)
        print(xpl.features_imp)
    print("--------------------------------------------------\n")

    # 8. SAVE THE BRAIN (The "Library" Phase)
    source_dir = os.path.dirname(os.path.abspath(args.csv_path))
    safe_target = str(target).replace(", ", "_").replace(" ", "_").lower()
    brain_filename = f"shapash_brain_{safe_target}.pkl"
    brain_path = os.path.join(source_dir, brain_filename)
    
    print("saving acoustic brain to: " + brain_path)
    xpl.save(brain_path)

    # 9. Launch the App (Dash 4.0 compatible)
    print("initializing webapp...")
    xpl.init_app()
    app = xpl.smartapp.app

    print("--------------------------------------------------")
    print("DASHBOARD STARTING (Explaining " + str(target) + ")...")
    print("Open your browser at: http://localhost:8050")
    print("--------------------------------------------------")

    # Auto-open the local server using the system 'open' command in background
    try:
        # We use a background shell command with sleep to give the server a moment to start
        subprocess.Popen('sleep 2 && open http://localhost:8050', shell=True)
    except Exception as e:
        print(f"Warning: Could not auto-open browser: {e}")

    app.run(debug=False, host="0.0.0.0", port=8050)

if __name__ == "__main__":
    main()
