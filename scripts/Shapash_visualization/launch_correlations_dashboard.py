import pandas as pd
import numpy as np
import h5py
from sklearn.ensemble import RandomForestRegressor
from shapash import SmartExplainer
import os
import argparse

def main():
    # 1. Setup Command Line Arguments
    parser = argparse.ArgumentParser(description="Terminal-only Shapash Acoustic Explainer")
    parser.add_argument("h5_path", help="Path to the full_event_log.h5 file")
    parser.add_argument("--target", help="Specific sound class to explain (default: the top sound detected)", default=None)
    parser.add_argument("--estimators", type=int, help="Number of trees in Random Forest", default=20)
    parser.add_argument("--samples", type=int, help="Number of background samples to include", default=200)
    
    args = parser.parse_args()

    # 2. Load the data
    if not os.path.exists(args.h5_path):
        print(f"Error: File not found: {args.h5_path}")
        return

    print("loading " + args.h5_path)
    try:
        with h5py.File(args.h5_path, 'r') as hf:
            data = hf['framewise_output'][:]
            labels = [l.decode('utf-8') for l in hf['labels'][:]]
            timestamps = hf['timestamps'][:]
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=labels)
            df['time'] = timestamps
    except Exception as e:
        print(f"Error reading HDF5: {e}")
        return

    # 3. Target Selection
    # Drop non-acoustic tracking columns often found in PANNs logs
    drop_cols = ["time", "_index_", "_predict_", "_target_", "_error_"]
    sounds_only = df.drop(columns=[c for c in drop_cols if c in df.columns])

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

    print(f"Test: explaining target class: {target} (avg prob: {round(avg_p, 4)})")

    # 4. Prepare Features (In-Memory Only)
    X = sounds_only.drop(columns=[target])
    y = sounds_only[target]

    # Filter features: only keep those with mean probability > 0.001 to keep training fast
    active_mask = X.mean() > 0.001
    X = X.loc[:, active_mask]
    print(f"analyzing {len(X.columns)} active features for {target}")

    # 5. Train the "Explainer" Model
    print(f"training model with {args.estimators} estimators...")
    model = RandomForestRegressor(n_estimators=args.estimators, random_state=42)
    model.fit(X, y)

    # 6. Strategic Sampling (Local Context)
    peaks_indices = df.sort_values(by=target, ascending=False).head(20).index.tolist()
    bg_indices = X.sample(n=min(args.samples, len(X)), random_state=42).index.tolist()
    sample_indices = sorted(list(set(peaks_indices + bg_indices)))

    X_sample = X.loc[sample_indices]

    print(f"compiling shapash in-memory on {len(X_sample)} samples...")
    xpl = SmartExplainer(model=model)
    xpl.compile(x=X_sample)

    # 7. Compute Global Importance
    print("computing acoustic correlations...")
    xpl.compute_features_import()
    
    # 8. Display Summary
    print("\n--- TOP ACOUSTIC WEIGHTS (PREDICTORS FOR " + str(target).upper() + ") ---")
    if hasattr(xpl, 'features_imp') and isinstance(xpl.features_imp, pd.Series):
        print(xpl.features_imp.sort_values(ascending=False).head(15))
    elif hasattr(xpl, 'features_imp'):
        print(xpl.features_imp)
    else:
        print("Summary unavailable.")
    print("--------------------------------------------------\n")
    print("Analysis complete. Terminal output is the authoritative summary.")

if __name__ == "__main__":
    main()
