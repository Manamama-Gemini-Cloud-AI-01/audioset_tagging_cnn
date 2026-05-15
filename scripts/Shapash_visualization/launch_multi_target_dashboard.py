import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from shapash import SmartExplainer
import shap
import os
import argparse
import joblib
from tqdm import tqdm
import warnings

# --- VERSIONING ---
VERSION = "1.0.7"
# ------------------

# Suppress warnings for cleaner CLI output
warnings.filterwarnings("ignore")

# Define a top-level function for monkeypatching to ensure picklability
class MultiOutputPredictProba:
    def __init__(self, model):
        self.model = model
    def __call__(self, x):
        return self.model.predict(x)

def main():
    print(f"--- Unified Acoustic Brain Dashboard v{VERSION} ---")
    
    parser = argparse.ArgumentParser(description="Unified Acoustic Brain Dashboard")
    parser.add_argument("csv_path", help="Path to the full_event_log.csv file")
    parser.add_argument("--top_n", type=int, help="Number of sound targets to explain (dropdown size)", default=50)
    parser.add_argument("--samples", type=int, help="Number of samples for SHAP calculation", default=200)
    parser.add_argument("--estimators", type=int, help="Number of trees in Random Forest", default=10)
    parser.add_argument("--force", action="store_true", help="Force retraining even if brain exists")
    
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: File not found: {args.csv_path}")
        return

    source_dir = os.path.dirname(os.path.abspath(args.csv_path))
    brain_path = os.path.join(source_dir, "shapash_unified_acoustic_brain.pkl")
    
    xpl = None
    if os.path.exists(brain_path) and not args.force:
        print(f"Found existing unified brain at {brain_path}. Loading...")
        try:
            xpl = joblib.load(brain_path)
            # Restore monkeypatching if it was lost during pickling
            if not hasattr(xpl.model, "predict_proba"):
                print("Restoring predict_proba monkeypatch...")
                xpl.model.predict_proba = MultiOutputPredictProba(xpl.model)
        except Exception as e:
            print(f"Warning: Could not load existing brain: {e}. Retraining...")

    if xpl is None:
        print(f"Loading {args.csv_path}...")
        df = pd.read_csv(args.csv_path)
        
        # 1. Prepare Data - "One CSV, One Model"
        sounds_df = df.drop(columns=["time"]) if "time" in df.columns else df
        
        # Filter out absolute silence (classes never detected)
        active_mask = (sounds_df != 0).any(axis=0)
        X = sounds_df.loc[:, active_mask]
        
        # Unified Ingestion: Use all sounds as features
        # Narrative Selection: Select Top N for the GUI dropdown
        all_active_names = X.columns.tolist()
        top_sounds = X.mean().sort_values(ascending=False).head(args.top_n)
        target_names = top_sounds.index.tolist()
        
        print(f"Unified Ingestion: {len(all_active_names)} active sound features.")
        print(f"Narrative Selection: Explaining the Top {len(target_names)} sounds in the GUI.")

        # Create mapping for numeric labels (Shapash requirement)
        name_to_id = {name: i for i, name in enumerate(target_names)}
        id_to_name = {i: name for i, name in enumerate(target_names)}

        # 2. Train ONE Multi-Output Model
        Y = X[target_names]
        print(f"Training Unified Acoustic Model (Forest depth=5, Estimators={args.estimators})...")
        model = RandomForestRegressor(n_estimators=args.estimators, max_depth=5, random_state=42, n_jobs=-1)
        model.fit(X, Y)
        
        # 3. SHAP Calculation (Strategic Sampling)
        print(f"Calculating SHAP contributions (sampling {args.samples} points)...")
        X_sample = X.sample(n=min(len(X), args.samples), random_state=42)
        explainer = shap.TreeExplainer(model)
        
        shap_results = explainer.shap_values(X_sample)
        
        # 4. Mask Self-Contribution & Format for Shapash
        print("Masking identity correlations...")
        list_contrib_df = []
        
        if isinstance(shap_results, list):
            list_contrib_raw = shap_results
        else:
            # 3D Array (samples, features, outputs)
            list_contrib_raw = [shap_results[:, :, i] for i in range(shap_results.shape[2])]

        for i, target in enumerate(target_names):
            contrib_arr = list_contrib_raw[i].copy() # Use copy to avoid modifying original SHAP values
            # If target is also a feature, mask it
            if target in X.columns:
                target_idx = X.columns.get_loc(target)
                contrib_arr[:, target_idx] = 0
            list_contrib_df.append(pd.DataFrame(contrib_arr, columns=X.columns, index=X_sample.index))

        # 5. The "Classifier Trick" to Enable GUI Dropdown
        # Revert to numeric IDs for compilation (Shapash requirement)
        name_to_id = {name: i for i, name in enumerate(target_names)}
        id_to_name = {i: name for i, name in enumerate(target_names)}
        
        model.classes_ = np.arange(len(target_names))
        model.predict_proba = MultiOutputPredictProba(model)
        
        # Predicted dominant class (Numeric IDs)
        y_pred_dominant_names = Y.idxmax(axis=1)
        y_pred_ids = y_pred_dominant_names.map(name_to_id)
        y_pred_sample = pd.DataFrame(y_pred_ids.loc[X_sample.index].values, columns=['pred'], index=X_sample.index)

        # CRITICAL: Force X_sample to 'object' type. 
        # This prevents the UI from locking the 'feature_value' column to float64,
        # which is what causes the crash when displaying string labels in the ID Card.
        X_sample_flexible = X_sample.astype(object)

        # 6. Initialize and Compile SmartExplainer
        print("\nCompiling Unified SmartExplainer...")
        xpl = SmartExplainer(
            model=model,
            features_dict={c: c for c in X.columns},
            label_dict=id_to_name,
            title_story=f'Unified Acoustic Brain (v{VERSION})'
        )
        
        xpl.compile(
            x=X_sample_flexible,
            contributions=list_contrib_df,
            y_pred=y_pred_sample,
            y_target=y_pred_sample
        )

        # 7. Persistence
        print(f"Saving unified brain to: {brain_path}")
        try:
            xpl.save(brain_path)
        except Exception as e:
            print(f"Warning: Could not save brain: {e}")

    # 8. Launch WebApp
    print("\n--------------------------------------------------")
    print("DASHBOARD STARTING (Unified Acoustic Brain)")
    print("URL: http://localhost:8050")
    print("GUI: Select sound targets from the dropdown.")
    print("--------------------------------------------------")
    
    # We use explicit initialization for robustness
    xpl.init_app()
    app = xpl.smartapp.app
    app.run(debug=False, host="0.0.0.0", port=8050)

if __name__ == "__main__":
    main()
