import pandas as pd
import os
import argparse
from shapash import SmartExplainer

def main():
    parser = argparse.ArgumentParser(description="Peek at a precomputed Shapash Acoustic Brain")
    parser.add_argument("brain_path", help="Path to the .pkl file")
    parser.add_argument("--launch", action="store_true", help="Launch the interactive web dashboard")
    
    args = parser.parse_args()

    if not os.path.exists(args.brain_path):
        print(f"Error: Brain file not found: {args.brain_path}")
        return

    print(f"🧠 Loading Acoustic Brain: {args.brain_path}...")
    
    # Load the precomputed explainer
    xpl = SmartExplainer.load(args.brain_path)
    
    # Display the summary table (instant)
    print("\n--- TOP ACOUSTIC WEIGHTS (RECALLED FROM BRAIN) ---")
    if isinstance(xpl.features_imp, pd.Series):
        print(xpl.features_imp.sort_values(ascending=False).head(10))
    else:
        print(xpl.features_imp)
    print("--------------------------------------------------\n")

    if args.launch:
        print("initializing webapp...")
        xpl.init_app()
        app = xpl.smartapp.app
        print("DASHBOARD STARTING...")
        print("Open your browser at: http://localhost:8050")
        app.run(debug=False, host="0.0.0.0", port=8050)

if __name__ == "__main__":
    main()
