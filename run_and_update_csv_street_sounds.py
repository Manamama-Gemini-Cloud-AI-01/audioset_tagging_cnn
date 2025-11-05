import os
import subprocess
import pandas as pd

# --- Configuration ---
# The directory containing the audio files to process
AUDIO_DIR = "/media/zezen/TOSH-P2_Data_Public/Temp_to_analyze/Street_sounds_archive/fold1"

# The main CSV file containing the metadata
CSV_FILE = "/media/zezen/TOSH-P2_Data_Public/Temp_to_analyze/Street_sounds_archive/UrbanSound8K.csv"

# The Python script to run for inference
INFERENCE_SCRIPT = "/home/zezen/Downloads/GitHub/audioset_tagging_cnn/pytorch/audioset_tagging_cnn_inference_6.py"

# The model to use
MODEL_TYPE = "Cnn14_DecisionLevelMax"

# The path to the model checkpoint
CHECKPOINT_PATH = "/home/zezen/Downloads/GitHub/audioset_tagging_cnn/Cnn14_DecisionLevelMax_mAP=0.385.pth"

# --- Script ---

def run_inference_and_update_csv():
    """
    Runs audio tagging inference on files in a directory, and updates a CSV with the results.
    """
    if not os.path.exists(AUDIO_DIR):
        print(f"Error: Audio directory not found at {AUDIO_DIR}")
        return

    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file not found at {CSV_FILE}")
        return

    if not os.path.exists(INFERENCE_SCRIPT):
        print(f"Error: Inference script not found at {INFERENCE_SCRIPT}")
        return

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        return

    # Load the metadata CSV
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Add a new column for the summary CSV path if it doesn't exist
    if "summary_csv_path" not in df.columns:
        df["summary_csv_path"] = ""

    # Filter for the files in the specified fold
    fold_df = df[df["fold"] == 1].copy()

    # Iterate over the audio files in the fold
    for index, row in fold_df.iterrows():
        slice_filename = row["slice_file_name"]
        audio_file_path = os.path.join(AUDIO_DIR, slice_filename)

        if not os.path.exists(audio_file_path):
            print(f"Warning: Audio file not found: {audio_file_path}")
            continue

        print(f"Processing {audio_file_path}...")

        # Construct the command to run the inference script
        command = [
            "python3",
            INFERENCE_SCRIPT,
            audio_file_path,
            "--model_type",
            MODEL_TYPE,
            "--checkpoint_path",
            CHECKPOINT_PATH,
            "--cuda" # Assuming you want to use CUDA if available
        ]

        # Run the inference script
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running inference on {slice_filename}:")
            print(e.stderr)
            continue

        # Determine the path to the generated summary_events.csv
        output_dir_name = f"{os.path.splitext(slice_filename)[0]}_audioset_tagging_cnn"
        summary_csv_path = os.path.join(AUDIO_DIR, output_dir_name, "summary_events.csv")

        if os.path.exists(summary_csv_path):
            # Update the DataFrame with the path to the summary CSV
            df.loc[index, "summary_csv_path"] = summary_csv_path
            print(f"  Updated CSV with summary path: {summary_csv_path}")
        else:
            print(f"  Warning: summary_events.csv not found for {slice_filename}")

    # Save the updated DataFrame back to the CSV file
    try:
        df.to_csv(CSV_FILE, index=False)
        print(f"\nSuccessfully updated {CSV_FILE}")
    except Exception as e:
        print(f"Error writing updated CSV file: {e}")

if __name__ == "__main__":
    run_inference_and_update_csv()
