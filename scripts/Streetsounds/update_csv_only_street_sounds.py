import os
import pandas as pd

# --- Configuration ---
# The directory containing the audio files and their output folders
AUDIO_DIR = "/media/zezen/TOSH-P2_Data_Public/Temp_to_analyze/Street_sounds_archive/fold1"

# The main CSV file containing the metadata
CSV_FILE = "/media/zezen/TOSH-P2_Data_Public/Temp_to_analyze/Street_sounds_archive/UrbanSound8K.csv"

# --- Script ---

def update_csv_with_existing_results():
    """
    Scans for existing inference results and updates the master CSV with their paths.
    Does NOT run any new inference.
    """
    if not os.path.exists(AUDIO_DIR):
        print(f"Error: Audio directory not found at {AUDIO_DIR}")
        return

    if not os.path.exists(CSV_FILE):
        print(f"Error: CSV file not found at {CSV_FILE}")
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

    # Filter for the files in the specified fold (assuming fold1 for now)
    # We iterate through the original DataFrame to ensure we cover all entries
    # and update them in place.
    
    updated_rows_count = 0

    print("Scanning for existing results and updating CSV...")

    for index, row in df.iterrows():
        # Only consider files that belong to the AUDIO_DIR (fold1 in this case)
        # This check is important if the CSV contains entries for other folds
        if row["fold"] != 1: # Assuming AUDIO_DIR is specifically for fold1
            continue

        slice_filename = row["slice_file_name"]
        
        # Construct the expected output directory name
        base_filename_no_ext = os.path.splitext(slice_filename)[0]
        output_dir_name = f"{base_filename_no_ext}_audioset_tagging_cnn"
        full_output_dir_path = os.path.join(AUDIO_DIR, output_dir_name)

        # Construct the expected summary_events.csv path
        expected_summary_csv_path = os.path.join(full_output_dir_path, "summary_events.csv")

        # Check if the summary_events.csv actually exists
        if os.path.exists(expected_summary_csv_path):
            # Check if the CSV entry is already correct to avoid unnecessary updates
            current_summary_path_in_df = str(row.get("summary_csv_path", "")).strip()
            if current_summary_path_in_df != expected_summary_csv_path:
                df.loc[index, "summary_csv_path"] = expected_summary_csv_path
                updated_rows_count += 1
                print(f"  Updated entry for {slice_filename} with path: {expected_summary_csv_path}")
        else:
            # If the summary_csv_path was previously set but the file is gone, clear it.
            # Or if it was never set and the file doesn't exist, ensure it's empty.
            if pd.notna(row.get("summary_csv_path")) and row["summary_csv_path"]:
                df.loc[index, "summary_csv_path"] = ""
                updated_rows_count += 1
                print(f"  Cleared summary_csv_path for {slice_filename} (file not found).")

    # Save the updated DataFrame back to the CSV file
    if updated_rows_count > 0:
        try:
            df.to_csv(CSV_FILE, index=False)
            print(f"\nSuccessfully updated {updated_rows_count} rows in {CSV_FILE}")
        except Exception as e:
            print(f"Error writing updated CSV file: {e}")
    else:
        print("\nNo new updates needed for the CSV file.")

if __name__ == "__main__":
    update_csv_with_existing_results()
