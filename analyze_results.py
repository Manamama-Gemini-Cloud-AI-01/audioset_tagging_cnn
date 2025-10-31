import os
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- Configuration ---
# The main CSV file containing the metadata and paths to results
CSV_FILE = "/media/zezen/TOSH-P2_Data_Public/Temp_to_analyze/Street_sounds_archive/UrbanSound8K.csv"

# The file containing the AudioSet class labels and indices
CLASS_LABELS_FILE = "/home/zezen/Downloads/GitHub/audioset_tagging_cnn/metadata/class_labels_indices.csv"

# Mapping from UrbanSound8K class names to the most relevant AudioSet display name
# This is based on the documentation and analysis of the AudioSet ontology.
US8K_TO_AUDIOSET_MAP = {
    'air_conditioner': 'Air conditioning',
    'car_horn': 'Vehicle horn, car horn, honking',
    'children_playing': 'Child speech, kid speaking',
    'dog_bark': 'Bark',
    'drilling': 'Drill',
    'engine_idling': 'Idling',
    'gun_shot': 'Gunshot, gunfire',
    'jackhammer': 'Jackhammer',
    'siren': 'Siren',
    'street_music': 'Music'
}


def analyze_results():
    """
    Analyzes the results of the audio tagging inference, calculates accuracy,
    and generates a confusion matrix.
    """
    if not os.path.exists(CSV_FILE):
        print(f"Error: Master CSV file not found at {CSV_FILE}")
        return

    if not os.path.exists(CLASS_LABELS_FILE):
        print(f"Error: Class labels file not found at {CLASS_LABELS_FILE}")
        return

    # Load the master metadata CSV
    try:
        master_df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Error reading master CSV file: {e}")
        return

    # Load AudioSet labels to create a reverse mapping from display name to index
    try:
        labels_df = pd.read_csv(CLASS_LABELS_FILE)
        audioset_name_to_index = {row['display_name']: row['index'] for _, row in labels_df.iterrows()}
    except Exception as e:
        print(f"Error reading class labels file: {e}")
        return

    # Lists to store true and predicted labels for confusion matrix
    y_true = []
    y_pred = []
    
    correct_predictions = 0
    total_processed = 0

    print("Analyzing results...")

    # Iterate over the rows of the master DataFrame
    for index, row in master_df.iterrows():
        summary_path = str(row.get("summary_csv_path", "")).strip()
        true_class_name = row['class']

        if not summary_path or not os.path.exists(summary_path):
            # Skip rows that haven't been processed yet
            continue

        total_processed += 1
        
        try:
            summary_df = pd.read_csv(summary_path)
        except (pd.errors.EmptyDataError, FileNotFoundError):
            # If the summary is empty or not found, count it as a wrong prediction
            print(f"Warning: Could not read or empty summary for {row['slice_file_name']}")
            y_true.append(true_class_name)
            y_pred.append("no_prediction")
            continue

        if summary_df.empty:
            # If the summary is empty, count it as a wrong prediction
            print(f"Warning: Empty summary_events.csv for {row['slice_file_name']}")
            y_true.append(true_class_name)
            y_pred.append("no_prediction")
            continue

        # Find the prediction with the highest average probability
        top_prediction_row = summary_df.loc[summary_df['average_probability'].idxmax()]
        predicted_class_name = top_prediction_row['sound_class']

        # Get the expected AudioSet class name for the true UrbanSound8K class
        expected_audioset_class = US8K_TO_AUDIOSET_MAP.get(true_class_name)

        y_true.append(true_class_name)
        y_pred.append(predicted_class_name) # For confusion matrix, we can use the direct predicted name

        # Check for a correct prediction
        if expected_audioset_class and expected_audioset_class == predicted_class_name:
            correct_predictions += 1
            print(f"✅ Correct: {row['slice_file_name']} - Predicted: '{predicted_class_name}'")
        else:
            print(f"❌ Incorrect: {row['slice_file_name']} - True: '{true_class_name}', Predicted: '{predicted_class_name}'")

    # --- Final Report ---
    if total_processed == 0:
        print("\nNo processed files found to analyze.")
        return

    accuracy = (correct_predictions / total_processed) * 100
    print(f"\n--- Analysis Complete ---")
    print(f"Total Files Analyzed: {total_processed}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

    # --- Confusion Matrix ---
    print("\nGenerating confusion matrix...")
    
    # Get all unique labels from both true and predicted lists
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
    
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    confusion_matrix_path = os.path.join(os.path.dirname(CSV_FILE), 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    
    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    print("Note: The confusion matrix shows the direct predicted AudioSet labels against the true UrbanSound8K labels.")


if __name__ == "__main__":
    analyze_results()
