import os
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- Configuration ---
CSV_FILE = "/media/zezen/TOSH-P2_Data_Public/Temp_to_analyze/Street_sounds_archive/UrbanSound8K.csv"
CLASS_LABELS_FILE = "/home/zezen/Downloads/GitHub/audioset_tagging_cnn/metadata/class_labels_indices.csv"

# Flexible mapping from UrbanSound8K class names to a list of acceptable AudioSet display names
US8K_TO_AUDIOSET_MAP = {
    'air_conditioner': ['Air conditioning'],
    'car_horn': ['Vehicle horn, car horn, honking', 'Toot', 'Car alarm'],
    'children_playing': ['Child speech, kid speaking', 'Children playing', 'Speech', 'Laughter', 'Giggle', 'Run'],
    'dog_bark': ['Bark', 'Dog', 'Bow-wow', 'Growling', 'Howl', 'Yip', 'Whimper (dog)'],
    'drilling': ['Drill', 'Dental drill, dentist\'s drill', 'Tools'],
    'engine_idling': ['Idling', 'Engine', 'Light engine (high frequency)', 'Medium engine (mid frequency)', 'Heavy engine (low frequency)'],
    'gun_shot': ['Gunshot, gunfire', 'Machine gun', 'Fusillade', 'Cap gun'],
    'jackhammer': ['Jackhammer'],
    'siren': ['Siren', 'Emergency vehicle', 'Police car (siren)', 'Ambulance (siren)', 'Fire engine, fire truck (siren)', 'Civil defense siren'],
    'street_music': ['Music', 'Street music', 'Busking', 'Accordion', 'Singing', 'Plucked string instrument', 'Violin, fiddle']
}

def analyze_results_properly():
    """
    Analyzes the results by reading the full_event_log.csv for each file,
    finding the single event with the highest peak probability, and using that as the prediction.
    """
    if not os.path.exists(CSV_FILE):
        print(f"Error: Master CSV file not found at {CSV_FILE}")
        return

    try:
        master_df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Error reading master CSV file: {e}")
        return

    y_true = []
    y_pred = []
    correct_predictions = 0
    total_processed = 0

    print("Analyzing results from full_event_log.csv files...")

    for index, row in master_df.iterrows():
        summary_path = str(row.get("summary_csv_path", "")).strip()
        true_class_name = row['class']

        if not summary_path or not os.path.exists(os.path.dirname(summary_path)):
            continue

        total_processed += 1
        full_log_path = os.path.join(os.path.dirname(summary_path), "full_event_log.csv")

        if not os.path.exists(full_log_path):
            print(f"Warning: full_event_log.csv not found for {row['slice_file_name']}")
            y_true.append(true_class_name)
            y_pred.append("no_prediction")
            continue

        try:
            log_df = pd.read_csv(full_log_path)
            if log_df.empty:
                print(f"Warning: Empty full_event_log.csv for {row['slice_file_name']}")
                y_true.append(true_class_name)
                y_pred.append("no_prediction")
                continue
        except (pd.errors.EmptyDataError, FileNotFoundError):
            print(f"Warning: Could not read or empty full_event_log for {row['slice_file_name']}")
            y_true.append(true_class_name)
            y_pred.append("no_prediction")
            continue

        # Find the single best prediction (highest probability peak)
        top_prediction_row = log_df.loc[log_df['probability'].idxmax()]
        predicted_class_name = top_prediction_row['sound']

        acceptable_predictions = US8K_TO_AUDIOSET_MAP.get(true_class_name, [])

        y_true.append(true_class_name)
        y_pred.append(predicted_class_name)

        if predicted_class_name in acceptable_predictions:
            correct_predictions += 1
            print(f"✅ Correct: {row['slice_file_name']} - True: '{true_class_name}', Predicted: '{predicted_class_name}' (Accepted)")
        else:
            print(f"❌ Incorrect: {row['slice_file_name']} - True: '{true_class_name}', Predicted: '{predicted_class_name}'")

    if total_processed == 0:
        print("\nNo processed files found to analyze.")
        return

    accuracy = (correct_predictions / total_processed) * 100
    print(f"\n--- Proper Analysis Complete ---")
    print(f"Total Files Analyzed: {total_processed}")
    print(f"Correct Predictions (Flexible, Peak): {correct_predictions}")
    print(f"Accuracy (Flexible, Peak): {accuracy:.2f}%")

    # --- Confusion Matrix ---
    if not y_true:
        print("\nNo valid predictions were made, skipping confusion matrix generation.")
        return

    print("\nGenerating and exporting new confusion matrix...")
labels = sorted(list(set(y_true) | set(y_pred)))
cm = confusion_matrix(y_true, y_pred, labels=labels)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
confusion_matrix_csv_path = os.path.join(os.path.dirname(CSV_FILE), 'confusion_matrix_proper.csv')
cm_df.to_csv(confusion_matrix_csv_path)
print(f"Proper confusion matrix data saved to: {confusion_matrix_csv_path}")

fig, ax = plt.subplots(figsize=(12, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')
plt.title('Confusion Matrix (Proper Analysis - Peak Probability)')
plt.tight_layout()
confusion_matrix_png_path = os.path.join(os.path.dirname(CSV_FILE), 'confusion_matrix_proper.png')
plt.savefig(confusion_matrix_png_path)
print(f"Proper confusion matrix image saved to: {confusion_matrix_png_path}")

if __name__ == "__main__":
    analyze_results_properly()
