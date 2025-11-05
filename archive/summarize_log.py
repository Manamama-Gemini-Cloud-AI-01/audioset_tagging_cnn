import pandas as pd
import argparse
import os

def summarize_full_log(log_file_path):
    """
    Parses a full_event_log.csv to find continuous event blocks for each sound class
    and saves them to a new, detailed CSV file.
    """
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at {log_file_path}")
        return

    try:
        log_df = pd.read_csv(log_file_path)
        if log_df.empty:
            print("Log file is empty. No summary to generate.")
            return
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    # --- Configuration for event block detection ---
    onset_threshold = 0.3  # Probability to start an event
    offset_threshold = 0.15 # Probability to end an event
    min_event_duration_seconds = 0.1 # Minimum duration for a block to be recorded

    all_event_blocks = []

    # Group by sound class to process each one individually
    for sound_class, group in log_df.groupby('sound'):
        in_event = False
        event_start_time = 0
        event_probs = []

        # Sort by time to ensure correct order
        group = group.sort_values(by='time')

        for index, row in group.iterrows():
            time = row['time']
            prob = row['probability']

            if not in_event and prob > onset_threshold:
                # Start of a new event block
                in_event = True
                event_start_time = time
                event_probs.append(prob)

            elif in_event:
                if prob > offset_threshold:
                    # Continue the current event block
                    event_probs.append(prob)
                else:
                    # End of the current event block
                    in_event = False
                    event_end_time = time
                    duration = event_end_time - event_start_time

                    if duration >= min_event_duration_seconds:
                        all_event_blocks.append({
                            'sound_class': sound_class,
                            'start_time': round(event_start_time, 3),
                            'end_time': round(event_end_time, 3),
                            'duration': round(duration, 3),
                            'peak_probability': max(event_probs),
                            'average_probability': sum(event_probs) / len(event_probs)
                        })
                    # Reset for the next block
                    event_probs = []

        # Handle case where the event is still ongoing at the end of the file
        if in_event and event_probs:
            event_end_time = group['time'].iloc[-1]
            duration = event_end_time - event_start_time
            if duration >= min_event_duration_seconds:
                all_event_blocks.append({
                    'sound_class': sound_class,
                    'start_time': round(event_start_time, 3),
                    'end_time': round(event_end_time, 3),
                    'duration': round(duration, 3),
                    'peak_probability': max(event_probs),
                    'average_probability': sum(event_probs) / len(event_probs)
                })

    if not all_event_blocks:
        print("No event blocks found meeting the threshold criteria.")
        return

    # Create a DataFrame from the detected blocks and sort it
    summary_df = pd.DataFrame(all_event_blocks)
    summary_df = summary_df.sort_values(by=['start_time', 'average_probability'], ascending=[True, False])

    # Save the new detailed summary
    output_path = os.path.join(os.path.dirname(log_file_path), 'detailed_summary.csv')
    summary_df.to_csv(output_path, index=False)

    print(f"Successfully generated detailed summary at: {output_path}")
    print("\n--- Top 10 Event Blocks ---")
    print(summary_df.head(10).to_string())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a detailed, human-readable summary from a full_event_log.csv file.')
    parser.add_argument('log_file', type=str, help='Path to the full_event_log.csv file to analyze.')
    args = parser.parse_args()
    summarize_full_log(args.log_file)
