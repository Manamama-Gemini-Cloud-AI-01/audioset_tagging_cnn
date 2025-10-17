import csv

def find_scariest_music(csv_file_path):
    max_probability = -1.0
    scariest_segment = None

    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['sound'] == 'Scary music':
                probability = float(row['probability'])
                if probability > max_probability:
                    max_probability = probability
                    scariest_segment = {
                        'start_time': row['start_time'],
                        'end_time': row['end_time'],
                        'probability': probability
                    }
    
    if scariest_segment:
        print(f"The 'Scary music' is scariest between {scariest_segment['start_time']} and {scariest_segment['end_time']} with a probability of {scariest_segment['probability']:.3f}.")
    else:
        print("No 'Scary music' found in the CSV file.")

if __name__ == "__main__":
    csv_file = "results/Terminator_1984_Best.csv"
    find_scariest_music(csv_file)
