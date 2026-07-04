import h5py
import pandas as pd
import numpy as np
import json

h5_path = "/mnt/HP_P7_Data/Videos/Watch/I_Exposed_My_Biggest_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/full_event_log.h5"

# Load the class labels to map indices
labels_path = "class_labels_indices.csv"
labels_df = pd.read_csv(labels_path)

# Find indices for laughter
laughter_indices = labels_df[labels_df['display_name'].str.contains('Laughter', case=False, na=False)].index.tolist()
print(f"Indices for Laughter: {laughter_indices}")

with h5py.File(h5_path, 'r') as f:
    # Based on previous project experience, data is likely under a 'framewise_output' key
    if 'framewise_output' in f:
        data = f['framewise_output'][:]
        
        # Extract laughter columns
        for idx in laughter_indices:
            probs = data[:, idx]
            print(f"--- Class: {labels_df.iloc[idx]['display_name']} (Index {idx}) ---")
            print(f"Max probability: {np.max(probs)}")
            print(f"Mean probability: {np.mean(probs)}")
            print(f"Frames with prob > 0.01: {np.sum(probs > 0.01)}")
    else:
        print("Dataset 'framewise_output' not found in HDF5 file.")
        print(f"Available keys: {list(f.keys())}")
