import h5py
import pandas as pd
import numpy as np
import datetime

h5_path = "/mnt/HP_P7_Data/Videos/Watch/I_Exposed_My_Biggest_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/full_event_log.h5"

# Based on project context, UI FPS is typically 5 Hz
UI_FPS = 5 

with h5py.File(h5_path, 'r') as f:
    data = f['framewise_output'][:]
    
    # Class 16 is Laughter
    probs = data[:, 16]
    
    # Find frames where prob > 0.01
    indices = np.where(probs > 0.01)[0]
    
    # Group into segments to avoid printing every single frame
    timestamps = []
    if len(indices) > 0:
        # Simple segmenter: if indices are close, they are the same event
        # Assuming 5Hz, 5 frames is 1 second
        segments = np.split(indices, np.where(np.diff(indices) > 5)[0] + 1)
        
        for seg in segments:
            start_frame = seg[0]
            start_time_sec = start_frame / UI_FPS
            timestamps.append(start_time_sec)

    print("Detected Laughter segments (up to 10):")
    for t in timestamps[:10]:
        print(datetime.timedelta(seconds=int(t)))
