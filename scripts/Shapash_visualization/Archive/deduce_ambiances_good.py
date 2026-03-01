import pandas as pd
import numpy as np

csv_path = "/home/zezen/Videos/Videos_on_P7/Metallica_-_Master_o_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/full_event_log.csv"
print("loading " + csv_path)
df = pd.read_csv(csv_path)

# 1. Filter for active columns (> 0.005 mean prob to remove silence/noise)
sounds_df = df.drop(columns=["time"])
active_cols = sounds_df.mean()[sounds_df.mean() > 0.005].index.tolist()
sounds_active = sounds_df[active_cols]

print("analyzing correlations for " + str(len(active_cols)) + " active sound classes")

# 2. Calculate the Correlation Matrix (Acoustic Co-occurrence)
corr_matrix = sounds_active.corr()

# 3. Extract the strongest "Ambiance Clusters"
# We look for pairs with correlation > 0.6
print("deducing ambiances from co-occurrence clusters")

seen = set()
clusters = []

for col in corr_matrix.columns:
    if col in seen:
        continue
    
    # Find all sounds strongly correlated with this one
    correlates = corr_matrix[col][corr_matrix[col] > 0.6].index.tolist()
    
    if len(correlates) > 1:
        clusters.append(correlates)
        for c in correlates:
            seen.add(c)

# 4. Print the deduced "Acoustic Scenes"
for i, cluster in enumerate(clusters):
    print("scene " + str(i+1) + " (deduced ambiance):")
    for sound in cluster:
        # Check if it's generally quiet or loud in this cluster
        avg_p = sounds_active[sound].mean()
        print("  - " + sound + " (avg prob: " + str(round(avg_p, 3)) + ")")

# 5. Save the full matrix for your own analysis
corr_matrix.to_csv("acoustic_co-occurrence_matrix.csv")
print("full relationship matrix saved to acoustic_co-occurrence_matrix.csv")
