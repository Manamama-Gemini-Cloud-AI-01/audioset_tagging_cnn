
import pandas as pd
import plotly.graph_objects as go
import os

# Path configuration
csv_path = '/home/zezen/Downloads/Test/kukea joy_Cnn14_DecisionLevelMax_mAP=0.385.pth_audioset_tagging_cnn/full_event_log.csv'
output_dir = os.path.dirname(csv_path)
html_output = os.path.join(output_dir, 'top_50_events_line_graph.html')

print(f"Reading matrix from {csv_path}...")
df = pd.read_csv(csv_path)

# 1. Identify top 50 classes by popularity (sum of probabilities)
# Exclude 'time' column for calculations
labels = [col for col in df.columns if col != 'time']
popularity = df[labels].sum().sort_values(ascending=False)
top_50_labels = popularity.head(50).index.tolist()

print(f"Generating Plotly line graph for top 50 events...")

fig = go.Figure()

for label in top_50_labels:
    # We only show the top 10 by default to keep the initial view readable
    is_initially_visible = label in top_50_labels[:10]
    
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df[label],
        mode='lines',
        name=label,
        visible=True if is_initially_visible else 'legendonly'
    ))

fig.update_layout(
    title="Top 50 Sound Events Over Time (kukea joy)",
    xaxis_title="Seconds",
    yaxis_title="Probability",
    legend_title="Sound Classes (Click to Toggle)",
    hovermode="x unified",
    template="plotly_white",
    height=800
)

# Save as HTML
fig.write_html(html_output)
print(f"✅ Success! Interactive graph saved to: {html_output}")
print(f"Initially visible: {', '.join(top_50_labels[:10])}")
