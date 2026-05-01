
import numpy as np

def analyze_mismatch(duration, frames_num, frames_per_second):
    print(f"--- Input Analysis ---")
    print(f"Metadata Duration: {duration}s")
    print(f"Actual Data Frames: {frames_num}")
    print(f"FPS for Vis: {frames_per_second}")
    print(f"Actual Time in Data: {frames_num / frames_per_second}s")
    
    tick_interval = max(5, int(duration / 20))
    print(f"\n--- Calculation ---")
    print(f"Tick Interval: {tick_interval}s")
    
    x_ticks = np.arange(0, frames_num, frames_per_second * tick_interval)
    x_labels = np.arange(0, int(duration) + 1, tick_interval)
    
    print(f"Number of Ticks: {len(x_ticks)}")
    print(f"Number of Labels: {len(x_labels)}")
    
    if len(x_ticks) != len(x_labels):
        print(f"\033[1;31mCRASH DETECTED: {len(x_ticks)} ticks vs {len(x_labels)} labels\033[0m")
    else:
        print("\033[1;32mNo mismatch found with these parameters.\033[0m")

    # Testing Option 2 Logic: Recalculate duration first
    print(f"\n--- testing Option 2 (Recalculate Duration) ---")
    new_duration = frames_num / frames_per_second
    new_tick_interval = max(5, int(new_duration / 20))
    new_x_ticks = np.arange(0, frames_num, frames_per_second * new_tick_interval)
    new_x_labels = np.arange(0, int(new_duration) + 1, new_tick_interval)
    
    print(f"New Duration: {new_duration}s")
    print(f"New Ticks: {len(new_x_ticks)}")
    print(f"New Labels: {len(new_x_labels)}")
    
    # Note: Even with Option 2, rounding or off-by-one in arange can still happen.
    # np.arange(0, 10, 2) -> [0, 2, 4, 6, 8] (5 elements)
    # np.arange(0, 11, 2) -> [0, 2, 4, 6, 8, 10] (6 elements)
    
    if len(new_x_ticks) != len(new_x_labels):
        print(f"\033[1;33mWarning: Recalculating duration ALONE still has off-by-one: {len(new_x_ticks)} vs {len(new_x_labels)}\033[0m")
    else:
        print("\033[1;32mOption 2 looks stable for these numbers.\033[0m")

# Values from the failing log in problem1.md
analyze_mismatch(duration=1763, frames_num=3772, frames_per_second=2)
