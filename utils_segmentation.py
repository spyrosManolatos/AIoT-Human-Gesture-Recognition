import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def extract_peak_centered_windows(df, ws=128, threshold=1.15, min_dist=100):
    """
    Finds movement peaks in acceleration magnitude and extracts windows centered on them.
    This ensures the 'action' is always in the middle of the window, removing silence.
    
    Args:
        df: Input DataFrame with sensor columns.
        ws: Window size in samples.
        threshold: Magnitude threshold to trigger a 'gesture' (e.g., 1.15g).
        min_dist: Minimum samples between two gestures to avoid double-counting.
    """
    # 1. Calculate Acceleration Magnitude
    # We use this to detect the 'energy' of the movement regardless of orientation
    acc_cols = ["acc_x", "acc_y", "acc_z"]
    mag = np.sqrt((df[acc_cols]**2).sum(axis=1))
    
    # 2. Find Peaks above the threshold
    # Height=threshold ignores gravity (1.0g) and small jitters
    peaks, _ = find_peaks(mag, height=threshold, distance=min_dist)
    
    windows = []
    half_ws = ws // 2
    
    for p in peaks:
        start = p - half_ws
        end = p + half_ws
        
        # Only take the window if it fits within the recording bounds
        if start >= 0 and end <= len(df):
            window = df.iloc[start:end].copy()
            windows.append(window)
            
    return windows
