import numpy as np
import matplotlib.pyplot as plt
import optimap as om
import math
import random
import numpy as np
from scipy.signal import correlate
import numba as nb
from tqdm import tqdm

@nb.njit
def normalize(series):
    min_val = np.min(series)
    max_val = np.max(series)
    if max_val == min_val:
        return np.zeros_like(series)
    return (series - min_val) / (max_val - min_val)

@nb.njit
def fast_corrcoef(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    numerator = np.sum(x_centered * y_centered)
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    
    if denominator == 0:
        return 0.0
    return numerator / denominator

@nb.njit
def process_pixel(v_series, cb_series, window_size=20, step_size=1):
    v_series_normalized = normalize(v_series)
    cb_series_normalized = normalize(cb_series)
    
    # Calculate global cross-correlation
    n = len(v_series_normalized)
    max_corr = -2.0  # Correlation is between -1 and 1
    global_time_delay = 0
    
    # Only check reasonable lags
    max_lag = min(n // 5, 50)
    
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            corr = fast_corrcoef(v_series_normalized[lag:], cb_series_normalized[:-lag])
        elif lag < 0:
            corr = fast_corrcoef(v_series_normalized[:lag], cb_series_normalized[-lag:])
        else:
            corr = fast_corrcoef(v_series_normalized, cb_series_normalized)
            
        if corr > max_corr:
            max_corr = corr
            global_time_delay = lag
    
    # Shift calcium signal
    if global_time_delay > 0:
        cb_shifted = np.zeros_like(cb_series_normalized)
        cb_shifted[global_time_delay:] = cb_series_normalized[:-global_time_delay]
    elif global_time_delay < 0:
        cb_shifted = np.zeros_like(cb_series_normalized)
        cb_shifted[:global_time_delay] = cb_series_normalized[-global_time_delay:]
    else:
        cb_shifted = cb_series_normalized
    
    # Windowed correlation analysis
    n_windows = len(v_series_normalized) - window_size + 1
    window_correlations = np.zeros(n_windows // step_size)
    
    for i in range(0, n_windows, step_size):
        idx = i // step_size
        v_window = v_series_normalized[i:i+window_size]
        cb_window = cb_shifted[i:i+window_size]
        window_correlations[idx] = fast_corrcoef(v_window, cb_window)
    
    return global_time_delay, window_correlations, cb_shifted

def process_matrices(v_out, cb_out, window_size=20, step_size=1):
    """Process matrices where dimensions are (time, x, y)"""
    assert v_out.shape == cb_out.shape, "Input matrices must have the same shape"
    
    time_points, width, height = v_out.shape
    n_windows = time_points - window_size + 1
    n_output_points = (n_windows + step_size - 1) // step_size
    
    # Create output arrays with correct dimensions
    global_delays = np.zeros((width, height), dtype=np.int32)  # 2D spatial map
    # Window correlations will have fewer time points due to windowing
    windowed_correlations = np.zeros((n_output_points, width, height))  
    ca_delayed = np.zeros((n_output_points, width, height))  
    
    # Process each spatial point
    for i in tqdm(range(width)):
        for j in range(height):
            # Extract time series for this spatial point
            v_series = v_out[:, i, j]  # Time series at position (i,j)
            cb_series = cb_out[:, i, j]
            
            # Process this pixel
            global_delay, window_corrs,cb_shifted = process_pixel(
                v_series, cb_series, window_size, step_size
            )
            
            # Store results
            global_delays[i, j] = global_delay
            # Store windowed correlations with time as first dimension
            windowed_correlations[:, i, j] = window_corrs
            # ca_delayed[:,i,j] = cb_shifted
    
    return global_delays, windowed_correlations, ca_delayed



