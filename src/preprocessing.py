import numpy as np
from scipy.signal import butter, filtfilt

def preprocess_signal(raw_signal, fs=250):
    """
    Applies Bandpass (1-45 Hz) and Notch (50 Hz) filtering for baseline wander and powerline removal.
    """
    nyq = 0.5 * fs
    low = 1.0 / nyq
    high = 45.0 / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, raw_signal)
    return filtered_signal
