import numpy as np
from scipy.signal import butter, sosfiltfilt

class BandpassFilter:
    
    def __init__(self, dataset):
        self.data = dataset.data.copy()
        self.indices = dataset.indices.copy()
        self.classes = dataset.classes.copy()
        self.filtered_data = np.array([])

    def apply_band_pass_filter(self, filter_low, filter_high, sample_rate, order):
        
        nyquist = 0.5 * sample_rate
        low = filter_low / nyquist
        high = filter_high / nyquist

        sos = butter(order, [low, high], btype='band', output='sos')
        self.filtered_data = sosfiltfilt(sos, self.data)