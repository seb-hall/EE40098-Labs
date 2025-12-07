import numpy as np
from scipy.signal import find_peaks

class SpikeDetector:

    def __init__(self, filtered_dataset):
        self.data = filtered_dataset.filtered_data
        self.indices = filtered_dataset.indices
        self.classes = filtered_dataset.classes

        self.detected_spikes = np.array([])

    def calculate_mad(self):
        
        # Median Absolute Deviation calculation
        return np.median(np.abs(self.data)) / 0.6745

    def detect_spikes(self, mad, mad_gain, distance):

        # Spike detection using MAD-based thresholding
        threshold = mad_gain * mad

        # distance is minimum number of samples between peaks
        peaks, _ = find_peaks(self.data * -1.0, height=threshold, distance=distance)

        self.detected_spikes = peaks