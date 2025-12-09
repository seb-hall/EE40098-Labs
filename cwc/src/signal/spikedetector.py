import numpy as np
from scipy.signal import find_peaks

class SpikeDetector:

    def __init__(self, filtered_dataset):
        self.data = filtered_dataset.filtered_data.copy()
        self.indices = filtered_dataset.indices.copy()
        self.classes = filtered_dataset.classes.copy()

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

    def two_pass_detection(self, mad, initial_gain=2.5, secondary_gain=2.0, distance=60):
        # First pass: conservative (higher threshold)
        peaks1, _ = find_peaks(self.data * -1.0, 
                            height=initial_gain * mad, 
                            distance=distance)
        
        # Subtract detected spikes (template subtraction)
        residual = self.data.copy()
        for peak in peaks1:
            # Zero out region around detected spike
            start = max(0, peak - 32)
            end = min(len(residual), peak + 32)
            residual[start:end] = 0
        
        # Second pass: more aggressive on residual
        peaks2, _ = find_peaks(residual * -1.0, 
                            height=secondary_gain * mad, 
                            distance=distance)
        
        self.detected_spikes = np.sort(np.concatenate([peaks1, peaks2]))