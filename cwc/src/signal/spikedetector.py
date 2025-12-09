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

    def adaptive_detect_spikes(self, distance=60):
        """
        Automatically determine threshold based on noise characteristics
        """
        mad = self.calculate_mad()
        
        # Estimate SNR from signal statistics
        signal_peaks = self.data[self.data < -3*mad]  # Likely spikes
        if len(signal_peaks) > 10:
            signal_strength = np.abs(np.percentile(signal_peaks, 5))  # 5th percentile of peaks
            snr_estimate = signal_strength / mad
        else:
            snr_estimate = 10  # Default assumption
        
        # Adaptive threshold
        if snr_estimate > 15:     # Very clean (D2)
            mad_gain = 2.8
        elif snr_estimate > 8:    # Clean (D3)
            mad_gain = 2.3
        elif snr_estimate > 4:    # Moderate noise (D4)
            mad_gain = 2.0
        elif snr_estimate > 2:    # Noisy (D5)
            mad_gain = 1.7
        else:                     # Very noisy (D6)
            mad_gain = 1.5
        
        threshold = mad_gain * mad
        peaks, _ = find_peaks(self.data * -1.0, height=threshold, distance=distance)
        
        self.detected_spikes = peaks
        print(f"SNR estimate: {snr_estimate:.2f}, MAD gain: {mad_gain:.2f}")