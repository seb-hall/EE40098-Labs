import numpy as np
from scipy.signal import butter, sosfiltfilt
import pywt

class BandpassFilter:
    
    def __init__(self, dataset):
        self.data = dataset.data.copy()
        self.indices = dataset.indices.copy()
        self.classes = dataset.classes.copy()
        self.filtered_data = dataset.data.copy()

    def apply_band_pass_filter(self, filter_low, filter_high, sample_rate, order):

        nyquist = 0.5 * sample_rate
        low = filter_low / nyquist
        high = filter_high / nyquist

        sos = butter(order, [low, high], btype='band', output='sos')
        self.filtered_data = sosfiltfilt(sos, self.data)

    def appply_wavelet_denoise(self, wavelet='db4', level=4, alpha=1.0):
        """
        Denoise using wavelet decomposition
        """

        print("data shape in wavelet denoise:", self.data.shape)

        # Decompose
        coeffs = pywt.wavedec(self.data, wavelet, level=level)
        
        # Estimate noise from first level details
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Threshold each level
        threshold = alpha * sigma * np.sqrt(2 * np.log(len(self.data)))
        
        # Soft thresholding
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        
        # Reconstruct
        denoised = pywt.waverec(coeffs_thresholded, wavelet)
        
        self.filtered_data = denoised[:len(self.data)]