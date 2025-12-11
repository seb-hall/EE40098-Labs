################################################################
##
## EE40098 Coursework C
##
## File         :  filter.py
## Author       :  samh25
## Created      :  2025-12-02 (YYYY-MM-DD)
## License      :  MIT
## Description  :  (Currently unused) Bandpass filter and wavelet denoising.
##
################################################################

################################################################
## MARK: INCLUDES
################################################################s

import numpy as np
from scipy.signal import butter, sosfiltfilt
import pywt

################################################################
## MARK: CLASS DEFINITIONS
################################################################

class BandpassFilter:
    
    ############################################################
    ## CONSTRUCTOR

    # instantiate a new bandpass filter
    def __init__(self, dataset):
        # copy dataset contents so we dont modify original
        self.data = dataset.data.copy()
        self.indices = dataset.indices.copy()
        self.classes = dataset.classes.copy()
        self.filtered_data = dataset.data.copy()


    ############################################################    
    ## INSTANCE METHODS

    # apply a band-pass filter to the data
    def apply_band_pass_filter(self, filter_low, filter_high, sample_rate, order):

        nyquist = 0.5 * sample_rate
        low = filter_low / nyquist
        high = filter_high / nyquist

        # use butterworth bandpass filter
        sos = butter(order, [low, high], btype='band', output='sos')
        self.filtered_data = sosfiltfilt(sos, self.data)


    # apply a wavelet denoising to the data
    def apply_wavelet_denoise(self, wavelet='db4', level=4, alpha=1.0):

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
        