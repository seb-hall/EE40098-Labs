################################################################
##
## EE40098 Coursework C
##
## File         :  spikedetector.py
## Author       :  samh25
## Created      :  2025-12-02 (YYYY-MM-DD)
## License      :  MIT
## Description  :  Class for detecting spikes in filtered signals.
##
################################################################

################################################################
## MARK: INCLUDES
################################################################

import numpy as np
from scipy.signal import find_peaks

################################################################
## MARK: CLASS DEFINITIONS
################################################################

class SpikeDetector:

    ############################################################
    ## CONSTRUCTOR

    # instantiate a new spike detector
    def __init__(self, filtered_dataset):
        # copy dataset contents so we dont modify original
        self.data = filtered_dataset.filtered_data.copy()
        self.indices = filtered_dataset.indices.copy()
        self.classes = filtered_dataset.classes.copy()

        self.detected_spikes = np.array([])

    ############################################################
    ## INSTANCE METHODS

    # calculate Median Absolute Deviation of the data
    def calculate_mad(self):
        
        # Median Absolute Deviation calculation
        return np.median(np.abs(self.data)) / 0.6745
    

    # detect spikes using MAD-based thresholding
    def detect_spikes(self, mad, mad_gain, distance):

        # Spike detection using MAD-based thresholding
        threshold = mad_gain * mad

        # distance is minimum number of samples between peaks
        peaks, _ = find_peaks(self.data, height=threshold, distance=distance)

        self.detected_spikes = peaks
