################################################################
##
## EE40098 Coursework C
##
## File         :  datanormaliser.py
## Author       :  samh25
## Created      :  2025-12-9 (YYYY-MM-DD)
## License      :  MIT
## Description  :  Data normalisation class.
##
################################################################

################################################################
## MARK: INCLUDES
################################################################

import numpy as np

################################################################
## MARK: CLASS DEFINITIONS
################################################################

class DataNormaliser:

    ############################################################
    ## CONSTRUCTOR

    # instantiate a new data normaliser
    def __init__(self, raw_dataset):

        # copy dataset contents so we dont modify original
        self.data = raw_dataset.data.copy()
        self.indices = raw_dataset.indices.copy()
        self.classes = raw_dataset.classes.copy()

    ############################################################
    ## INSTANCE METHODS

    # create test data with a sine wave offset
    def un_normalise(self, gain, cycles):

        dataLength = len(self.data)
        sinePhase = np.linspace(0, 2 * np.pi * cycles, dataLength)
        offsetWave = np.sin(sinePhase) * gain
        self.data += offsetWave

    
    # normalise data in windows using IQR mean
    def normalise(self, window_size=100):
        
        # calculate global IQR mean
        q75, q25 = np.percentile(self.data, [75 ,25])
        iqr_mask = (self.data >= q25) & (self.data <= q75)
        iqr_mean = np.mean(self.data[iqr_mask])

        # calculate number of windows required
        num_windows = int(np.ceil(len(self.data) / window_size))

        # now iterate over windows and normalise each   
        for w in range(num_windows):

            # get window
            start_idx = w * window_size
            end_idx = min((w + 1) * window_size, len(self.data))
            window = self.data[start_idx:end_idx]

            # calculate window IQR mean
            window_q75, window_q25 = np.percentile(window, [75 ,25])
            window_iqr_mask = (window >= window_q25) & (window <= window_q75)
            window_iqr_mean = np.mean(window[window_iqr_mask])

            # apply adjustment
            adjustment = iqr_mean - window_iqr_mean
            self.data[start_idx:end_idx] += adjustment
