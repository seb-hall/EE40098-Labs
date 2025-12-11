################################################################
##
## EE40098 Coursework C
##
## File         :  noisydata.py
## Author       :  samh25
## Created      :  2025-12-08 (YYYY-MM-DD)
## License      :  MIT
## Description  :  Class for adding noise to data.
##
################################################################

################################################################
## MARK: INCLUDES
################################################################

import numpy as np

################################################################
## MARK: CLASS DEFINITIONS
################################################################

class NoisyData:

    ############################################################
    ## CONSTRUCTOR

    # instantiate a new noisy data object
    def __init__(self, raw_dataset):
        # copy dataset contents so we dont modify original
        self.data = raw_dataset.data.copy()
        self.indices = raw_dataset.indices.copy()
        self.classes = raw_dataset.classes.copy()

    ############################################################
    ## INSTANCE METHODS

    # add linear Gaussian noise to data
    def noisify(self, noise_ratio):
        signal_peak = np.max(np.abs(self.data))
        noise_sigma = signal_peak * noise_ratio
        noise = np.random.normal(0, noise_sigma, self.data.shape)
        
        # add noise to data
        self.data += noise
        