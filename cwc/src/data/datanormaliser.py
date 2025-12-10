import numpy as np

class DataNormaliser:

    def __init__(self, raw_dataset):
        self.data = raw_dataset.data.copy()
        self.indices = raw_dataset.indices.copy()
        self.classes = raw_dataset.classes.copy()

    def un_normalise(self, gain, cycles):

        dataLength = len(self.data)
        sinePhase = np.linspace(0, 2 * np.pi * cycles, dataLength)
        offsetWave = np.sin(sinePhase) * gain
        
        self.data += offsetWave

    def normalise(self, window_size=100):
        
        # calculate IQR mean
        q75, q25 = np.percentile(self.data, [75 ,25])
        iqr_mask = (self.data >= q25) & (self.data <= q75)
        iqr_mean = np.mean(self.data[iqr_mask])

        # now iterate over windows and normalise each   
        num_windows = int(np.ceil(len(self.data) / window_size))

        for w in range(num_windows):
            start_idx = w * window_size
            end_idx = min((w + 1) * window_size, len(self.data))
            window = self.data[start_idx:end_idx]

            window_q75, window_q25 = np.percentile(window, [75 ,25])
            window_iqr_mask = (window >= window_q25) & (window <= window_q75)
            window_iqr_mean = np.mean(window[window_iqr_mask])

            adjustment = iqr_mean - window_iqr_mean
            self.data[start_idx:end_idx] += adjustment




        