import numpy as np

class NoisyData:

    def __init__(self, raw_dataset):
        self.data = raw_dataset.data.copy()
        self.indices = raw_dataset.indices.copy()
        self.classes = raw_dataset.classes.copy()

    def noisify(self, noise_ratio):
        signal_peak = np.max(np.abs(self.data))
        noise_sigma = signal_peak * noise_ratio
        noise = np.random.normal(0, noise_sigma, self.data.shape)
        
        self.data += noise