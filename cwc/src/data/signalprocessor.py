
import scipy.io as spio
from scipy.signal import butter, filtfilt, find_peaks
import numpy as np

class SignalProcessor:

    def __init__(self, dataset):
        self.data = dataset.data
        self.indices = dataset.indices
        self.classes = dataset.classes
        self.filtered_data = np.array([])
        self.spikes = np.array([])

        self.mean_offset = 0
        self.std_offset = 0
        self.min_offset = 0
        self.max_offset = 0

        self.mad = 0
        self.detected_spikes = np.array([])
        

    def apply_band_pass_filter(self, filter_low, filter_high, sample_rate, order):
        
        nyquist = 0.5 * sample_rate
        low = filter_low / nyquist
        high = filter_high / nyquist

        b, a = butter(order, [low, high], btype='band')
        self.filtered_data = filtfilt(b, a, self.data)


    def extract_spikes(self, window_size):

        spikes = []

        for index in self.indices:

            if index + window_size <= len(self.data):

                spike = self.data[index:index + window_size]
                spikes.append(spike)

        self.spikes = np.array(spikes)


    def compute_offsets(self):

        peak_offsets = []

        for spike in self.spikes:
            peak_offsets.append(np.argmax(np.abs(spike)))
            
        self.mean_offset = np.mean(peak_offsets)
        self.std_offset = np.std(peak_offsets)
        self.min_offset = np.min(peak_offsets)
        self.max_offset = np.max(peak_offsets)


    def calculate_mad(self):
        
        self.mad = np.median(np.abs(self.filtered_data)) / 0.6745


    def detect_spikes(self, mad_gain, distance):

        threshold = mad_gain * self.mad

        peaks, _ = find_peaks(np.abs(self.filtered_data), height=threshold, distance=distance)
        peaks = peaks - round(self.mean_offset)

        self.detected_spikes = peaks
        

        


        


