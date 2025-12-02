
import scipy.io as spio
from scipy.signal import butter, sosfiltfilt, find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

from cwc.src.classifier import Classifier

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
        self.aligned_spikes = np.array([])
        self.aligned_indices = np.array([])

        self.features = np.array([])
        self.correlated_classes = np.array([])
        
        self.pca = PCA(n_components=8, whiten=True, random_state=42)


    def apply_band_pass_filter(self, filter_low, filter_high, sample_rate, order):
        
        nyquist = 0.5 * sample_rate
        low = filter_low / nyquist
        high = filter_high / nyquist

        sos = butter(order, [low, high], btype='band', output='sos')
        self.filtered_data = sosfiltfilt(sos, self.data)


    def extract_spikes(self, window_size):

        spikes = []

        for index in self.indices:

            if index + window_size <= len(self.data):

                spike = self.filtered_data[index:index + window_size]
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


    def align_spikes(self, target_peak_pos, window_size):

        aligned_spikes = []
        aligned_indices = []

        for peak in self.detected_spikes:

            start = peak - target_peak_pos
            end = start + window_size

            if start >= 0 and end <= len(self.filtered_data):
                spike = self.filtered_data[start:end]
                
                peak_within_spike = np.argmax(np.abs(spike))
                shift = target_peak_pos - peak_within_spike

                aligned = np.zeros(window_size)

                if shift > 0:
                    aligned[shift:] = spike[:-shift]
                elif shift < 0:
                    aligned[:shift] = spike[-shift:]
                else:
                    aligned = spike

                aligned_spikes.append(aligned)
                aligned_indices.append(peak)

        self.aligned_spikes = np.array(aligned_spikes)
        self.aligned_indices = np.array(aligned_indices)


    def extract_features(self):

        scaler = StandardScaler()
        scaled_spikes = scaler.fit_transform(self.aligned_spikes)

        self.features = self.pca.fit_transform(scaled_spikes)

    def correlate_classes(self, distance_threshold):

        correlated_classes = []

        # We need to filter features and aligned_indices simultaneously
        # so we create a boolean mask
        mask = np.zeros(len(self.aligned_indices), dtype=bool)

        for i, index in enumerate(self.aligned_indices):
            # Find the closest ground truth index
            closest_idx_loc = np.argmin(np.abs(self.indices - index))
            closest_true_index = self.indices[closest_idx_loc]
            
            # Check the distance
            dist = abs(closest_true_index - index)

            # ONLY keep this for training if it is close to a real spike
            if dist <= distance_threshold:
                correlated_classes.append(self.classes[closest_idx_loc])
                mask[i] = True
            # Else: It is noise (False Positive), ignore it for training

        # Apply the filtering
        self.correlated_classes = np.array(correlated_classes)
        
        # Update internal state to remove noise from training set
        self.aligned_indices = self.aligned_indices[mask]
        self.features = self.features[mask]

    def create_classifier(self):

        classifer = Classifier(self.features, self.correlated_classes)
        classifer.train()

        return classifer
    
    def classify_detected_spikes(self, classifier):

        predictions = classifier.classifier.predict(self.features)
        
        return self.aligned_indices, predictions
        

        


        


