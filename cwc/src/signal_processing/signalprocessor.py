################################################################
##
## EE40098 Coursework C
##
## File         :  signalprocessor.py
## Author       :  samh25
## Created      :  2025-12-02 (YYYY-MM-DD)
## License      :  MIT
## Description  :  Class for aligning spikes and extracting features.
##
################################################################

################################################################
## MARK: INCLUDES
################################################################

import scipy.io as spio
from scipy.signal import butter, sosfiltfilt, find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

from .classifier import Classifier

################################################################
## MARK: CLASS DEFINITIONS
################################################################

class SignalProcessor:

    ############################################################
    ## CONSTRUCTOR

    # instantiate a new signal processor
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
        self.scaler = None

        self.pca = PCA(n_components=32, whiten=True, random_state=42)
    
    ############################################################
    ## INSTANCE METHODS

    # align detected spikes to a target peak position
    def align_spikes(self, target_peak_pos, window_size):

        aligned_spikes = []
        aligned_indices = []

        # for each detected spike, align to target peak position
        for peak in self.detected_spikes:

            start = peak - target_peak_pos
            end = start + window_size

            # only align if within bounds
            if start >= 0 and end <= len(self.filtered_data):
                spike = self.filtered_data[start:end]

                # identify actual peak within the spike, and target alignment
                peak_within_spike = np.argmax(np.abs(spike))
                shift = target_peak_pos - peak_within_spike

                aligned = np.zeros(window_size)

                # perform the shift
                if shift > 0:
                    aligned[shift:] = spike[:-shift]
                elif shift < 0:
                    aligned[:shift] = spike[-shift:]
                else:
                    aligned = spike

                # add to list
                aligned_spikes.append(aligned)
                aligned_indices.append(peak)

        # store aligned spikes and their indices
        self.aligned_spikes = np.array(aligned_spikes)
        self.aligned_indices = np.array(aligned_indices)
        

    # extract features from aligned spikes
    def extract_features(self):

        features_list = []
        
        # for each aligned spike, extract features
        for spike in self.aligned_spikes:

            # Waveform shape features
            peak_amplitude = np.min(spike)
            peak_idx = np.argmin(spike)
            
            # Temporal features
            half_width = self.calculate_half_width(spike)
            
            # Derivative features
            first_deriv = np.diff(spike)
            max_rise = np.max(first_deriv)
            max_fall = np.min(first_deriv)
            
            # Combine features
            feat = np.concatenate([
                spike,  # Raw waveform
                [peak_amplitude, peak_idx, half_width, max_rise, max_fall]
            ])
            features_list.append(feat)
        
        # convert to numpy array
        features = np.array(features_list)
        
        # if scaler and pca not fitted yet, fit them
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled = self.scaler.fit_transform(features)
            self.features = self.pca.fit_transform(scaled)
        else:
            scaled = self.scaler.transform(features)
            self.features = self.pca.transform(scaled)


    # calculate half-width of a spike
    def calculate_half_width(self, spike):

        min_val = np.min(spike)
        half_max = min_val / 2
        below_half = spike < half_max
        if np.any(below_half):
            return np.sum(below_half)
        return 0
    

    # correlate detected spikes with ground truth classes
    def correlate_classes(self, distance_threshold):

        correlated_classes = []

        # create a boolean mask
        mask = np.zeros(len(self.aligned_indices), dtype=bool)

        for i, index in enumerate(self.aligned_indices):
            # Find the closest ground truth index
            closest_idx_loc = np.argmin(np.abs(self.indices - index))
            closest_true_index = self.indices[closest_idx_loc]
            
            # Check the distance
            dist = abs(closest_true_index - index)

            # only keep this for training if it is close to a real spike
            if dist <= distance_threshold:
                correlated_classes.append(self.classes[closest_idx_loc])
                mask[i] = True

        # Apply the filtering
        self.correlated_classes = np.array(correlated_classes)
        
        # Update internal state to remove noise from training set
        self.aligned_indices = self.aligned_indices[mask]
        self.features = self.features[mask]


    # create, train and return a classifier
    def create_classifier(self):

        classifer = Classifier(self.features, self.correlated_classes)
        classifer.train()

        return classifer
    