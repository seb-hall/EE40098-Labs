################################################################
##
## EE40098 Coursework C
##
## File         :  main.py
## Author       :  samh25
## Created      :  2025-12-11 (YYYY-MM-DD)
## License      :  MIT
## Description  :  Main program for Coursework C
##
################################################################

################################################################
## MARK: INCLUDES
################################################################

import numpy as np
import matplotlib.pyplot as plt

from data import Dataset
from data import DataNormaliser
from data import NoisyData
from signal_processing import BandpassFilter
from signal_processing import SignalProcessor
from signal_processing import SpikeDetector

################################################################
## MARK: FUNCTIONS
################################################################

# evaluate spike detection performance
def evaluate_detection(detected_spikes, true_indices, tolerance=50):

    detected_spikes = np.array(detected_spikes)
    true_indices = np.array(true_indices)
    
    true_matched = np.zeros(len(true_indices), dtype=bool)
    detected_matched = np.zeros(len(detected_spikes), dtype=bool)
    
    # For each detected spike, find closest true spike
    for i, det_spike in enumerate(detected_spikes):
        distances = np.abs(true_indices - det_spike)
        closest_idx = np.argmin(distances)
        
        # If within tolerance and not already matched
        if distances[closest_idx] <= tolerance and not true_matched[closest_idx]:
            true_matched[closest_idx] = True
            detected_matched[i] = True
    
    true_positives = np.sum(true_matched)
    false_positives = np.sum(~detected_matched)
    false_negatives = np.sum(~true_matched)
    
    precision = true_positives / len(detected_spikes) if len(detected_spikes) > 0 else 0
    recall = true_positives / len(true_indices) if len(true_indices) > 0 else 0
    
    return {
        'TP': true_positives,
        'FP': false_positives,
        'FN': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    }


# main program entry point
def main():

    print("Running Coursework C...")

    #### LOAD BASE DATASET ####

    data_folder = "datasets"
    output_folder = "datasets/output"

    dataset = Dataset()
    dataset.load_from_mat(f"{data_folder}/D1.mat")
    
    print("Loaded Dataset:")
    print("\tData shape:", dataset.data.shape)
    print("\tIndices shape:", dataset.indices.shape)
    print("\tClasses shape:", dataset.classes.shape)
    print("\tUnique classes:", np.unique(dataset.classes))
    print("\tClass instances:", np.bincount(dataset.classes))
    print("\tMin distance between spikes:", np.min(np.diff(np.sort(dataset.indices))))

    #### SPLIT INTO TRAINING AND TEST SETS ####

    train_data = dataset
    test_data = train_data.split_into_test(test_ratio=0.2)

    print("\nTraining Set:")
    print("\tData shape:", train_data.data.shape)
    print("\tIndices shape:", train_data.indices.shape)
    print("\tClasses shape:", train_data.classes.shape)
    print("\tClass instances:", np.bincount(train_data.classes))
    print("\nTest Set:")
    print("\tData shape:", test_data.data.shape)
    print("\tIndices shape:", test_data.indices.shape)
    print("\tClasses shape:", test_data.classes.shape)
    print("\tClass instances:", np.bincount(test_data.classes), "\n")

    ### CREATE TRAINING DATASET AND TRAIN CLASSIFIER ###

    # initial setup and spike detection
    train_filter = BandpassFilter(train_data)
    train_spikes = SpikeDetector(train_filter)
    train_mad = train_spikes.calculate_mad()
    train_spikes.detect_spikes(mad=train_mad, mad_gain=4.0, distance=25)

    # extract features and correlate with classes
    processor = SignalProcessor(train_data)
    processor.filtered_data = train_filter.filtered_data
    processor.detected_spikes = train_spikes.detected_spikes
    processor.align_spikes(target_peak_pos=20, window_size=64)
    processor.extract_features()
    processor.correlate_classes(distance_threshold=50)

    augmented_features = [processor.features]
    augmented_labels = [processor.correlated_classes]

    # iterate over three different noise levels
    for noise_level in [0.15, 0.30, 0.45]:

        # create noisy version of training data
        noisy_version = NoisyData(train_data)
        noisy_version.noisify(noise_level)
        
        # detect spikes
        noisy_filter = BandpassFilter(noisy_version)
        noisy_spikes = SpikeDetector(noisy_filter)
        mad = noisy_spikes.calculate_mad()
        noisy_spikes.detect_spikes(mad=mad, mad_gain=3.5, distance=25)
        
        # extract features and correlate with classes
        noisy_processor = SignalProcessor(noisy_version)
        noisy_processor.filtered_data = noisy_filter.filtered_data
        noisy_processor.detected_spikes = noisy_spikes.detected_spikes
        noisy_processor.align_spikes(target_peak_pos=20, window_size=64)
        noisy_processor.scaler = processor.scaler  # Share scaler
        noisy_processor.pca = processor.pca  # Share PCA
        noisy_processor.extract_features()
        noisy_processor.correlate_classes(distance_threshold=50)

        # append to augmented dataset
        augmented_features.append(noisy_processor.features)
        augmented_labels.append(noisy_processor.correlated_classes)

    # Combine all training data
    processor.features = np.vstack(augmented_features)
    processor.correlated_classes = np.concatenate(augmented_labels)

    # Train on augmented dataset
    classifier = processor.create_classifier()
    print(f"Classifier trained with {len(processor.correlated_classes)} augmented samples\n")

    ### APPLY ON OTHER DATASETS ###

    unlabelled_datasets = ["D2.mat", "D3.mat", "D4.mat", "D5.mat", "D6.mat"]
    mad_gains = [3.2, 3.0, 3.0, 2.9, 2.9]
    min_distance = 60

    i = 0
    # for each dataset
    for dataset in unlabelled_datasets:

        dataset_path = f"{data_folder}/{dataset}"

        print("Processing dataset:", dataset)

        mad_gain = mad_gains[i]
        i += 1

        # load unlabelled dataset
        unlabelled_data = Dataset()
        unlabelled_data.load_from_mat_unlabelled(dataset_path)

        # apply sliding window normalisation
        normalised_data = DataNormaliser(unlabelled_data)
        normalised_data.normalise(window_size=1000)

        # run spike detection
        unlabelled_filter = BandpassFilter(normalised_data)
        unlabelled_spikes = SpikeDetector(unlabelled_filter)
        unlabelled_mad = unlabelled_spikes.calculate_mad()
        unlabelled_spikes.detect_spikes(mad=unlabelled_mad, mad_gain=mad_gain, distance=min_distance)
        
        # align spikes and extract features
        unlabelled_processor = SignalProcessor(normalised_data)
        unlabelled_processor.filtered_data = unlabelled_filter.filtered_data
        unlabelled_processor.detected_spikes = unlabelled_spikes.detected_spikes
        unlabelled_processor.align_spikes(target_peak_pos=20, window_size=64)

        unlabelled_processor.scaler = processor.scaler  # Use the same scaler as training
        unlabelled_processor.pca = processor.pca  # Use the same PCA as training
        unlabelled_processor.extract_features()

        # run classification
        unlabelled_predictions = classifier.classifier.predict(unlabelled_processor.features)

        # save predictions to file
        unlabelled_data.write_to_mat(f"{output_folder}/{dataset}", unlabelled_processor.aligned_indices, unlabelled_predictions)

        print(f"Predictions saved to {dataset_path.replace('cwc/data/;', 'cwc/data/output/')}")
        print(f"Predictions for {dataset_path}: {np.bincount(unlabelled_predictions)}")
        print(f"Total spikes detected: {len(unlabelled_processor.aligned_indices)}\n")
    
    print("Coursework C complete.")


# assign main function to entry point
if __name__ == '__main__':
    main()