# %%
def in_jupyter():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        print(f"Running in shell: {shell}")
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False

if in_jupyter():
    from IPython import get_ipython
    ip = get_ipython()
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "3")


import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")

def evaluate_detection_proper(detected_spikes, true_indices, tolerance=50):
    """
    Proper evaluation - each true spike can only be matched once
    """
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

# %%
# LOAD BASE DATASET

from cwc.src.data import Dataset
from cwc.src.data import DataNormaliser
from cwc.src.signal import BandpassFilter
from cwc.src.signal import NoisyData
import numpy as np
import matplotlib.pyplot as plt


dataset = Dataset()
dataset.load_from_mat("cwc/data/D1.mat")

print("Loaded Dataset:")
print("\tData shape:", dataset.data.shape)
print("\tIndices shape:", dataset.indices.shape)
print("\tClasses shape:", dataset.classes.shape)
print("\tUnique classes:", np.unique(dataset.classes))
print("\tClass instances:", np.bincount(dataset.classes))
print("\tMin distance between spikes:", np.min(np.diff(np.sort(dataset.indices))))

# SPLIT DATASET INTO TRAINING AND TEST SETS

train_data = dataset
test_data = train_data.split_into_test(test_ratio=0.2)

# print basic info about the training and test sets
print("\nTraining Set:")
print("\tData shape:", train_data.data.shape)
print("\tIndices shape:", train_data.indices.shape)
print("\tClasses shape:", train_data.classes.shape)
print("\tClass instances:", np.bincount(train_data.classes))
print("\nTest Set:")
print("\tData shape:", test_data.data.shape)
print("\tIndices shape:", test_data.indices.shape)
print("\tClasses shape:", test_data.classes.shape)
print("\tClass instances:", np.bincount(test_data.classes))


# %%
# compare wavelet denoisuing vs bandpass filtering vs no filtering

base_data = train_data.data.copy()

noisy_data = NoisyData(train_data)
noisy_data.noisify(0.2)

wavelet_filter = BandpassFilter(noisy_data)
wavelet_filter.appply_wavelet_denoise(wavelet='db4', level=4, alpha=2.0)

wavelet_filter_2 = BandpassFilter(noisy_data)
wavelet_filter_2.appply_wavelet_denoise(wavelet='db4', level=4, alpha=1.0)

wavelet_filter_3 = BandpassFilter(noisy_data)
wavelet_filter_3.appply_wavelet_denoise(wavelet='db4', level=4, alpha=0.1)


if in_jupyter():
   
    plt.plot(base_data, label='Original Signal', alpha=0.5)
    plt.plot(noisy_data.data, label='Noisy Signal', alpha=0.5)
    plt.plot(wavelet_filter.filtered_data, label='Wavelet Denoised (Alpha 2.0)', alpha=0.8)
    plt.plot(wavelet_filter_2.filtered_data, label='Wavelet Denoised (Alpha 1.0)', alpha=0.8)
    plt.plot(wavelet_filter_3.filtered_data, label='Wavelet Denoised (Alpha 0.5)', alpha=0.8)

    plt.legend()
    plt.title("Signal Filtering Comparison")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()

# %%
# identify the spacing between spikes in the training set
# sort the indices and calculate the differences between consecutive indices

indices = train_data.indices.copy()
indices.sort()

distances = {}
for i in range(1, len(indices)):
    distance = indices[i] - indices[i - 1]
    
    if distance not in distances:
        distances[distance] = 0
    
    distances[distance] += 1

# sort the distances dictionary by key
distances = dict(sorted(distances.items()))

# How many spikes are within 60 samples of each other?
close_pairs = 0
for i in range(len(indices)-1):
    if indices[i+1] - indices[i] < 60:
        close_pairs += 1

print(f"Spike pairs closer than 60 samples: {close_pairs}")
print(f"Percentage of spikes affected: {100*close_pairs/len(indices):.1f}%")

for distance, count in distances.items():
    print(f"Distance: {distance} samples, Count: {count}")

    

# now plot a histogram of the distances

if in_jupyter():
    plt.figure(figsize=(10, 5))
    plt.bar(distances.keys(), distances.values())
    plt.title("Histogram of Distances Between Spikes in Training Set")
    plt.xlabel("Distance (samples)")
    plt.ylabel("Count")
    plt.show()


# %%

if in_jupyter():
    plt.plot(train_data.data)
    plt.show()

if in_jupyter():
    plt.plot(test_data.data)
    plt.show()

# %%
# MAKE NOISIER DATASETS

noise_levels = [0.1, 0.3, 0.5]

noisy_datasets = []

for noise_level in noise_levels:
    noisy_data_maker = NoisyData(train_data)
    noisy_data_maker.noisify(noise_level)
    noisy_datasets.append(noisy_data_maker)
    print(f"\nNoisy Dataset (noise level={noise_level}):")
    print("\tData shape:", noisy_data_maker.data.shape)
    print("\tIndices shape:", noisy_data_maker.indices.shape)
    print("\tClasses shape:", noisy_data_maker.classes.shape)
    print("\tClass instances:", np.bincount(noisy_data_maker.classes))

    if in_jupyter():
        plt.plot(noisy_data_maker.data)
        plt.show()


# %%
# add normaliser class
un_normalised_data = DataNormaliser(train_data)
un_normalised_data.un_normalise(gain=1.0, cycles=5)

if in_jupyter():
    plt.plot(un_normalised_data.data)
    plt.title("Unnormalised Data (gain=5.0, wavelength=5e5)")
    plt.show()

normalised_data = DataNormaliser(un_normalised_data)
normalised_data.normalise(window_size=1000)

if in_jupyter() or True:
    plt.plot(normalised_data.data, label='Normalised Data', alpha=0.5)
    plt.plot(train_data.data, alpha=0.5, label='Original Data')
    plt.title("Normalised Data (window_size=1000)")
    plt.legend()
    plt.show()


# %%
from cwc.src.signal import BandpassFilter
import matplotlib.pyplot as plt

train_filter = BandpassFilter(train_data)
#train_filter.appply_wavelet_denoise(alpha=0.5)

test_filter = BandpassFilter(test_data)
#test_filter.appply_wavelet_denoise(alpha=0.5)

if in_jupyter():
    plt.plot(train_filter.filtered_data)
    plt.show()

if in_jupyter():
    plt.plot(test_filter.filtered_data)
    plt.show()


# %%
# TEST PURE MAD SPIKE DETECTION

from cwc.src.signal import SpikeDetector

mad_gain = 4.0
min_distance = 25

train_spikes = SpikeDetector(train_filter)
train_mad = train_spikes.calculate_mad()
train_spikes.detect_spikes(mad=train_mad, mad_gain=mad_gain, distance=min_distance)

test_spikes = SpikeDetector(test_filter)
test_mad = test_spikes.calculate_mad()
test_spikes.detect_spikes(mad=test_mad, mad_gain=mad_gain, distance=min_distance)

## evaluate spike detection performance on training set
correct_detections = 0
false_positives = 0
false_negatives = 0

for detected_spike in train_spikes.detected_spikes:
    if any(np.abs(train_spikes.indices - detected_spike) <= min_distance):
        correct_detections += 1
    else:
        false_positives += 1

incorrect_detections = false_positives + (len(train_spikes.indices) - correct_detections)

print("Training Set Spike Detection:")
print("\tCorrect Detections:", correct_detections, "out of ", len(train_spikes.indices))
print("\tTotal Detected Spikes:", len(train_spikes.detected_spikes))
print("\tFalse Positives:", false_positives)
print("\tFalse Negatives:", len(train_spikes.indices) - correct_detections)
print("\tIncorrect Detections:", incorrect_detections)
print("\tRecall: {:.2f}%".format(100 * correct_detections / len(train_spikes.indices)))
print("\tPrecision: {:.2f}%".format(100 * correct_detections / (correct_detections + false_positives)))

# plot detected spikes on training data, alonside the true spike indices

if in_jupyter():
    plt.plot(train_filter.filtered_data)
    plt.scatter(train_spikes.detected_spikes, train_filter.filtered_data[train_spikes.detected_spikes], color='red')
    plt.scatter(train_spikes.indices, train_filter.filtered_data[train_spikes.indices], color='green', marker='x')
    plt.show()

if in_jupyter():
    plt.plot(test_filter.filtered_data)
    plt.scatter(test_spikes.detected_spikes, test_filter.filtered_data[test_spikes.detected_spikes], color='red')
    plt.scatter(test_spikes.indices, test_filter.filtered_data[test_spikes.indices], color='green', marker='x')
    plt.show()


# %%
# After spike detection
from cwc.src.data import SignalProcessor

processor = SignalProcessor(train_data)
processor.filtered_data = train_filter.filtered_data

# Align and extract spikes
processor.detected_spikes = train_spikes.detected_spikes
processor.align_spikes(target_peak_pos=20, window_size=64)

# Extract PCA features
processor.extract_features()

# Match to ground truth for training
processor.correlate_classes(distance_threshold=50)

# Train classifier
classifier = processor.create_classifier()

print(f"Classifier trained with {len(processor.correlated_classes)} samples")
print(f"Training accuracy: {classifier.accuracy:.2%}")

# %%
# Original training data
train_filter = BandpassFilter(train_data)
train_spikes = SpikeDetector(train_filter)
train_mad = train_spikes.calculate_mad()
train_spikes.detect_spikes(mad=train_mad, mad_gain=4.0, distance=25)

processor = SignalProcessor(train_data)
processor.filtered_data = train_filter.filtered_data
processor.detected_spikes = train_spikes.detected_spikes
processor.align_spikes(target_peak_pos=20, window_size=64)
processor.extract_features()
processor.correlate_classes(distance_threshold=50)

# ADD AUGMENTED TRAINING DATA
augmented_features = [processor.features]
augmented_labels = [processor.correlated_classes]

# Create noisy versions and add to training
for noise_level in [0.15, 0.30, 0.45]:
    noisy_version = NoisyData(train_data)
    noisy_version.noisify(noise_level)
    
    noisy_filter = BandpassFilter(noisy_version)
    noisy_spikes = SpikeDetector(noisy_filter)
    mad = noisy_spikes.calculate_mad()
    noisy_spikes.detect_spikes(mad=mad, mad_gain=3.5, distance=25)
    
    noisy_processor = SignalProcessor(noisy_version)
    noisy_processor.filtered_data = noisy_filter.filtered_data
    noisy_processor.detected_spikes = noisy_spikes.detected_spikes
    noisy_processor.align_spikes(target_peak_pos=20, window_size=64)
    noisy_processor.scaler = processor.scaler  # Share scaler
    noisy_processor.pca = processor.pca  # Share PCA
    noisy_processor.extract_features()
    noisy_processor.correlate_classes(distance_threshold=50)
    
    augmented_features.append(noisy_processor.features)
    augmented_labels.append(noisy_processor.correlated_classes)

# Combine all training data
processor.features = np.vstack(augmented_features)
processor.correlated_classes = np.concatenate(augmented_labels)

# Train on augmented dataset
classifier = processor.create_classifier()
print(f"Classifier trained with {len(processor.correlated_classes)} augmented samples")

# %%
# %%
# COMPARISON: Old Training vs Augmented Training on Noisy Data
print("=" * 80)
print("TRAINING METHOD COMPARISON - Old vs Augmented")
print("=" * 80)

from cwc.src.data import SignalProcessor
from cwc.src.signal import BandpassFilter, SpikeDetector, NoisyData
import numpy as np

# Test noise levels
test_noise_levels = [0.1, 0.3, 0.5]

# Create test datasets at different noise levels
print("\nCreating noisy test datasets...")
noisy_test_datasets = []
for noise_level in test_noise_levels:
    noisy_test = NoisyData(test_data)
    noisy_test.noisify(noise_level)
    noisy_test_datasets.append(noisy_test)
    print(f"  Created test set with noise={noise_level}")

# ============================================================================
# METHOD 1: OLD TRAINING (Clean data only)
# ============================================================================
print("\n" + "=" * 80)
print("METHOD 1: Training on CLEAN data only")
print("=" * 80)

train_filter_old = BandpassFilter(train_data)
train_spikes_old = SpikeDetector(train_filter_old)
train_mad_old = train_spikes_old.calculate_mad()
train_spikes_old.detect_spikes(mad=train_mad_old, mad_gain=4.0, distance=25)

processor_old = SignalProcessor(train_data)
processor_old.filtered_data = train_filter_old.filtered_data
processor_old.detected_spikes = train_spikes_old.detected_spikes
processor_old.align_spikes(target_peak_pos=20, window_size=64)
processor_old.extract_features()
processor_old.correlate_classes(distance_threshold=50)

classifier_old = processor_old.create_classifier()
print(f"Old method: Trained with {len(processor_old.correlated_classes)} samples")
print(f"Old method: Training accuracy = {classifier_old.accuracy:.2%}")

# Test old method on noisy data
print("\nOld Method Performance on Noisy Test Data:")
print("-" * 80)
old_results = []

for i, noisy_test in enumerate(noisy_test_datasets):
    noise_level = test_noise_levels[i]
    
    # Process noisy test data
    test_filter_noisy = BandpassFilter(noisy_test)
    test_spikes_noisy = SpikeDetector(test_filter_noisy)
    test_mad_noisy = test_spikes_noisy.calculate_mad()
    test_spikes_noisy.detect_spikes(mad=test_mad_noisy, mad_gain=3.5, distance=60)
    
    test_processor_noisy = SignalProcessor(noisy_test)
    test_processor_noisy.filtered_data = test_filter_noisy.filtered_data
    test_processor_noisy.detected_spikes = test_spikes_noisy.detected_spikes
    test_processor_noisy.align_spikes(target_peak_pos=20, window_size=64)
    test_processor_noisy.scaler = processor_old.scaler
    test_processor_noisy.pca = processor_old.pca
    test_processor_noisy.extract_features()
    
    # Classify
    predictions_old = classifier_old.classifier.predict(test_processor_noisy.features)
    
    # Evaluate
    correct = 0
    total = 0
    for j, idx in enumerate(test_processor_noisy.aligned_indices):
        distances = np.abs(noisy_test.indices - idx)
        closest = np.argmin(distances)
        if distances[closest] <= 50:
            total += 1
            if predictions_old[j] == noisy_test.classes[closest]:
                correct += 1
    
    accuracy = 100 * correct / total if total > 0 else 0
    old_results.append(accuracy)
    print(f"  Noise={noise_level:.1f}: Accuracy={accuracy:.1f}% ({correct}/{total} correct)")

# ============================================================================
# METHOD 2: AUGMENTED TRAINING (Clean + Noisy data)
# ============================================================================
print("\n" + "=" * 80)
print("METHOD 2: Training on CLEAN + AUGMENTED NOISY data")
print("=" * 80)

train_filter_new = BandpassFilter(train_data)
train_spikes_new = SpikeDetector(train_filter_new)
train_mad_new = train_spikes_new.calculate_mad()
train_spikes_new.detect_spikes(mad=train_mad_new, mad_gain=4.0, distance=25)

processor_new = SignalProcessor(train_data)
processor_new.filtered_data = train_filter_new.filtered_data
processor_new.detected_spikes = train_spikes_new.detected_spikes
processor_new.align_spikes(target_peak_pos=20, window_size=64)
processor_new.extract_features()
processor_new.correlate_classes(distance_threshold=50)

# Collect augmented training data
augmented_features = [processor_new.features]
augmented_labels = [processor_new.correlated_classes]

print("Adding augmented noisy training data...")
for noise_level in [0.15, 0.30, 0.45]:
    noisy_version = NoisyData(train_data)
    noisy_version.noisify(noise_level)
    
    noisy_filter = BandpassFilter(noisy_version)
    noisy_spikes = SpikeDetector(noisy_filter)
    mad = noisy_spikes.calculate_mad()
    noisy_spikes.detect_spikes(mad=mad, mad_gain=3.5, distance=25)
    
    noisy_processor = SignalProcessor(noisy_version)
    noisy_processor.filtered_data = noisy_filter.filtered_data
    noisy_processor.detected_spikes = noisy_spikes.detected_spikes
    noisy_processor.align_spikes(target_peak_pos=20, window_size=64)
    noisy_processor.scaler = processor_new.scaler
    noisy_processor.pca = processor_new.pca
    noisy_processor.extract_features()
    noisy_processor.correlate_classes(distance_threshold=50)
    
    augmented_features.append(noisy_processor.features)
    augmented_labels.append(noisy_processor.correlated_classes)
    print(f"  Added {len(noisy_processor.correlated_classes)} samples at noise={noise_level}")

# Combine all training data
processor_new.features = np.vstack(augmented_features)
processor_new.correlated_classes = np.concatenate(augmented_labels)

classifier_new = processor_new.create_classifier()
print(f"\nNew method: Trained with {len(processor_new.correlated_classes)} samples")
print(f"New method: Training accuracy = {classifier_new.accuracy:.2%}")

# Test new method on noisy data
print("\nNew Method Performance on Noisy Test Data:")
print("-" * 80)
new_results = []

for i, noisy_test in enumerate(noisy_test_datasets):
    noise_level = test_noise_levels[i]
    
    # Process noisy test data (same detection as before)
    test_filter_noisy = BandpassFilter(noisy_test)
    test_spikes_noisy = SpikeDetector(test_filter_noisy)
    test_mad_noisy = test_spikes_noisy.calculate_mad()
    test_spikes_noisy.detect_spikes(mad=test_mad_noisy, mad_gain=3.5, distance=60)
    
    test_processor_noisy = SignalProcessor(noisy_test)
    test_processor_noisy.filtered_data = test_filter_noisy.filtered_data
    test_processor_noisy.detected_spikes = test_spikes_noisy.detected_spikes
    test_processor_noisy.align_spikes(target_peak_pos=20, window_size=64)
    test_processor_noisy.scaler = processor_new.scaler
    test_processor_noisy.pca = processor_new.pca
    test_processor_noisy.extract_features()
    
    # Classify
    predictions_new = classifier_new.classifier.predict(test_processor_noisy.features)
    
    # Evaluate
    correct = 0
    total = 0
    for j, idx in enumerate(test_processor_noisy.aligned_indices):
        distances = np.abs(noisy_test.indices - idx)
        closest = np.argmin(distances)
        if distances[closest] <= 50:
            total += 1
            if predictions_new[j] == noisy_test.classes[closest]:
                correct += 1
    
    accuracy = 100 * correct / total if total > 0 else 0
    new_results.append(accuracy)
    print(f"  Noise={noise_level:.1f}: Accuracy={accuracy:.1f}% ({correct}/{total} correct)")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print(f"{'Noise Level':<15} {'Old Method':<15} {'New Method':<15} {'Improvement':<15}")
print("-" * 80)

total_old = 0
total_new = 0

for i, noise_level in enumerate(test_noise_levels):
    improvement = new_results[i] - old_results[i]
    improvement_str = f"+{improvement:.1f}%" if improvement >= 0 else f"{improvement:.1f}%"
    print(f"{noise_level:<15.1f} {old_results[i]:<14.1f}% {new_results[i]:<14.1f}% {improvement_str:<15}")
    total_old += old_results[i]
    total_new += new_results[i]

avg_old = total_old / len(test_noise_levels)
avg_new = total_new / len(test_noise_levels)
avg_improvement = avg_new - avg_old

print("-" * 80)
print(f"{'AVERAGE':<15} {avg_old:<14.1f}% {avg_new:<14.1f}% {'+' if avg_improvement >= 0 else ''}{avg_improvement:.1f}%")
print("=" * 80)

if avg_improvement > 0:
    print(f"\n✓ AUGMENTED TRAINING WINS by {avg_improvement:.1f}% on average!")
    print("  Recommendation: Use augmented training for final submission")
else:
    print(f"\n✗ Old training performs better by {abs(avg_improvement):.1f}%")
    print("  Recommendation: Stick with clean training only")

# %%
# Create new processor for test data
test_processor = SignalProcessor(test_data)
test_processor.filtered_data = test_filter.filtered_data
test_processor.detected_spikes = test_spikes.detected_spikes
test_processor.align_spikes(target_peak_pos=20, window_size=64)
test_processor.scaler = processor.scaler  # Use the same scaler as training
test_processor.pca = processor.pca  # Use the same PCA as training
test_processor.extract_features()

# Classify
predictions = classifier.classifier.predict(test_processor.features)

# Evaluate test set performance
correct_predictions = 0

for i, detected_index in enumerate(test_processor.aligned_indices):
    predicted_class = predictions[i]
    
    # Find closest ground truth spike
    distances = np.abs(test_data.indices - detected_index)
    closest_idx = np.argmin(distances)
    
    # Only count if detection is close enough to a real spike
    if distances[closest_idx] <= 50:
        true_class = test_data.classes[closest_idx]
        if predicted_class == true_class:
            correct_predictions += 1

total_predictions = len(predictions)
print("Test Set Classification:")
print("\tCorrect Predictions:", correct_predictions, "out of ", total_predictions)
print("\tAccuracy: {:.2f}%".format(100 * correct_predictions / total_predictions if total_predictions > 0 else 0))

# %%
# test on noisy data

min_distance = 60

mad_gains = [3.6, 3.0, 2.6]

print("====STARTING NOISY DATA TESTS====")
for i, noisy_dataset in enumerate(noisy_datasets):
    
    unlabelled_filter = BandpassFilter(noisy_dataset)
    #unlabelled_filter.appply_wavelet_denoise()
    
    unlabelled_spikes = SpikeDetector(unlabelled_filter)
    unlabelled_mad = unlabelled_spikes.calculate_mad()
    unlabelled_spikes.detect_spikes(mad=unlabelled_mad, mad_gain=mad_gains[i], distance=min_distance)

    if in_jupyter():
        plt.plot(unlabelled_filter.filtered_data)
        plt.scatter(unlabelled_spikes.detected_spikes, unlabelled_filter.filtered_data[unlabelled_spikes.detected_spikes], color='red')
        plt.show()
    
    metrics = evaluate_detection_proper(unlabelled_spikes.detected_spikes, 
                                       unlabelled_spikes.indices, 
                                       tolerance=50)
    
    print(f"Single Pass Spike Detection ({noise_levels[i]}):")    
    print(f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")


# %%
# SYSTEMATIC THRESHOLD SWEEP
print("====THRESHOLD SWEEP====")

test_gains = np.arange(1.6, 5.0, 0.1)

for noise_idx, noisy_dataset in enumerate(noisy_datasets):
    print(f"\n=== Noise Level {noise_levels[noise_idx]} ===")
    
    unlabelled_filter = BandpassFilter(noisy_dataset)
    #unlabelled_filter.appply_wavelet_denoise(alpha=0.0)
    
    best_f1 = 0
    best_gain = 0
    results = []
    
    for gain in test_gains:
        detector = SpikeDetector(unlabelled_filter)
        mad = detector.calculate_mad()
        detector.detect_spikes(mad=mad, mad_gain=gain, distance=60)
        
        metrics = evaluate_detection_proper(detector.detected_spikes, 
                                           noisy_dataset.indices, 
                                           tolerance=50)
        
        results.append({
            'gain': gain,
            'p': metrics['precision'],
            'r': metrics['recall'],
            'f1': metrics['f1']
        })
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_gain = gain
    
    # Print all results
    for r in results:
        marker = " <-- BEST" if r['gain'] == best_gain else ""
        print(f"  Gain {r['gain']:.1f}: P={r['p']:.3f}, R={r['r']:.3f}, F1={r['f1']:.3f}{marker}")
    
    print(f"Best: gain={best_gain:.1f}, F1={best_f1:.3f}")

# %%
# Now apply to the other datasets
unlabelled_datasets = ["cwc/data/D2.mat", "cwc/data/D3.mat", "cwc/data/D4.mat", "cwc/data/D5.mat", "cwc/data/D6.mat"]
mad_gains = [3.2, 3.0, 3.0, 2.9, 2.9]
min_distance = 60

i = 0
for dataset_path in unlabelled_datasets:

    mad_gain = mad_gains[i]
    i += 1

    unlabelled_data = Dataset()
    unlabelled_data.load_from_mat_unlabelled(dataset_path)

    normalised_data = DataNormaliser(unlabelled_data)
    normalised_data.normalise(window_size=1000)
    
    unlabelled_filter = BandpassFilter(normalised_data)
    #unlabelled_filter.apply_band_pass_filter(filter_low=300, filter_high=3000, sample_rate=25000, order=4)
    
    unlabelled_spikes = SpikeDetector(unlabelled_filter)
    unlabelled_mad = unlabelled_spikes.calculate_mad()
    unlabelled_spikes.detect_spikes(mad=unlabelled_mad, mad_gain=mad_gain, distance=min_distance)

    if in_jupyter():
        plt.plot(unlabelled_filter.filtered_data)
        plt.scatter(unlabelled_spikes.detected_spikes, unlabelled_filter.filtered_data[unlabelled_spikes.detected_spikes], color='red')
        plt.show()
    
    unlabelled_processor = SignalProcessor(normalised_data)
    unlabelled_processor.filtered_data = unlabelled_filter.filtered_data
    unlabelled_processor.detected_spikes = unlabelled_spikes.detected_spikes
    unlabelled_processor.align_spikes(target_peak_pos=20, window_size=64)

    unlabelled_processor.scaler = processor_new.scaler  # Use the same scaler as training
    unlabelled_processor.pca = processor_new.pca  # Use the same PCA as training
    unlabelled_processor.extract_features()
    
    unlabelled_predictions = classifier_new.classifier.predict(unlabelled_processor.features)


    unlabelled_data.write_to_mat(dataset_path.replace('cwc/data/', 'cwc/data/output/'), unlabelled_processor.aligned_indices, unlabelled_predictions)
    print(f"Predictions saved to {dataset_path.replace('cwc/data/;', 'cwc/data/output/')}")
    print(f"Predictions for {dataset_path}: {np.bincount(unlabelled_predictions)}")
    print(f"Total spikes detected: {len(unlabelled_processor.aligned_indices)}")

    
    


# %%
# Add this after your loop to verify
import scipy.io as spio
test_file = spio.loadmat('cwc/data/output/D6.mat')
print(f"D2 Index length: {len(test_file['Index'][0])}")
print(f"D2 Class length: {len(test_file['Class'][0])}")
print(f"Match: {len(test_file['Index']) == len(test_file['Class'])}")


