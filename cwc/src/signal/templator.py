# class to create a template profile for each detected spike class

import numpy as np
from scipy.signal import find_peaks

class Templator:

    def __init__(self, filtered_dataset):
        self.data = filtered_dataset.filtered_data
        self.indices = filtered_dataset.indices
        self.classes = filtered_dataset.classes

        self.templates = []
        self.template_classes = []

    def create_templates(self, window_size=64, peak_offset=20):
        
        unique_classes = np.unique(self.classes)

        for cls in unique_classes:

            # extract indices of spikes belonging to the current class
            class_indices = self.indices[self.classes == cls]

            spikes = []

            for index in class_indices:

                start = index - peak_offset
                end = index + window_size

                if index + window_size <= len(self.data):
                    spike = self.data[start:end]
                    spike = (spike - np.mean(spike)) / (np.std(spike) + 1e-10)  # normalize spike
                    spikes.append(spike)

            if spikes:
                self.templates.append(np.mean(spikes, axis=0))
                self.template_classes.append(cls)
            else:
                print(f"No spikes found for class {cls} to create a template.")
                self.templates.append(np.zeros(window_size)) 
                self.template_classes.append(cls)


    def detect_with_templates(self, correlation_threshold=0.65, 
                                     min_distance=25, max_detections_per_class=500):
        all_detections = []

        # normalise the signal
        signal_norm = (self.data - np.mean(self.data)) / (np.std(self.data) + 1e-10)

        i = 0
        for template in self.templates:

            # compute cross-correlation
            correlation = np.correlate(signal_norm, template, mode='valid')

            # Normalize correlation
            correlation = correlation / (np.linalg.norm(template) * np.sqrt(len(template)))
            
            # Find peaks in correlation
            peaks, properties = find_peaks(
                correlation,
                height=correlation_threshold,
                distance=min_distance,
                prominence=0.1
            )

            # Adjust peak positions (correlation shifts the signal)
            adjusted_peaks = peaks + len(template) // 2

            # Store detections with their correlation strength
            for peak, corr_value in zip(adjusted_peaks, properties['peak_heights']):
                all_detections.append({
                    'index': peak,
                    'class': self.template_classes[i],
                    'correlation': corr_value
                })
                
            i += 1

        # Sort by correlation strength
        all_detections.sort(key=lambda x: x['correlation'], reverse=True)

        # Non-maximum suppression: keep highest correlation within distance window
        final_detections = []
        used_positions = set()
        
        for detection in all_detections:
            idx = detection['index']
            
            # Check if position is too close to already selected spike
            too_close = False
            for used_idx in used_positions:
                if abs(idx - used_idx) < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                final_detections.append(detection)
                used_positions.add(idx)
        
        # Sort by index
        final_detections.sort(key=lambda x: x['index'])
        
        # Extract indices and classes
        self.indices = np.array([d['index'] for d in final_detections])
        self.classes = np.array([d['class'] for d in final_detections])
