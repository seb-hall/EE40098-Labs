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

    def create_templates(self, window_size=64, initial_offset=0, target_offset=20):
        
        unique_classes = np.unique(self.classes)

        for cls in unique_classes:

            # extract indices of spikes belonging to the current class
            class_indices = self.indices[self.classes == cls]

            spikes = []

            for index in class_indices:

                start = index - initial_offset
                end = start + window_size

                if end <= len(self.data):
                    spike = self.data[start:end]

                    actual_peak_pos = np.argmax(spike * -1.0)
                    shift = target_offset - actual_peak_pos

                    start = start - shift
                    end = start + window_size

                    if end <= len(self.data):
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


    def detect_with_templates(self, correlation_threshold=0.65, min_distance=25):
        all_detections = []

        # Normalize the signal
        signal_norm = (self.data - np.mean(self.data)) / (np.std(self.data) + 1e-10)

        for i, (template, cls) in enumerate(zip(self.templates, self.template_classes)):
            
            if len(template) == 0 or np.all(template == 0):
                continue

            # Cross-correlation
            correlation = np.correlate(signal_norm, template, mode='valid')

            # Normalize correlation
            template_energy = np.linalg.norm(template)
            correlation = correlation / (template_energy * np.sqrt(len(template)))
            
            # Find peaks in correlation
            peaks, properties = find_peaks(
                correlation,
                height=correlation_threshold,
                distance=min_distance,
                prominence=0.05
            )

            # CORRECT adjustment: 
            # correlation[i] means template starts at signal[i]
            # Since template has peak at position 0 (target_offset=0),
            # the spike peak in signal is at position i + 0
            adjusted_peaks = peaks + 0  # Or just: adjusted_peaks = peaks

            # Store detections with their correlation strength
            for peak, corr_value in zip(adjusted_peaks, properties['peak_heights']):
                all_detections.append({
                    'index': peak,
                    'class': cls,
                    'correlation': corr_value
                })

        # Sort by correlation strength
        all_detections.sort(key=lambda x: x['correlation'], reverse=True)

        # Non-maximum suppression
        final_detections = []
        used_positions = set()
        
        for detection in all_detections:
            idx = detection['index']
            
            # Check if too close to existing detection
            too_close = any(abs(idx - used_idx) < min_distance for used_idx in used_positions)
            
            if not too_close:
                final_detections.append(detection)
                used_positions.add(idx)
        
        # Sort by index
        final_detections.sort(key=lambda x: x['index'])
        
        # Extract results
        self.indices = np.array([d['index'] for d in final_detections])
        self.classes = np.array([d['class'] for d in final_detections])