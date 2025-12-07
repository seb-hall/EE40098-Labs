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
        self.window_size = 64

    def create_templates(self, window_size=64, initial_offset=0, target_offset=0):
        
        self.window_size = window_size
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


    def compare_to_templates(self):

        self.classes = np.array([])

        # compare data and indices to templates to classify spikes
        for index in self.indices:

            start = index
            end = index + self.window_size

            if end > len(self.data):
                self.classes = np.append(self.classes, -1)  # unclassified
                continue

            spike = self.data[start:end]

            # Find the negative peak within this window
            actual_peak_pos = np.argmax(spike * -1.0)
            
            # Realign so peak is at position 0 (matching template alignment)
            shift = 0 - actual_peak_pos
            aligned_spike = np.zeros(self.window_size)

            if shift > 0:
                aligned_spike[shift:] = spike[:self.window_size-shift]
            elif shift < 0:
                aligned_spike[:self.window_size+shift] = spike[-shift:]
            else:
                aligned_spike = spike

            # Normalize
            aligned_spike = (aligned_spike - np.mean(aligned_spike)) / (np.std(aligned_spike) + 1e-10)

            # Compare to templates
            best_match_class = None
            best_match_score = float('inf')

            for template, cls in zip(self.templates, self.template_classes):
                score = np.sum((aligned_spike - template) ** 2)

                if score < best_match_score:
                    best_match_score = score
                    best_match_class = cls

            self.classes = np.append(self.classes, best_match_class)