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

    def create_templates(self, window_size=64, initial_offset=0, target_offset=20):
        
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
        # compare data and indices to templates to classify spikes
        for index in self.indices:

            if index + self.window_size >= len(self.data) - 1:
                self.classes = np.append(self.classes, -1)  # unclassified
                continue

            spike = self.data[index:index + self.window_size]
            spike = (spike - np.mean(spike)) / (np.std(spike) + 1e-10)  # normalize spike

            best_match_class = None
            best_match_score = float('inf')

            for template, cls in zip(self.templates, self.template_classes):
                score = np.sum((spike - template) ** 2)  # Mean Squared Error

                if score < best_match_score:
                    best_match_score = score
                    best_match_class = cls

            # assign the best matching class to the spike
            self.classes = np.append(self.classes, best_match_class)
        