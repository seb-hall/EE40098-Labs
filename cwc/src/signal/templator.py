# class to create a template profile for each detected spike class

import numpy as np

class Templator:

    def __init__(self, filtered_dataset):
        self.data = filtered_dataset.filtered_data
        self.indices = filtered_dataset.indices
        self.classes = filtered_dataset.classes

        self.templates = []

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
            else:
                print(f"No spikes found for class {cls} to create a template.")
                self.templates.append(np.zeros(window_size)) 

    
