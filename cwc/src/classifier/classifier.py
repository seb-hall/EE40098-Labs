from sklearn.ensemble import RandomForestClassifier

import numpy as np

class Classifier:

    def __init__(self, features, labels):

        self.classifier = RandomForestClassifier(
            n_estimators=500,  # More trees
            max_depth=None,    # Allow full depth
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )

        self.features = features
        self.labels = labels

        self.accuracy = 0.0


    def train(self):

        self.classifier.fit(self.features, self.labels)

        self.accuracy = self.classifier.score(self.features, self.labels)