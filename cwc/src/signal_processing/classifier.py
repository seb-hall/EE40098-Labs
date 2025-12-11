################################################################
##
## EE40098 Coursework C
##
## File         :  classifier.py
## Author       :  samh25
## Created      :  2025-12-02 (YYYY-MM-DD)
## License      :  MIT
## Description  :  Classifier class for spike classification.
##
################################################################

from sklearn.ensemble import RandomForestClassifier
import numpy as np

################################################################
## MARK: CLASS DEFINITIONS
################################################################

class Classifier:

    ############################################################
    ## CONSTRUCTOR

    # instantiate a new classifier
    def __init__(self, features, labels):

        # create a new random forest classifier
        self.classifier = RandomForestClassifier(
            n_estimators=1000,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )

        self.features = features
        self.labels = labels

        self.accuracy = 0.0

    ############################################################
    ## INSTANCE METHODS

    # train the classifier and evaluate accuracy
    def train(self):

        self.classifier.fit(self.features, self.labels)
        self.accuracy = self.classifier.score(self.features, self.labels)
        