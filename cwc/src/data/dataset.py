################################################################
##
## EE40098 Coursework C
##
## File         :  dataset.py
## Author       :  samh25
## Created      :  2025-12-02 (YYYY-MM-DD)
## License      :  MIT
## Description  :  Dataset class for loading and saving data.
##
################################################################

################################################################
## MARK: INCLUDES
################################################################

import scipy.io as spio
import numpy as np

################################################################
## MARK: CLASS DEFINITIONS
################################################################

class Dataset:

    ############################################################
    ## CONSTRUCTOR

    # instantiate a new dataset
    def __init__(self):

        # create empty dataset
        self.data = np.array([])
        self.indices = np.array([])
        self.classes = np.array([])

    ############################################################
    ## INSTANCE METHODS

    # load dataset from a labelled .mat file
    def load_from_mat(self, file_path):
        mat = spio.loadmat(file_path)
        self.data = mat["d"][0]
        self.indices = mat["Index"][0]
        self.classes = mat["Class"][0]
        

    # load dataset from an unlabelled .mat file
    def load_from_mat_unlabelled(self, file_path):
        mat = spio.loadmat(file_path)
        self.data = mat["d"][0]


    # write dataset to a labelled .mat file
    def write_to_mat(self, file_path, indices, classes):
        mat_dict = {
            "Index": indices,
            "Class": classes
        }
        spio.savemat(file_path, mat_dict)


    # split a dataset into training and test sets
    def split_into_test(self, test_ratio=0.1):
        total_samples = len(self.data)
        test_size = int(total_samples * test_ratio)

        test_start_index = total_samples - test_size
        test_mask = self.indices >= test_start_index
        train_mask = ~test_mask

        # create test dataset from the end of the data
        test_dataset = Dataset()
        test_dataset.data = self.data[test_start_index:] # last test_size samples
        
        # adjust indices to be relative to the test data
        test_dataset.indices = self.indices[test_mask] - test_start_index
        test_dataset.classes = self.classes[test_mask]
        
        # update current dataset to only have training data
        self.data = self.data[:test_start_index] # all but last test_size samples
        self.indices = self.indices[train_mask]
        self.classes = self.classes[train_mask]
        
        return test_dataset
