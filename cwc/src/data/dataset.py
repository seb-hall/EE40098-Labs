
import scipy.io as spio
import numpy as np

class Dataset:

    def __init__(self):
        self.data = np.array([])
        self.indices = np.array([])
        self.classes = np.array([])

    def load_from_mat(self, file_path):
        mat = spio.loadmat(file_path)
        self.data = mat["d"][0]
        self.indices = mat["Index"][0]
        self.classes = mat["Class"][0]

    def load_from_mat_unlabelled(self, file_path):
        mat = spio.loadmat(file_path)
        self.data = mat["d"][0]

    def write_to_mat(self, file_path, indices, classes):
        mat_dict = {
            "Index": indices,
            "Class": classes
        }
        spio.savemat(file_path, mat_dict)

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
        
        self.data = self.data[:test_start_index] # all but last test_size samples
        self.indices = self.indices[train_mask]
        self.classes = self.classes[train_mask]
        
        return test_dataset
