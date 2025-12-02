
import scipy.io as spio
import numpy as np

class DataLoader:

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