import logging
import numpy as np
import pickle

from dataset.dataset import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Cifar10(Dataset):
    """
        Dataset object for CIFAR10 dataset as downloaded from:
        https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

        The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch. 
        Each of these files is a Python "pickled" dictionary that has:
        - data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
            The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
            The image is stored in row-major order, so that the first 32 entries of the array are
                the red channel values of the first row of the image.

        - labels -- a list of 10000 numbers in the range 0-9.
            The number at index i indicates the label of the ith image in the array data.
    """
    def __init__(self, name, path, preprocessing):
        super(Cifar10, self).__init__(name, path, preprocessing)
        self.train_filenames = [self.path / "data_batch_{}".format(str(i)) for i in range(1, 6)]
        self.test_filename = self.path / "test_batch"
        self.labels_map_file = self.path / "batches.meta"

    @staticmethod
    def _unpickle_batch(file):
        with open(file, "rb") as fo:
            data = pickle.load(fo, encoding="bytes")
        return data

    def _set_arrays(self):
        self.features_train = np.zeros((50000, 3072))
        self.labels_train = np.zeros((50000, 1), dtype=np.int8)
        for i, train_fname in enumerate(self.train_filenames):
            unpickled_train_data = Cifar10._unpickle_batch(train_fname)
            batch_idx_start = i * 10000
            batch_idx_end = batch_idx_start + 10000
            self.features_train[batch_idx_start:batch_idx_end, :] = unpickled_train_data[b"data"]
            self.labels_train[batch_idx_start:batch_idx_end] = np.array(unpickled_train_data[b"labels"],
                                                                        ndmin=2, dtype=np.int8).T

        unpickled_test_data = self._unpickle_batch(self.test_filename)
        self.features_test = unpickled_test_data[b"data"]
        self.labels_test = np.array(unpickled_test_data[b"labels"], ndmin=2, dtype=np.int8).T
        self.shape = (32, 32, 3)
