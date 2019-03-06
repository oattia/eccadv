import gzip
import logging 
import pickle

from dataset.dataset import Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Mnist(Dataset):
    """
        Dataset object for MNIST dataset as downloaded from:
            http://deeplearning.net/data/mnist/mnist.pkl.gz
        It is a pickled file with train/test split.
    """
    def __init__(self, name, path, preprocessing):
        super(Mnist, self).__init__(name, path, preprocessing)
        self.mnist_file = self.path / "mnist.pkl.gz"

    def _set_arrays(self):
        with gzip.open(self.mnist_file.as_posix(), "rb") as f:
                ((self.features_train, self.labels_train),
                 (self.features_test, self.labels_test), _) = pickle.load(f, encoding="latin-1")
