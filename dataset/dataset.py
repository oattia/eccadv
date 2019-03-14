import math
from pathlib import Path
import logging

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf 
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Dataset:
    """
    Abstraction to load and preprocess the datasets into numpy arrays.
    """
    def __init__(self, name, path, preprocessing):
        self.name = name
        self.path = Path(path)
        self.preprocessing = preprocessing
        
        # numpy arrays for features and labels
        # 2-D matrices (#samples x unrolled vector)
        self.features_train: np.array = None
        self.labels_train: np.array = None
        self.features_test: np.array = None
        self.labels_test: np.array = None

        # real shape tuple of the features
        self.shape = None
        
        self.loaded = False
    
    def _set_arrays(self):
        raise NotImplementedError
    
    def _preprocess(self):
        sample = self.preprocessing.get("sample", None)
        if sample:
            splitter = StratifiedShuffleSplit(train_size=sample)
            train_index, _ = next(splitter.split(self.features_train, self.labels_train))
            self.features_train = self.features_train[train_index, :]
            self.labels_train = self.labels_train[train_index]

    def load(self):
        if self.loaded:
            return
        self._set_arrays()
        self._preprocess()
        self.loaded = True

    def _is_loaded(self):
        assert self.loaded
        assert self.features_train is not None
        assert self.labels_train is not None
        assert self.features_test is not None
        assert self.labels_test is not None
        assert self.shape is not None

    @staticmethod
    def _iter_array(batch_size, shape, features, labels):
        assert len(features) == len(labels)
        num_batches = math.ceil(len(labels) / batch_size)
        for i in range(num_batches):
            start = i * batch_size
            end = min(i + batch_size, len(features))
            yield features[start: end, :].reshape((batch_size,) + shape), labels[start: end]

    def iter_train(self, shape, batch_size):
        self._is_loaded()
        assert batch_size > 0
        return Dataset._iter_array(batch_size, shape, self.features_train, self.labels_train)

    def iter_test(self, shape, batch_size=None):
        self._is_loaded()
        if not batch_size:
            batch_size = len(self.labels_test)
        assert batch_size > 0
        return Dataset._iter_array(batch_size, shape, self.features_test, self.labels_test)
    
    def get_labels(self):
        self._is_loaded()
        return np.unique(self.labels_train)

    def get_split(self):
        self._is_loaded()
        return self.features_train, self.labels_train, self.features_test, self.labels_test
    
    def to_tf(self):
        self._is_loaded()
        return map(tf.convert_to_tensor, self.get_split())

    def to_torch(self):
        self._is_loaded()
        return map(torch.tensor, self.get_split())
