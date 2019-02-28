from pathlib import Path
import logging
import numpy as np
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
        
        # numpy arrays for features and lables
        self.features_train = None
        self.labels_train = None
        self.features_test = None
        self.labels_test = None
        
        self.loaded = False
    
    def _set_arrays(self):
        pass
    
    def _preprocess(self):
        pass
    
    def load(self):
        if self.loaded:
            return
        self._set_arrays()
        self._preprocess()
        self.loaded = True

    def _check_loaded(self):
        assert self.loaded
        assert self.features_train is not None
        assert self.labels_train is not None
        assert self.features_test is not None
        assert self.labels_test is not None
    
    def get_labels(self):
        self._check_loaded()
        return np.unique(self.labels_train)

    def get_split(self):
        self._check_loaded()
        return (self.features_train, self.labels_train, self.features_test, self.labels_test)
    
    def to_tf(self):
        self._check_loaded()
        return map(tf.convert_to_tensor, self.get_split())

    def to_torch(self):
        self._check_loaded()
        return map(torch.tensor, self.get_split())
        