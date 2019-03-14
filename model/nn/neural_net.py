from enum import Enum


class Libraries(Enum):
    KERAS   = 0
    TF      = 1
    TORCH   = 2


class NeuralNetModel:
    """
    Abstract Class to hide the details of the deep learning library used (Keras, TF, Pytorch).
    """
    def __init__(self, name, config):
        self.model = None

    def initialize(self, input_shape, output_dim):
        raise NotImplementedError

    def train_batch(self, features, labels):
        """
        Train and update the parameters for one batch.
        """
        raise NotImplementedError

    def predict(self, features):
        """
        Returns the network encoding for these features.
        """
        raise NotImplementedError

    def save_to(self, path):
        """
        Dump model to path.
        """
        raise NotImplementedError

    def load_from(self, path):
        """
        Load model from path.
        """
        raise NotImplementedError

    def context(self):
        """
        Returns all information about the model necessary for attackers.
        """
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError
