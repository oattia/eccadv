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
        self.name = name
        self.config = config

        self.network_model = None
        self.n_classes = -1
        self.input_shape = None
        self.output_size = -1
        self.is_trainable = True

    def initialize(self, input_shape, output_dim, n_classes, load_from=None):
        self.input_shape = input_shape
        self.output_size = output_dim
        self.n_classes = n_classes
        if load_from:
            self.load_from(load_from.as_posix())
            self.is_trainable = self.config.get("train", False)
        else:
            self._build_model()

    def _build_model(self):
        """
        Builds a new model from scratch according to the config.
        """
        raise NotImplementedError

    def compile(self, atkr=None):
        """
        Compiles the  model adding adversarial training if specified
        """
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
