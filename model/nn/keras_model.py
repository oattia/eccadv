from model.nn.neural_net import NeuralNetModel

from tensorflow import keras


class KerasNnModel(NeuralNetModel):
    def __init__(self, name, config):
        super(KerasNnModel).__init__(name, config)

    def initialize(self, input_shape, output_dim):
        raise NotImplementedError

    def train_batch(self, features, labels):
        raise NotImplementedError

    def predict(self, features):
        raise NotImplementedError

    def save_to(self, path):
        raise NotImplementedError

    def load_from(self, path):
        raise NotImplementedError

    def context(self):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError
