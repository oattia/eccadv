
class NeuralNetModel:
    """
    Abstract Class to abstract the details of the libraries used (Keras, TF, Pytorch)
    """
    def initialize(self, input_dim, output_dim):
        raise NotImplementedError

    def train_batch(self, features, labels):
        raise NotImplementedError

    def predict_batch(self, features):
        raise NotImplementedError

    def save_to(self, path):
        raise NotImplementedError

    def load_from(self, path):
        raise NotImplementedError


class ModelManager:
    """
    Class to manage different NeuralNetModel models by providing the interface for other classes to access the model:
        Saving, loading, relationship to dataset-coders-attackers, parameter management, evaluation management.
    """
    def __init__(self, name, config):
        self.name = name
        self.nn: NeuralNetModel = None

        # loading params

        # init params
        self.dataset = None
        self.ccoder = None
        self.output_dir = None

        # structure params
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]

        # training params
        self.trainable = config.get("is_trainable", True)
        self.batch_size = config.get("batch_size", 128)
        self.optimizer = config.get("optimizer", "sgd")
        self.lr = config.get("lr", 0.001)
        self.epochs = config.get("epochs", 6)

    def is_trainable(self):
        return self.trainable

    def initialize(self, dataset, coder, output_dir):
        raise NotImplementedError

    def train_model(self):
        for e in range(1, self.epochs+1):
            for features, labels in self.dataset.iter_train(self.batch_size):
                encoded_labels = [self.ccoder.encode(y) for y in labels]
                self.nn.train_batch(features, encoded_labels)
        self.nn.save_to(self.output_dir)

    def predict_labels(self):
        nn_labels = self.nn.predict_batch(self.dataset.features_test)
        return [self.ccoder.decode(nn_label) for nn_label in nn_labels]

    def evaluate(self):
        labels = self.predict_labels()
