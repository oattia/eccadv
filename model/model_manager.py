from dataset.dataset import Dataset
from coders.channel.channel_coder import ChannelCoder
from model.nn.neural_net import NeuralNetModel
from model.attacker.attacker import Attacker

from sklearn.metrics import accuracy_score


class ModelManager:
    """
    Class to manage different NeuralNetModel models by providing the interface for other classes to access the model:
        Saving, loading, relationship to dataset-coders-attackers, parameter management, and evaluations.
    """
    def __init__(self, name, config):
        self.name = name

        # main model components
        self.nn: NeuralNetModel = None
        self.attacker: Attacker = None

        # init params
        self.dataset: Dataset = None
        self.ccoder: ChannelCoder = None
        self.output_dir = None

        # loading params
        self.load_if_exists = config.get("load_if_exists", True)

        # structure params
        self.input_dim = None
        self.output_dim = None

        # training params
        self.trainable = config.get("is_trainable", True)
        self.batch_size = config.get("batch_size", 128)
        self.optimizer = config.get("optimizer", "sgd")
        self.lr = config.get("lr", 0.001)
        self.epochs = config.get("epochs", 6)

        self.initialized = False

    def is_trainable(self):
        return self.trainable

    def initialize(self, dataset, coder: ChannelCoder, output_dir):
        self.dataset = dataset
        self.ccoder = coder
        # load and wire components together
        self.ccoder.set_alphabet(self.dataset.get_labels())
        self.output_dir = output_dir / f"model-{self.name}"
        self.input_dim = self.dataset.shape
        self.output_dim = self.ccoder.output_size()
        self.initialized = True

    def train_model(self):
        assert self.initialized
        if not self.is_trainable():
            return
        for e in range(1, self.epochs+1):
            for features, labels in self.dataset.iter_train(self.input_dim, self.batch_size):
                encoded_labels = [self.ccoder.encode(y) for y in labels]
                self.nn.train_batch(features, encoded_labels)
        self.nn.save_to(self.output_dir)

    def _predict_labels(self, features):
        nn_labels = self.nn.predict(features)
        return [self.ccoder.decode(nn_label) for nn_label in nn_labels]

    def evaluate(self):
        """
        Evaluates the model on benign and adversarial examples.
        """
        assert self.initialized
        for test_features, ground_truth_lables in self.dataset.iter_test(self.input_dim):
            benign_labels = self._predict_labels(test_features)
            perturbed_features = self.attacker.perturb(test_features)
            adv_labels = self._predict_labels(perturbed_features)

            model_acc = accuracy_score(ground_truth_lables, benign_labels)
            adv_acc = accuracy_score(ground_truth_lables, adv_labels)
            changed_acc = accuracy_score(benign_labels, adv_labels)
            yield model_acc, adv_acc, changed_acc
