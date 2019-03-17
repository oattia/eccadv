import logging
from pathlib import Path

from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
import torch

logger = logging.getLogger(__name__)


class Experiment:
    """
    Class for Experiment logic.
    Links and initializes components, manages NeuralNetModel models, attacker and evaluations.
    """
    def __init__(self, name, seed, dataset, source_coder, channel_coder, model, attacker):
        self.name = name
        self.seed = seed
        self.dataset = dataset
        self.scoder = source_coder
        self.ccoder = channel_coder
        self.model = model
        self.attacker = attacker
        self.output_dir = Path("exp_output") / name

    def _set_seed(self):
        np.random.seed(self.seed)
        tf.random.set_random_seed(self.seed)
        torch.manual_seed(self.seed)

    def _initialize(self):
        # set all seeds
        self._set_seed()

        # create the output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # load and preprocess the dataset according to the config
        self.dataset.load()

        # set up the coders for this dataset
        ds_labels = self.dataset.get_labels()
        self.ccoder.set_source_coder(self.scoder)
        self.ccoder.set_alphabet(ds_labels)

        # set up the model and the attacker
        self.model.initialize(self.dataset.shape, self.ccoder.output_size(), len(ds_labels))
        self.attacker.initialize(self.model)

    def _train_model(self):
        if not self.model.is_trainable:
            return
        for e in range(1, self.model.config["epochs"] + 1):
            for features, labels in self.dataset.iter_train(self.dataset.shape, self.model.config["batch_size"]):
                encoded_labels = np.array([[int(b) for b in code] for code in [self.ccoder.encode(y) for y in labels]])
                self.model.train_batch(features, encoded_labels)
        self.model.save_to(self.output_dir / self.model.name)

    def _predict_labels(self, features):
        nn_labels = self.model.predict(features)
        return np.array([self.ccoder.decode(nn_label) for nn_label in nn_labels])

    def _evaluate(self):
        """
        Evaluates the model on benign and adversarial examples.
        """
        for test_features, ground_truth_lables in self.dataset.iter_test(self.dataset.shape):
            model_acc, adv_acc, changed_acc = 0.0, 0.0, 0.0
            benign_labels = self._predict_labels(test_features)
            model_acc = accuracy_score(ground_truth_lables.astype(str), benign_labels)

            perturbed_features = self.attacker.perturb(test_features)
            adv_labels = self._predict_labels(perturbed_features)
            adv_acc = accuracy_score(ground_truth_lables.astype(str), adv_labels)
            changed_acc = accuracy_score(benign_labels, adv_labels)

            yield model_acc, adv_acc, changed_acc

    def run(self):
        # 1- initialize and set experiment assets
        self._initialize()

        # 2- train the model if needed
        self._train_model()

        # 3- Evaluate the trained model on benign and adversarial examples
        for result in self._evaluate():
            print(result)

    def __repr__(self):
        return "Experiment(name={}, seed={}, dataset={}, source_coder={}, channel_coder={}, model={})".format(self.name, self.seed, self.dataset.name, self.scoder.name, self.ccoder.name, self.model.name)
