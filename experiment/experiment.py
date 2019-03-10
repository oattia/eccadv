import logging
from pathlib import Path
import time

from attacker.attacker import Attacker
from coders.channel.channel_coder import ChannelCoder
from coders.source.source_coder import SourceCoder
from dataset.dataset import Dataset
from model.model import Model

import tensorflow as tf
import torch
import numpy as np

logger = logging.getLogger(__name__)


class Experiment:
    """
    Links all components together and runs the experiment according to the configuration.
    """
    def __init__(self, name, seed, dataset, source_coder, channel_coder, model, attacker):
        self.name = name
        self.seed = seed
        self.dataset: Dataset = dataset
        self.scoder: SourceCoder = source_coder
        self.ccoder: ChannelCoder = channel_coder
        self.model: Model = model
        self.attacker: Attacker = attacker
        self.output_dir = Path("ex_output") / (name + int(time.time()))

    def _initialize(self):
        # make output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # load and wire components together
        self.dataset.load()
        self.ccoder.set_source_coder(self.scoder)
        self.ccoder.set_alphabet(self.dataset.get_labels())
        self.model.initialize()

    def _train_model(self):
        if not self.model.is_trainable():
            return
        for features, labels in self.dataset.iter_train(self.model.batch_size):
            encoded_labels = [self.ccoder.encode(y) for y in labels]
            self.model.train_batch(features, encoded_labels)
        self.model.save(self.output_dir)

    def _evaluate_model(self, model: Model):
        pass

    def _attack_model(self, model: Model) -> Model:
        pass

    def run(self):
        # set all seeds
        self._set_seed()

        # initialize experiment assets
        self._initialize()

        # start the experiment
        self._train_model()
        self._evaluate_model(self.model)
        attacked_model = self._attack_model(self.model)
        self._evaluate_model(attacked_model)

    def _set_seed(self):
        np.random.seed(self.seed)
        tf.random.set_random_seed(self.seed)
        torch.manual_seed(self.seed)

    def __repr__(self):
        return f"Experiment(name={self.name}, seed={self.seed}, dataset={self.dataset.name}, " \
            f"source_coder={self.scoder.name}, channel_coder={self.ccoder.name}, " \
            f"model={self.model.name}, attacker={self.attacker.name})"
