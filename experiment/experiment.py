import logging

import tensorflow as tf
import torch
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Experiment:
    def __init__(self, name, seed, dataset, ecc, model, attacker):
        self.name = name
        self.seed = seed
        self.dataset = dataset
        self.ecc = ecc
        self.model = model
        self.attacker = attacker

    def run(self):
        self._set_seed()
        self.dataset.load()
        print(f"{self.dataset.name}: {self.dataset.get_labels()}")
        print(f"{self.dataset.name}: {self.dataset.features_train[0, :]}: {self.dataset.features_train[0, :].shape}")
        logger.info(f"{self.dataset.get_labels()}")
    
    def _set_seed(self):
        tf.random.set_random_seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
