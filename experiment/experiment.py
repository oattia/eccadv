import logging
from pathlib import Path

from coders.channel.channel_coder import ChannelCoder
from coders.source.source_coder import SourceCoder
from dataset.dataset import Dataset
from model.model_manager import ModelManager

import tensorflow as tf
import torch
import numpy as np

logger = logging.getLogger(__name__)


class Experiment:
    """
    Links components together, runs the experiment according to the configuration, and manages outputs.
    """
    def __init__(self, name, seed, dataset, source_coder, channel_coder, nn_model, adversary):
        self.name = name
        self.seed = seed
        self.dataset: Dataset = dataset
        self.scoder: SourceCoder = source_coder
        self.ccoder: ChannelCoder = channel_coder
        self.model_manager: ModelManager = nn_model
        self.output_dir = Path("exp_output") / name

    def _initialize(self):
        # create the output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset.load()
        self.ccoder.set_source_coder(self.scoder)
        self.model_manager.initialize(self.dataset, self.ccoder, self.output_dir)

    def run(self):
        # set all seeds
        self._set_seed()

        # initialize and set experiment assets
        self._initialize()

        # start the experiment sequence

        # 1- train the model if needed
        self.model_manager.train_model()

        # 2- Evaluate the trained model on benign and adversarial examples
        self.model_manager.evaluate()

    def _set_seed(self):
        np.random.seed(self.seed)
        tf.random.set_random_seed(self.seed)
        torch.manual_seed(self.seed)

    def __repr__(self):
        return f"Experiment(name={self.name}, seed={self.seed}, dataset={self.dataset.name}, " \
            f"source_coder={self.scoder.name}, channel_coder={self.ccoder.name}, " \
            f"model={self.model_manager.name})"
