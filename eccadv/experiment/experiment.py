import logging
from pathlib import Path

from coders.channel import ChannelCoder
from model.utils import threshold
from dataset.utils import dump_image

import pandas as pd
import tensorflow as tf
import numpy as np
import torch
from tqdm import tqdm, trange


logger = logging.getLogger(__name__)


class Experiment:

    eval_type1 = "benign"
    eval_type2 = "adv"
    dump_type1 = "wrong"
    dump_type2 = "couldnt_predict"

    exp_dir = Path("exp_output")

    """
    Class for Experiment logic.
    Links and initializes components, manages NeuralNetModel models, attacker and evaluations.
    """
    def __init__(self, name, seed, dataset, source_coder, channel_coder, model, attacker, thresholding, max_steps, eval_freq, adv_train):
        # passed
        self.name = name
        self.seed = seed
        self.dataset = dataset
        self.scoder = source_coder
        self.ccoder = channel_coder
        self.model = model
        self.attacker = attacker
        self.thresholding = thresholding
        self.max_steps = max_steps
        self.eval_freq = eval_freq
        self.adv_train = adv_train

        # computed state
        self.output_dir = Experiment.exp_dir / name
        self.models_dir = Experiment.exp_dir / "models"
        self.model_id = self._global_model_id()

    def _set_seed(self):
        np.random.seed(self.seed)
        tf.random.set_random_seed(self.seed)
        torch.manual_seed(self.seed)

    def _global_model_id(self):
        orderd_params = "_".join([str(v) + "-" + str(k) for k, v in sorted(self.model.config.items())])
        return "{}_{}_{}_{}_{}".format(self.model.name, str(self.model.__class__.__name__), orderd_params, self.scoder.name, self.ccoder.name)

    def _initialize(self):
        # set all seeds
        self._set_seed()

        # create the output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # create models directory if doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # load and preprocess the dataset according to the config
        self.dataset.load()

        # set up the coders for this dataset
        ds_labels = self.dataset.get_labels()
        self.scoder.set_alphabet(ds_labels)
        self.ccoder.set_source_coder(self.scoder)

        load_from = self.models_dir / self.model_id if (self.models_dir / self.model_id).exists() else None

        # set up the model and the attacker
        self.model.initialize(self.dataset.shape, self.ccoder.output_size(), len(ds_labels), load_from=load_from)
        self.attacker.initialize(self.model)

        if self.model.is_trainable:
            adv_trainer = self.attacker if self.adv_train else None
            self.model.compile(adv_trainer)

    def _train_model(self):
        if not self.model.is_trainable:
            return
        batch_size = self.model.config["batch_size"]
        epochs = self.model.config["epochs"]
        save_model = self.model.config.get("save", True)
        t_batches = self.dataset.get_num_train_batches(batch_size)
        epoch_pbar = trange(1,  epochs + 1)
        for e in epoch_pbar:
            batches_pbar = tqdm(self.dataset.iter_train(self.dataset.shape, batch_size), total=t_batches)
            for i, (features, labels) in enumerate(batches_pbar):
                encoded_labels = np.array([[int(bit) for bit in codeword] for codeword in [self.ccoder.encode(y) for y in labels]])
                metrics = self.model.train_batch(features, encoded_labels.astype(float))
                epoch_pbar.set_description("Epoch: {}. Batch: {}. Loss: {:0.2f}. Acc: {:0.2f}".format(e, i+1, metrics[0], metrics[1]))
        if save_model:
            self.model.save_to(self.models_dir / self.model_id)

    def _predict_labels(self, features):
        return np.array([self.ccoder.decode(threshold(y, **self.thresholding)) for y in self.model.predict(features)])

    def _dump_samples(self, dump_type, eval_type, features, ground_truth_lables, predicted, idx, sample_size, step):
        choice_size = min(sample_size, len(idx))
        idx_sample = np.random.choice(idx, choice_size, replace=False)
        for i in range(choice_size):
            image_name = "{}_{}_step_{}_label_{}".format(eval_type,
                                                         dump_type,
                                                         str(step),
                                                         ground_truth_lables[idx_sample[i]])
            if dump_type == Experiment.dump_type1:
                image_name += "_predicted_{}".format(predicted[idx_sample[i]])
            dump_image(features[idx_sample[i]], self.output_dir / (image_name + "_" + str(i) + ".png"))

    def _eval_labels(self, eval_type, step, test_features, ground_truth, predicted, sample_size=2):
        ground_truth_lables = ground_truth.astype(str)
        correct_idx = [i for i in range(len(predicted)) if predicted[i] == ground_truth_lables[i]]
        couldnot_predict_idx = [i for i in range(len(predicted)) if predicted[i].startswith(ChannelCoder.CANT_DECODE)]
        couldnot_predict_idx_set = set(couldnot_predict_idx)
        wrong_idx = [i for i in range(len(predicted)) if predicted[i] != ground_truth_lables[i] and i not in couldnot_predict_idx_set]

        self._dump_samples(Experiment.dump_type1, eval_type, test_features, ground_truth_lables, predicted,
                           wrong_idx, sample_size, step)
        self._dump_samples(Experiment.dump_type2, eval_type, test_features, ground_truth_lables, predicted,
                           couldnot_predict_idx, sample_size, step)
        return correct_idx, wrong_idx, couldnot_predict_idx

    def _evaluate(self):
        """
        Evaluates the model on benign and adversarial examples.
        """
        results_to_ret = []

        test_features, ground_truth = list(self.dataset.iter_test(self.dataset.shape))[0]

        labels = self._predict_labels(test_features)
        correct_idx, wrong_idx, couldnot_predict_idx = \
            self._eval_labels(Experiment.eval_type1, 0, test_features, ground_truth, labels)

        results_to_ret.append({
            "step": 0,
            "correct": len(correct_idx),
            "wrong": len(wrong_idx),
            "couldnot_predict": len(couldnot_predict_idx)
        })

        for step in range(1, self.max_steps+1):
            test_features = self.attacker.perturb(test_features)
            if (step == 1) or (step % self.eval_freq == 0):
                adv_labels = self._predict_labels(test_features)
                adv_correct_idx, adv_wrong_idx, adv_couldnot_predict_idx = \
                    self._eval_labels(Experiment.eval_type2, step, test_features, ground_truth, adv_labels)

                results_to_ret.append({
                    "step": step,
                    "correct": len(adv_correct_idx),
                    "wrong": len(adv_wrong_idx),
                    "couldnot_predict": len(adv_couldnot_predict_idx),
                })

                if len(adv_correct_idx) == 0:
                    break

        return pd.DataFrame(results_to_ret, columns=["step", "correct", "wrong", "couldnot_predict"])

    def run(self):
        # 1- initialize and set experiment assets
        self._initialize()

        # 2- train the model if needed
        self._train_model()

        # 3- Evaluate the trained model on benign and adversarial examples
        return self._evaluate()

    def cleanup(self):
        # This is done to free up the memory after execution
        self.name = None
        self.seed = None
        self.dataset = None
        self.scoder = None
        self.ccoder = None
        self.model = None
        self.attacker = None
        self.thresholding = None
        self.max_steps = None
        self.eval_freq = None
        self.adv_train = None

        # computed state
        self.output_dir = None
        self.models_dir = None
        self.model_id = None

    def __repr__(self):
        return "Experiment(name={}, seed={}, dataset={}, source_coder={}, channel_coder={}, model={})".format(self.name, self.seed, self.dataset.name, self.scoder.name, self.ccoder.name, self.model.name)
