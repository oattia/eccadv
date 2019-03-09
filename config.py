import os.path
import logging 
import yaml

from experiment.experiment import Experiment
from dataset.mnist import Mnist
from dataset.cifar10 import Cifar10

logger = logging.getLogger(__name__)


class Config:
    """
        Reads YAML config and stores experiments parameters.
    """
    def __init__(self, cfg_fpath):
        if not os.path.isfile(cfg_fpath):
            raise Exception(f"Configuration file not found at {cfg_fpath}.")
        
        self._config = None
        with open(cfg_fpath, "r") as cfg_file:
            self._config = yaml.load(cfg_file.read())
        
        # Load experiment components
        self.datasets = self._load_datasets()

        self.source_coders = self._load_source_coders()
        self.source_coders[None] = None

        self.channel_coders = self._load_channel_coders()
        self.channel_coders[None] = None

        self.models = self._load_models()
        self.models[None] = None

        self.attackers = self._load_attackers()
        self.attackers[None] = None
    
        # Final executable block
        self.experiments = self._load_experiments()

    def _load_datasets(self):
        to_load = self._config["datasets"] or {}
        datasets = {}
        for ds_id, ds_desc in to_load.items():
            path = ds_desc.get("path", "data")
            preprocessing = ds_desc.get("preprocessing", {})
            if ds_desc["type"] == "mnist":
                datasets[ds_id] = Mnist(name=ds_id, path=path, preprocessing=preprocessing)
            elif ds_desc["type"] == "cifar10":
                datasets[ds_id] = Cifar10(name=ds_id, path=path, preprocessing=preprocessing)
            else:
                raise Exception("Unsupported dataset.")
        return datasets

    def _load_source_coders(self):
        return {}
    
    def _load_channel_coders(self):
        to_load = self._config["eccs"] or {}
        eccs = {}
        for ecc_id, ecc_desc in to_load.items():
            pass
        return eccs

    def _load_models(self):
        to_load = self._config["models"] or {}
        models = {}
        for model_id, model_desc in to_load.items():
            pass
        return models

    def _load_attackers(self):
        to_load = self._config["attackers"] or {}
        attackers = {}
        for att_id, att_desc in to_load.items():
            pass
        return attackers
        
    def _load_experiments(self):
        to_load = self._config["experiments"] or {}
        exps = {}
        for ex_id, ex_desc in to_load.items():
            exps[ex_id] = Experiment(name=ex_id,
                                     seed=ex_desc.get("seed", 777),
                                     dataset=self.datasets[ex_desc["dataset"]],
                                     ecc=self.channel_coders[ex_desc["coders"]],
                                     model=self.models[ex_desc["model"]],
                                     attacker=self.attackers[ex_desc["attacker"]])
        return exps
