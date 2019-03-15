import os.path
import logging 
import yaml

from coders.source import *
from coders.channel import *
from dataset.mnist import Mnist
from dataset.cifar10 import Cifar10
from experiment.experiment import Experiment

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
        self.scoders = self._load_scoders()
        self.ccoders = self._load_ccoders()
        self.models = self._load_models()
        self.attackers = self._load_attackers()
        self.experiments = self._load_experiments()

    def _load_datasets(self):
        to_load = self._config["datasets"] or {}
        datasets = {}
        for ds_id, ds_desc in to_load.items():
            path = ds_desc.get("path", "data")
            preprocessing = ds_desc.get("preprocessing", {})
            ds_type = ds_desc["type"]
            if ds_type == "mnist":
                datasets[ds_id] = Mnist(name=ds_id, path=path, preprocessing=preprocessing)
            elif ds_type == "cifar10":
                datasets[ds_id] = Cifar10(name=ds_id, path=path, preprocessing=preprocessing)
            else:
                raise Exception(f"Unsupported dataset type {ds_type}")
        return datasets

    def _load_scoders(self):
        to_load = self._config["source_coders"] or {}
        scoders = {}
        for coder_id, scoder_desc in to_load.items():
            s_type = scoder_desc["type"]
            allow_zero = s_type.get("allow_zero", False)
            output_size = s_type.get("output_size", -1)
            if s_type == "rand":
                scoders[coder_id] = RandomCoder(name=coder_id, allow_zero=allow_zero, output_size=output_size)
            elif s_type == "gray":
                scoders[coder_id] = GrayCoder(name=coder_id, allow_zero=allow_zero, output_size=output_size)
            elif s_type == "bcd":
                scoders[coder_id] = BcdCoder(name=coder_id, allow_zero=allow_zero, output_size=output_size)
            elif s_type == "oh":
                scoders[coder_id] = OneHotCoder(name=coder_id, allow_zero=allow_zero, output_size=output_size)
            else:
                raise Exception(f"Unsupported source coder type {s_type}")
        return scoders

    def _load_ccoders(self):
        to_load = self._config["channel_coders"] or {}
        ccoders = {}
        for coder_id, coder_desc in to_load.items():
            c_type = coder_desc["type"]
            prob = coder_desc.get("prob", False)
            factor = coder_desc.get("factor", 1)
            if c_type == "rs":
                ccoders[coder_id] = ReedSolomonCoder(name=coder_id, prob=prob, factor=factor, n=coder_desc["n"])
            elif c_type == "rep":
                ccoders[coder_id] = RepetitionCoder(name=coder_id,  prob=prob, factor=factor,
                                                    repetition=coder_desc["rep"],
                                                    method=coder_desc.get("method", "block"))
            elif c_type == "dummy":
                ccoders[coder_id] = DummyChannelCoder(name=coder_id, prob=prob, factor=factor)
            else:
                raise Exception(f"Unsupported channel coder type {c_type}")
        return ccoders

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
                                     dataset=self.datasets.get(ex_desc["dataset"], None),
                                     source_coder=self.scoders.get(ex_desc["source_coder"], None),
                                     channel_coder=self.ccoders.get(ex_desc["channel_coder"], None),
                                     nn_model=self.models.get(ex_desc["model"], None))
        return exps
