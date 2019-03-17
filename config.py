import os.path
import logging 
import yaml

from coders.source import *
from coders.channel import *
from dataset import *
from model.attacker import *
from model.nn import *
from experiment.experiment import Experiment

logger = logging.getLogger(__name__)


class Config:
    """
    Reads YAML config and stores experiments parameters.
    """
    def __init__(self, cfg_fpath):
        if not os.path.isfile(cfg_fpath):
            raise Exception("Configuration file not found at {}.".format(cfg_fpath))
        
        self._config = None
        with open(cfg_fpath, "r") as cfg_file:
            self._config = yaml.load(cfg_file.read(), Loader=yaml.FullLoader)
        
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
                raise Exception("Unsupported dataset type {}".format(ds_type))
        return datasets

    def _load_scoders(self):
        to_load = self._config["source_coders"] or {}
        scoders = {}
        for coder_id, scoder_desc in to_load.items():
            s_type = scoder_desc["type"]
            allow_zero = scoder_desc.get("allow_zero", False)
            output_size = scoder_desc.get("output_size", -1)
            if s_type == "rand":
                scoders[coder_id] = RandomCoder(name=coder_id, allow_zero=allow_zero, output_size=output_size)
            elif s_type == "gray":
                scoders[coder_id] = GrayCoder(name=coder_id, allow_zero=allow_zero, output_size=output_size)
            elif s_type == "bcd":
                scoders[coder_id] = BcdCoder(name=coder_id, allow_zero=allow_zero, output_size=output_size)
            elif s_type == "oh":
                scoders[coder_id] = OneHotCoder(name=coder_id, allow_zero=allow_zero, output_size=output_size)
            elif s_type == "dummy":
                scoders[coder_id] = DummySourceCoder(name=coder_id, codes=open(scoder_desc["from"], "r").read().split())
            else:
                raise Exception("Unsupported source coder type {}".format(s_type))
        return scoders

    def _load_ccoders(self):
        to_load = self._config["channel_coders"] or {}
        ccoders = {}
        for coder_id, ccoder_desc in to_load.items():
            c_type = ccoder_desc["type"]
            prob = ccoder_desc.get("prob", False)
            factor = ccoder_desc.get("factor", 1)
            if c_type == "rs":
                ccoders[coder_id] = ReedSolomonCoder(name=coder_id, prob=prob, factor=factor, n=ccoder_desc["n"])
            elif c_type == "rep":
                ccoders[coder_id] = RepetitionCoder(name=coder_id,  prob=prob, factor=factor,
                                                    repetition=ccoder_desc["rep"],
                                                    method=ccoder_desc.get("method", "block"))
            elif c_type == "dummy":
                ccoders[coder_id] = DummyChannelCoder(name=coder_id, prob=prob, factor=factor)
            else:
                raise Exception("Unsupported channel coder type {}".format(c_type))
        return ccoders

    def _load_models(self):
        to_load = self._config["models"] or {}
        models = {}
        for model_id, model_desc in to_load.items():
            model_type = model_desc["type"]
            params = model_desc["params"]
            if model_type == Libraries.KERAS.name:
                models[model_id] = KerasNnModel(model_id, params)
            pass
        return models

    def _load_attackers(self):
        to_load = self._config["attackers"] or {}
        attackers = {}
        for atkr_id, atkr_desc in to_load.items():
            atkr_type = atkr_desc["type"]
            att_params = atkr_desc.get("params", {})
            if atkr_type == "ch":
                att_params = atkr_desc["params"]
                attackers[atkr_id] = CleverhansAttacker(atkr_id, att_params)
            elif atkr_type == "dummy":
                attackers[atkr_id] = DummyAttacker(atkr_id, att_params)
            else:
                raise Exception("Unsupported attack type {}".format(atkr_type))
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
                                     model=self.models.get(ex_desc["model"], None),
                                     attacker=self.attackers.get(ex_desc["attacker"], None),
                                     thresholding=ex_desc["thresholding"])
        return exps
