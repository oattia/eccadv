from enum import Enum


class Attacks(Enum):
    FGSM = 0
    BIM  = 1
    MIM  = 2


class Attacker:
    """
    Abstract Class to hide the details of the adversarial attack library used:
        (Cleverhans, Foolbox, Advertorch, Adversarial-Robustness-Toolbox).
    """
    def __init__(self, name, attack_params):
        self.name = name
        self.nn_model = None
        self.attack_params = attack_params

    def initialize(self, nn_model):
        self.nn_model = nn_model

    def get_params(self):
        return self.attack_params

    def perturb(self, samples):
        """"
        Perturb the samples according to the adversarial attack.
        """
        raise NotImplementedError


class DummyAttacker(Attacker):
    """
    Dummy attacker that does not perturb the given samples.
    """
    def __init__(self, name, attack_params={}):
        super(DummyAttacker, self).__init__(name, {})

    def perturb(self, samples):
        return samples
