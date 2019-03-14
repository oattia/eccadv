from model.nn.neural_net import NeuralNetModel


class Attacks:
    FGSM    = 0
    BIM     = 1
    MIM     = 2


class Attacker:
    """
    Abstract Class to hide the details of the adversarial attack library used:
        (Cleverhans, Foolbox, Advertorch, Adversarial-Robustness-Toolbox).
    """
    def __init__(self, name, nn_model, attack_params):
        self.name = name
        self.nn_model: NeuralNetModel = nn_model
        self.attack_params = attack_params

    def perturb(self, samples):
        """"
        Perturb the samples according to the adversarial attack.
        """
        raise NotImplementedError


class DummyAttacker(Attacker):
    """
    Dummy attacker that does not perturb the given samples.
    """
    def __init__(self, name, nn_model=None, attack_params=None):
        super(DummyAttacker, self).__init__(name, None, None)

    def perturb(self, samples):
        return samples
