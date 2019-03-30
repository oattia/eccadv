from model.attacker.attacker import Attacker, Attacks
from model.nn.neural_net import Libraries
from model.utils import tf_session

from cleverhans.attacks import *
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf


class CleverhansAttacker(Attacker):
    """
    Attacker based on the cleverhans library.
    """
    def __init__(self, name, params):
        super(CleverhansAttacker, self).__init__(name, params)
        self.attack = None

    def initialize(self, nn_model):
        super(CleverhansAttacker, self).initialize(nn_model)

        # model context for attack, it must have: library, network
        model_context = self.nn_model.context()

        # model wrapping for cleverhans
        library = model_context["library"]
        network = model_context["network"]
        if library == Libraries.KERAS.name:
            cleverhans_model = KerasModelWrapper(network)
        elif library == Libraries.TORCH.name:
            cleverhans_model = CallableModelWrapper(convert_pytorch_model_to_tf(network), output_layer="logits")
        elif library == Libraries.TF.name:
            # assume it will be a tf-cleverhans model, best effort trial, will fail next if not
            cleverhans_model = network
        else:
            raise Exception("Unsupported library '{}' for Cleverhans.".format(library))

        # attack choice and initialization
        attack_method = self.attack_params["method"]
        if attack_method == Attacks.FGSM.name:
            self.attack = FastGradientMethod(cleverhans_model, sess=tf_session)
        elif attack_method == Attacks.BIM.name:
            self.attack = BasicIterativeMethod(cleverhans_model, sess=tf_session)
        elif attack_method == Attacks.MIM.name:
            self.attack = MomentumIterativeMethod(cleverhans_model, sess=tf_session)
        else:
            raise Exception("Unsupported attack '{}' for Cleverhans.".format(attack_method))

    def perturb(self, samples):
        # perturb the features according to the attack params
        params = {param: self.attack_params[param] for param in self.attack.feedable_kwargs
                  if param in self.attack_params}
        return self.attack.generate_np(samples, **params)
