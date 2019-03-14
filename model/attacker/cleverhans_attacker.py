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
    def __init__(self, name,  nn_model, attack_params):
        super(CleverhansAttacker, self).__init__(name, nn_model, attack_params)

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
            # assume it will be tf-cleverhans model, best effort trial, will fail next if not
            cleverhans_model = network
        else:
            raise Exception(f"Unsupported library '{library}' for Cleverhans.")

        # attack choice and initialization
        attack_type = self.attack_params["type"]
        if attack_type == Attacks.FGSM.name:
            self.attack = FastGradientMethod(cleverhans_model, sess=tf_session)
        elif attack_type == Attacks.BIM.name:
            self.attack = BasicIterativeMethod(cleverhans_model, sess=tf_session)
        elif attack_type == Attacks.MIM.name:
            self.attack = MomentumIterativeMethod(cleverhans_model, sess=tf_session)
        else:
            raise Exception(f"Unsupported attack '{attack_type}' for Cleverhans.")

    def perturb(self, samples):
        # perturb the features according to the attack params
        return self.attack.generate_np(samples, **self.attack_params)
