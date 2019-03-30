from model.attacker.attacker import Attacker, Attacks
from model.nn.neural_net import Libraries

from art.classifiers import KerasClassifier, PyTorchClassifier, TFClassifier
from art.attacks.fast_gradient import FastGradientMethod


class ArtAttacker(Attacker):
    """
    Attacker based on the adversarial-robustness-toolbox library.
    """
    def __init__(self, name, params):
        super(ArtAttacker, self).__init__(name, params)
        self.attack = None

    def initialize(self, nn_model):
        super(ArtAttacker, self).initialize(nn_model)

        # model context for attack, it must have: library, network
        model_context = self.nn_model.context()

        # model wrapping for art
        library = model_context["library"]
        network = model_context["network"]
        if library == Libraries.KERAS.name:
            art_model = KerasClassifier((0.0, 1.0), network,
                                        use_logits=False,
                                        # channel_index=3,
                                        # custom_activation=False
                                        )
        elif library == Libraries.TORCH.name:
            art_model = None
            # art_model = PyTorchClassifier(clip_values, network,
            #                               loss, optimizer,
            #                               input_shape, nb_classes, channel_index=1)
        elif library == Libraries.TF.name:
            art_model = None
            # art_model = TFClassifier(clip_values, input_ph, logits,
            #                          output_ph=None, train=None, loss=None,
            #                          learning=None, sess=None,
            #                          channel_index=3, defences=None)
        else:
            raise Exception("Unsupported library '{}' for adversarial-robustness-toolbox..".format(library))

        # attack choice and initialization
        attack_method = self.attack_params["method"]
        if attack_method == Attacks.FGSM.name:
            self.attack = FastGradientMethod(art_model)
        else:
            raise Exception("Unsupported attack '{}' for adversarial-robustness-toolbox.".format(attack_method))

    def perturb(self, samples):
        # perturb the features according to the attack params
        params = {param: self.attack_params[param] for param in self.attack.attack_params
                  if param in self.attack_params}
        return self.attack.generate(x=samples, **params)
