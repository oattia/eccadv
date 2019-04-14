from model.nn.neural_net import NeuralNetModel, Libraries

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Assignment rather than import because direct import from within Keras
# doesn't work in tf 1.8
Sequential = keras.models.Sequential
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Activation = keras.layers.Activation
Flatten = keras.layers.Flatten
MaxPooling2D = keras.layers.MaxPooling2D
Dropout = keras.layers.Dropout


def ch_cnn_model(input_shape, output_size, config):
    """
    Defines a CNN model using Keras sequential model, from cleverhans.
    """
    model = Sequential()
    nb_filters = config["nb_filters"]
    layers = [
                Conv2D(filters=nb_filters, kernel_size=(8, 8), strides=(2, 2), padding="same", input_shape=input_shape),
                Activation("relu"),
                Conv2D(filters=nb_filters * 2, kernel_size=(6, 6), strides=(2, 2), padding="valid"),
                Activation("relu"),
                Conv2D(filters=nb_filters * 2, kernel_size=(5, 5), strides=(1, 1), padding="valid"),
                Activation("relu"),
                Flatten(),
                Dense(output_size)
    ]

    for layer in layers:
        model.add(layer)

    return model


def art_cnn_model(input_shape, output_size, config):
    import keras.models as km
    import keras.layers as kl
    nb_filters = config["nb_filters"]
    model = km.Sequential()
    model.add(kl.Conv2D(nb_filters, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(kl.Conv2D(nb_filters*2, (3, 3), activation='relu'))
    model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    model.add(kl.Dropout(0.25))
    model.add(kl.Flatten())
    model.add(kl.Dense(nb_filters*4, activation='relu'))
    model.add(kl.Dropout(0.5))
    model.add(kl.Dense(output_size))
    return model


KERAS_MODEL_CATALOG = {
    "ch_cnn": ch_cnn_model,
    "art_cnn": art_cnn_model
}


def mae(labels, preds, from_logits=True):
    return keras.losses.mean_absolute_error(labels, preds)


KERAS_LOSS_CATALOG = {
    "cross_entropy": keras.losses.categorical_crossentropy,
    "binary_entropy": keras.losses.binary_crossentropy,
    "mse": keras.losses.mean_squared_error,
    "sig_entropy": tf.losses.sigmoid_cross_entropy,
    "mae": keras.losses.mean_absolute_error  # mae  # keras.losses.mean_absolute_error
}


def get_adversarial_acc_metric(model, attacker, attacker_params):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = attacker.generate(model.input, **attacker_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)

    return adv_acc


def get_adversarial_loss(model, loss, attacker, attacker_params):
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = loss(y, preds)

        # Generate adversarial examples
        x_adv = attacker.generate(model.input, **attacker_params)

        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = loss(y, preds_adv)

        return 0.5 * cross_ent + 0.5 * cross_ent_adv
    return adv_loss


# For ART to work
import keras.backend as k
setattr(k, "mae", KERAS_LOSS_CATALOG["mae"])


class KerasNnModel(NeuralNetModel):
    def __init__(self, name, config):
        super(KerasNnModel, self).__init__(name, config)

    def _build_model(self):
        model_choice = self.config["structure"]
        self.network_model = KERAS_MODEL_CATALOG[model_choice](self.input_shape, self.output_size, self.config)

        if self.n_classes == self.output_size:
            self.network_model.add(Activation("softmax"))
        else:
            self.network_model.add(Activation("sigmoid"))

        self.network_model.summary()

        # To be able to call the model in the custom loss, we need to call it once
        # before, see https://github.com/tensorflow/tensorflow/issues/23769
        self.network_model(self.network_model.input)

    def compile(self, atkr=None):
        loss = KERAS_LOSS_CATALOG[self.config["loss"]]
        metrics = ["accuracy"]

        if atkr:
            attacker = atkr.attack
            attacker_params = atkr.get_params()
            loss = get_adversarial_loss(self.network_model, loss, attacker, attacker_params)
            metrics.append(get_adversarial_acc_metric(self.network_model, attacker, attacker_params))

        self.network_model.compile(
            optimizer=keras.optimizers.Adam(self.config["lr"]),
            loss=loss,
            metrics=metrics
        )

        k.set_learning_phase(1)

    def train_batch(self, features, labels):
        return self.network_model.train_on_batch(features, labels)

    def predict(self, features):
        return self.network_model.predict_on_batch(features)

    def loss(self, features, labels):
        loss = KERAS_LOSS_CATALOG[self.config["loss"]]
        preds = self.predict(features).astype(float)
        return k.get_session().run(loss(labels.astype(float), preds))

    def save_to(self, path):
        self.network_model.save(path.as_posix())

    def load_from(self, path):
        self.network_model = keras.models.load_model(path)
        self.network_model.summary()

    def context(self):
        return {
            "network": self.network_model,
            "library": Libraries.KERAS.name
        }
