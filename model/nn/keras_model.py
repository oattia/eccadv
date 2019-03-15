from model.nn.neural_net import NeuralNetModel, Libraries

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Conv2D


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


KERAS_MODEL_CATALOG = {
    "ch_cnn": ch_cnn_model
}

KERAS_LOSS_CATALOG = {
    "cross_entropy": "categorical_crossentropy"
}


class KerasNnModel(NeuralNetModel):
    def __init__(self, name, config):
        super(KerasNnModel).__init__(name, config)
        self.model_choice = self.config["catalog_id"]

    def initialize(self, input_shape, n_classes, output_dim):
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.output_size = output_dim
        self.network_model = KERAS_MODEL_CATALOG[self.model_choice](input_shape, output_dim, self.config)
        if self.n_classes == self.output_size:
            self.network_model.add(Activation("softmax"))
        else:
            self.network_model.add(Activation("sigmoid"))

        # To be able to call the model in the custom loss, we need to call it once
        # before, see https://github.com/tensorflow/tensorflow/issues/23769
        self.network_model(self.network_model.input)
        self.network_model.compile(
            optimizer=keras.optimizers.Adam(self.config["lr"]),
            loss=KERAS_LOSS_CATALOG[self.config["loss"]]
        )

    def train_batch(self, features, labels):
        self.network_model.train_on_batch(features, labels)

    def predict(self, features):
        predicted = self.network_model.predict_on_batch(features)
        if self.n_classes == self.output_size:
            return (predicted == predicted.max()).astype(int)
        else:
            threshold = self.config

    def save_to(self, path):
        self.network_model.save(path / self.name)

    def load_from(self, path):
        self.network_model = load_model(path / self.name)

    def context(self):
        return {
            "network": self.network_model,
            "library": Libraries.KERAS
        }
