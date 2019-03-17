from model.nn.neural_net import NeuralNetModel, Libraries

import tensorflow as tf
from tensorflow import keras

# Assignment rather than import because direct import from within Keras
# doesn't work in tf 1.8
Sequential = keras.models.Sequential
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Activation = keras.layers.Activation
Flatten = keras.layers.Flatten


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


def hamming_loss(y_true, y_pred):
    tf.convert_to_tensor_or_sparse_tensor()



KERAS_MODEL_CATALOG = {
    "ch_cnn": ch_cnn_model
}

KERAS_LOSS_CATALOG = {
    "cross_entropy": "categorical_crossentropy",
    "binary_entropy": "binary_crossentropy",
    "mse": "mean_squared_error",
    "hamming_loss": hamming_loss,
    "sig_entropy": tf.losses.sigmoid_cross_entropy,
    "edit_distance": tf.edit_distance
}


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

        self.network_model.compile(
            optimizer=keras.optimizers.Adam(self.config["lr"]),
            loss=KERAS_LOSS_CATALOG[self.config["loss"]],
            metrics=["accuracy"]
        )

    def train_batch(self, features, labels):
        return self.network_model.train_on_batch(features, labels)

    def predict(self, features):
        return self.network_model.predict_on_batch(features)

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
