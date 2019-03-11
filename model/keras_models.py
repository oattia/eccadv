from model.model import NeuralNetModel

from tensorflow import keras

Sequential = keras.models.Sequential
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Activation = keras.layers.Activation
Flatten = keras.layers.Flatten
KerasModel = keras.models.Model


def conv_2d(filters, kernel_shape, strides, padding, input_shape=None):
    """
      Defines the right convolutional layer according to the
      version of Keras that is installed.
      :param filters: (required integer) the dimensionality of the output
                      space (i.e. the number output of filters in the
                      convolution)
      :param kernel_shape: (required tuple or list of 2 integers) specifies
                           the kernel shape of the convolution
      :param strides: (required tuple or list of 2 integers) specifies
                           the strides of the convolution along the width and
                           height.
      :param padding: (required string) can be either 'valid' (no padding around
                      input or feature map) or 'same' (pad to ensure that the
                      output feature map size is identical to the layer input)
      :param input_shape: (optional) give input shape if this is the first
                          layer of the model
      :return: the Keras layer
      """
    if input_shape is not None:
        return Conv2D(filters=filters, kernel_size=kernel_shape,
                      strides=strides, padding=padding,
                      input_shape=input_shape)
    else:
        return Conv2D(filters=filters, kernel_size=kernel_shape,
                      strides=strides, padding=padding)


def cnn_model(logits=False, input_ph=None, img_rows=28, img_cols=28,
              channels=1, nb_filters=64, nb_classes=10):
  """
  Defines a CNN model using Keras sequential model
  :param logits: If set to False, returns a Keras model, otherwise will also
                  return logits tensor
  :param input_ph: The TensorFlow tensor for the input
                  (needed if returning logits)
                  ("ph" stands for placeholder but it need not actually be a
                  placeholder)
  :param img_rows: number of row in the image
  :param img_cols: number of columns in the image
  :param channels: number of color channels (e.g., 1 for MNIST)
  :param nb_filters: number of convolutional filters per layer
  :param nb_classes: the number of output classes
  :return:
  """

  # 1 epoch, on 6 cores, on 60k mnist, takes < 4 minutes
  model = Sequential()

  # Define the layers successively (convolution layers are version dependent)
  if keras.backend.image_data_format() == 'channels_first':
    input_shape = (channels, img_rows, img_cols)
  else:
    assert keras.backend.image_data_format() == 'channels_last'
    input_shape = (img_rows, img_cols, channels)

  layers = [conv_2d(nb_filters, (8, 8), (2, 2), "same",
                    input_shape=input_shape),
            Activation('relu'),
            conv_2d((nb_filters * 2), (6, 6), (2, 2), "valid"),
            Activation('relu'),
            conv_2d((nb_filters * 2), (5, 5), (1, 1), "valid"),
            Activation('relu'),
            Flatten(),
            Dense(nb_classes)]

  for layer in layers:
    model.add(layer)

  if logits:
    logits_tensor = model(input_ph)
  model.add(Activation('softmax'))

  if logits:
    return model, logits_tensor
  else:
    return model


class KerasNnModel(NeuralNetModel):
    def __init__(self, name, config):
        super(KerasNnModel).__init__(name, config)
        self.model = cnn_model()

    def initialize(self):
        raise NotImplementedError

    def train_batch(self, features, labels):
        raise NotImplementedError

    def predict_batch(self, features):
        raise NotImplementedError

    def save_to(self, path):
        raise NotImplementedError

    def load_from(self, path):
        raise NotImplementedError
