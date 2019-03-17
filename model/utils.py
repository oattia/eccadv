"""
Utilities used by multiple parts of the model module.
"""
import numpy as np
from tensorflow import keras
import tensorflow as tf

tf_session_config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
tf_session = tf.Session(config=tf_session_config)

keras.backend.set_session(tf_session)
keras.backend.set_image_data_format("channels_last")


def threshold(float_arr, **kwargs):
    method = kwargs["method"]
    if method == "max_k":
        k = kwargs.get("k", 1)
        ones = float_arr.argsort()[-k:]
        zeros = np.zeros_like(float_arr, dtype=int)
        zeros[ones] = 1
        return zeros
    elif method == "threshold":
        t = kwargs.get("t", 0.5)
        return (float_arr >= t).astype(int)
    else:
        raise Exception("Unsupported thresholding method.")
