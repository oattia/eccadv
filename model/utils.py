"""
Utilities used by multiple parts of the model module.
"""

from tensorflow import keras
import tensorflow as tf

tf_session_config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
tf_session = tf.Session(config=tf_session_config)

keras.backend.set_session(tf_session)
keras.backend.set_image_data_format("channels_last")
