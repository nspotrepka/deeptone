import numpy as np
import tensorflow as tf
from tensorflow import keras

class DeepTone:
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(64, 64)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

    def print_tensorflow_version():
        print("TensorFlow {}".format(tf.__version__))
