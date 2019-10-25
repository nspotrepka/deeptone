import tensorflow as tf
from tensorflow import keras
import numpy as np

class DeepTone:
    self.model = None

    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(64, 64)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
