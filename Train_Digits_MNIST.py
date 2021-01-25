import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from convautoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np


class ConvMNIST:
    @staticmethod
    def build(image_size=784):
        n_input = image_size  # number of features
        n_hidden_1 = 300
        n_hidden_2 = 100
        n_hidden_3 = 100
        n_hidden_4 = 200
        num_digits = 10

        Inp = Input(shape=(n_input,))
        x = Dense(n_hidden_1, activation='relu', name="Hidden_Layer_1")(Inp)
        x = Dropout(0.3)(x)
        x = Dense(n_hidden_2, activation='relu', name="Hidden_Layer_2")(x)
        x = Dropout(0.3)(x)
        x = Dense(n_hidden_3, activation='relu', name="Hidden_Layer_3")(x)
        x = Dropout(0.3)(x)
        x = Dense(n_hidden_4, activation='relu', name="Hidden_Layer_4")(x)
        output = Dense(num_digits, activation='softmax', name="Output_Layer")(x)

        model = Model(Inp, output)

        return model

