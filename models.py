############################################################################
# Contains deep learning models for detecting seizures from EEG maps
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, ConvLSTM2D, CuDNNGRU, CuDNNLSTM, \
    concatenate, Dense, Dropout, Flatten, Input, MaxPool2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf


# A class that contains EEG models for detecting seizure
class EEGModel:

    # A 2D convolutional neural network model that resembles Inception v1 approach
    # Inputs
    #   input_data: sample input that defines the size of the input layer
    # Outputs
    #   model: the model that
    @staticmethod
    def convolutional_network(input_data):
        input_layer = Input(shape=np.shape(input_data))
        conv1a = Conv2D(8, kernel_size=(3, 3), activation=tf.nn.relu)(input_layer)
        conv2a = Conv2D(16, kernel_size=(3, 3), activation=tf.nn.relu)(conv1a)
        pool1a = MaxPool2D(pool_size=(2, 2))(conv2a)
        conv3a = Conv2D(32, kernel_size=(2, 2), activation=tf.nn.relu)(pool1a)
        conv4a = Conv2D(32, kernel_size=(2, 2), activation=tf.nn.relu)(conv3a)
        pool2a = MaxPool2D(pool_size=(2, 2))(conv4a)
        flat1 = Flatten()(pool2a)
        conv1b = Conv2D(10, kernel_size=(5, 5), strides=(2, 2), activation=tf.nn.relu)(input_layer)
        conv2b = Conv2D(15, kernel_size=(4, 4), activation=tf.nn.relu)(conv1b)
        conv3b = Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu)(conv2b)
        conv4b = Conv2D(32, kernel_size=(2, 2), activation=tf.nn.relu)(conv3b)
        pool1b = MaxPool2D(pool_size=(2, 2))(conv4b)
        flat2 = Flatten(pool1b)
        concat = concatenate([flat1, flat2])
        fc1 = Dense(64, activation=tf.nn.relu)(concat)
        drop1 = Dropout(0.2)(fc1)
        fc2 = Dense(32, activation=tf.nn.relu)(drop1)
        drop2 = Dropout(0.2)(fc2)
        out = Dense(2, activation=tf.nn.softmax)(drop2)
        model = Model(inputs=input_layer, outputs=out)
        model.compile(loss=CategoricalCrossentropy, optimizer=Adam, metrics=['accuracy'])
        return model
