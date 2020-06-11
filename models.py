############################################################################
# Contains deep learning models for detecting seizures from EEG maps
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, ConvLSTM2D, CuDNNGRU, CuDNNLSTM, \
    concatenate, Dense, Dropout, Flatten, Input, MaxPool2D, BatchNormalization, \
    MaxPool3D, TimeDistributed
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import load_model
import tensorflow as tf


# A class that contains EEG models for detecting seizure
class EEGModel:

    # A simple neural network model
    # Inputs
    #   input_shape: shape of the input data
    # Outputs
    #   model: the ANN model for seizure detection
    @staticmethod
    def simple_network(input_shape):
        input_layer = Input(shape=input_shape)
        # Immediately flatten the image and pass through a single hidden layer
        flat1 = Flatten()(input_layer)
        fc1 = Dense(64, activation=tf.nn.relu)(flat1)
        drop1 = Dropout(0.2)(fc1)
        out = Dense(2, activation=tf.nn.softmax)(drop1)
        # Initialize and compile the model
        model = Model(inputs=input_layer, outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # A 2D convolutional neural network model that resembles Inception v1 approach
    # Inputs
    #   input_shape: shape of the input data
    # Outputs
    #   model: the CNN model for seizure detection
    @staticmethod
    def convolutional_network(input_shape):
        input_layer = Input(shape=input_shape)
        # First set of convolution/pooling layers (smaller receptive field)
        conv1a = Conv2D(8, kernel_size=(3, 3), activation=tf.nn.relu)(input_layer)
        conv2a = Conv2D(16, kernel_size=(3, 3), activation=tf.nn.relu)(conv1a)
        pool1a = MaxPool2D(pool_size=(2, 2))(conv2a)
        conv3a = Conv2D(32, kernel_size=(2, 2), activation=tf.nn.relu)(pool1a)
        conv4a = Conv2D(32, kernel_size=(2, 2), activation=tf.nn.relu)(conv3a)
        pool2a = MaxPool2D(pool_size=(2, 2))(conv4a)
        flat1 = Flatten()(pool2a)
        # Second set of convolution/pooling layers (larger receptive field)
        conv1b = Conv2D(10, kernel_size=(5, 5), strides=(2, 2), activation=tf.nn.relu)(input_layer)
        conv2b = Conv2D(15, kernel_size=(4, 4), activation=tf.nn.relu)(conv1b)
        conv3b = Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu)(conv2b)
        conv4b = Conv2D(32, kernel_size=(2, 2), activation=tf.nn.relu)(conv3b)
        pool1b = MaxPool2D(pool_size=(2, 2))(conv4b)
        flat2 = Flatten()(pool1b)
        # Concatenate flattened representations
        concat = concatenate([flat1, flat2])
        fc1 = Dense(64, activation=tf.nn.relu)(concat)
        drop1 = Dropout(0.2)(fc1)
        fc2 = Dense(32, activation=tf.nn.relu)(drop1)
        drop2 = Dropout(0.2)(fc2)
        out = Dense(2, activation=tf.nn.softmax)(drop2)
        # Initialize and compile the model
        model = Model(inputs=input_layer, outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # A 2D Conv-LSTM model for seizure detection
    # Inputs
    #   input_shape: shape of the input data
    # Outputs
    #   model: the Conv-LSTM model for seizure detection
    @staticmethod
    def conv_lstm_network(input_shape):
        input_layer = Input(input_shape)
        # Two layers of ConvLSTMs with batch normalization, followed by max pooling
        convlstm1 = ConvLSTM2D(32, kernel_size=(3, 3), return_sequences=True)(input_layer)
        norm1 = BatchNormalization()(convlstm1)
        convlstm2 = ConvLSTM2D(16, kernel_size=(3, 3), return_sequences=True)(norm1)
        norm2 = BatchNormalization()(convlstm2)
        pool1 = MaxPool3D(pool_size=(1, 2, 2))(norm2)
        # One layer of ConvLSTM with only last hidden state as output, followed by max pooling
        convlstm3 = ConvLSTM2D(8, kernel_size=(3, 3), return_sequences=False)(pool1)
        pool2 = MaxPool2D(pool_size=(2, 2))(convlstm3)
        flat = Flatten()(pool2)
        # Fully-connected layers after flattening
        fc1 = Dense(64, activation=tf.nn.relu)(flat)
        drop1 = Dropout(0.2)(fc1)
        fc2 = Dense(32, activation=tf.nn.relu)(drop1)
        drop2 = Dropout(0.2)(fc2)
        out = Dense(2, activation=tf.nn.softmax)(drop2)
        # Initialize and compile the model
        model = Model(inputs=input_layer, outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
