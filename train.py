############################################################################
# Trains various deep learning models to detect seizure from EEG recordings
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
from generator import EEGDataGenerator
from models import EEGModel
from tensorflow.keras.backend import set_image_data_format
import numpy as np
import os
import random

# List of all patient IDs
patient_list = ['RID0061', 'RID0062', 'RID0063', 'RID0064']


# A class that runs a 2D convolutional neural network
class EEGLearner:

    # The constructor for the EEGLearner class
    # Fields
    #   length: the length of each EEG sample
    #   model: the model used for the classifier
    #   shape: the shape of the EEG data
    #   train: the method used for training the EEG data
    def __init__(self):
        # Reset the image ordering so that channels occupy the first dimension
        set_image_data_format('channels_first')
        # Initialize fields
        self.train = None
        self.model = None
        # Check the shape of the EEG data
        with open('data/%s_data.h5' % patient_list[0], 'r') as file:
            self.shape = np.shape(file['maps'][0])
            self.length = file['labels'][0, 2] - file['labels'][0, 1]

    # Trains a CNN model based on the EEG data
    # Inputs
    #   epochs: number of epochs during the training process
    #   cross_val: whether to use cross validation on the EEG dataset
    #   save: whether to save the trained model
    #   verbose: level of verbosity of the training process
    #   visualize: whether to visualize the train/validation/test results
    def train_cnn(self, epochs, cross_val=True, save=True, verbose=1, visualize=True):
        self.train = 'conv'
        if cross_val:
            # Iterate over all patients for cross validation
            for ii in range(len(patient_list)):
                # Initialize generators for training, validation and testing data
                train_patients, validation_patients, test_patients = self.split_data_fix(0.9, ii)
                train_generator = EEGDataGenerator(train_patients, batch_size=1e4, shuffle=True)
                validation_generator = EEGDataGenerator(validation_patients, batch_size=1e4, shuffle=True)
                test_generator = EEGDataGenerator(test_patients, batch_size=1e4, shuffle=True)
                # Load and train the model
                model = EEGModel.convolutional_network(self.shape)
                model.fit_generator(generator=train_generator, validation_data=validation_generator,
                                    epochs=epochs, use_multiprocessing=True, workers=os.cpu_count())
                # Save model and obtain predictions for test data
                model.save('ICU-EEG-CNN-%d.h5' % ii, save_format='h5')
                predict = model.predict_generator(test_generator, use_multiprocessing=True, workers=os.cpu_count())

        else:
            # Initialize generators for training, validation and testing data
            train_patients, validation_patients, test_patients = self.split_data(0.8, 0.1)
            train_generator = EEGDataGenerator(train_patients, batch_size=1e4, shuffle=True)
            validation_generator = EEGDataGenerator(validation_patients, batch_size=1e4, shuffle=True)
            test_generator = EEGDataGenerator(test_patients, batch_size=1e4, shuffle=True)
            # Load and train the model
            model = EEGModel.convolutional_network(self.shape)
            model.fit_generator(generator=train_generator, validation_data=validation_generator,
                                epochs=epochs, use_multiprocessing=True, workers=os.cpu_count())
            # Save model and obtain predictions for test data
            model.save('ICU-EEG-CNN.h5', save_format='h5')
            predict = model.predict_generator(test_generator, use_multiprocessing=True, workers=os.cpu_count())

    # Computes evaluation metrics for given predictions and labels
    # Inputs
    #   labels: a list of seizure annotations given as a 1D numpy array
    #   predictions: a list of seizure predictions given as a 1D numpy array
    # Outputs
    #   accuracy: accuracy of the outputs
    #   sensitivity_pos: sensitivity of the positive outputs (seizures)
    #   precision_pos: precision of the positive outputs
    #   sensitivity_neg: sensitivity of the negative outputs (non-seizures)
    #   precision_neg: precision of the negative outputs
    @staticmethod
    def evaluate_metrics(labels, predictions):
        assert(np.size(labels, axis=0) == np.size(predictions, axis=0))
        # Compute the accuracy of the outputs
        seizure_correct = np.sum(np.multiply(labels, predictions))
        normal_correct = np.sum(np.multiply(1 - labels, 1 - predictions))
        accuracy = (seizure_correct + normal_correct) / len(labels)
        # Compute the sensitivity of the positive outputs
        if np.sum(labels) > 0:
            sensitivity_pos = seizure_correct / np.sum(labels)
        else:
            print("Sensitivity is not defined for positive outputs")
            sensitivity_pos = np.nan
        # Compute the precision of the positive outputs
        if np.sum(predictions) > 0:
            precision_pos = seizure_correct / np.sum(predictions)
        else:
            print("Precision is not defined for positive outputs")
            precision_pos = np.nan
        # Compute the sensitivity of the negative outputs
        if np.sum(1 - labels) > 0:
            sensitivity_neg = normal_correct / np.sum(1 - labels)
        else:
            print("Sensitivity is not defined for negative outputs")
            sensitivity_neg = np.nan
        # Compute the precision of the negative outputs
        if np.sum(1 - predictions) > 0:
            precision_neg = normal_correct / np.sum(1 - predictions)
        else:
            print("Precision is not defined for negative outputs")
            precision_neg = np.nan
        return accuracy, sensitivity_pos, precision_pos, sensitivity_neg, precision_neg

    # Divides training, validation and testing data
    # Inputs
    #   train_split: proportion of training data
    #   validation_split: proportion of validation data
    # Outputs
    #   train_patients: list of patient IDs to be used for training
    #   validation_patients: list of patient IDs to be used for validation
    #   test_patients: list of patient IDs to be used for testing
    @staticmethod
    def split_data(train_split, validation_split):
        # Reject any invalid input
        if train_split + validation_split > 1:
            print("Please provide an appropriate split input.")
            return
        # Shuffle the list of patient IDs
        new_patient_list = random.shuffle(patient_list)
        num = len(patient_list)
        # Obtain the list of patients used for training, validating and testing
        train_patients = new_patient_list[:int(train_split * num)]
        validation_patients = new_patient_list[int(train_split * num):int((train_split + validation_split) * num)]
        test_patients = new_patient_list[int((train_split + validation_split) * num):]
        return train_patients, validation_patients, test_patients

    # Divides training and validation data with a specified test patient
    # Inputs
    #   train_split: proportion of training data
    #   idx: index of the testing patient
    # Outputs
    #   train_patients: list of patient IDs to be used for training
    #   validation_patients: list of patient IDs to be used for validation
    #   test_patients: list of patient IDs to be used for testing
    @staticmethod
    def split_data_fix(train_split, idx):
        # Reject any invalid input
        if train_split > 1:
            print("Please provide an appropriate split input.")
            return
        # Shuffle the list of patient IDs except for the designated test patient
        new_patient_list = [patient for patient in patient_list if patient != patient_list[idx]]
        new_patient_list = random.shuffle(new_patient_list)
        num = len(patient_list)
        # Obtain the list of patients used for training, validating and testing
        train_patients = new_patient_list[:int(train_split * num)]
        validation_patients = new_patient_list[int(train_split * num):]
        test_patients = [patient_list[idx]]
        return train_patients, validation_patients, test_patients
