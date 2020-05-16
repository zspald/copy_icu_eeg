############################################################################
# Trains various deep learning models to detect seizure from EEG recordings
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
from evaluate import EEGEvaluator
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
            # Initialize list containers for training history and metrics
            history_list = [None for _ in range(len(patient_list))]
            metric_list = [None for _ in range(len(patient_list))]
            # Iterate over all patients for cross validation
            for ii in range(len(patient_list)):
                # Initialize generators for training, validation and testing data
                train_patients, validation_patients, test_patients = self.split_data_fix(0.9, ii)
                train_generator = EEGDataGenerator(train_patients, batch_size=1e4, shuffle=True)
                validation_generator = EEGDataGenerator(validation_patients, batch_size=1e4, shuffle=True)
                test_generator = EEGDataGenerator(test_patients, batch_size=1e4, shuffle=True)
                # Load and train the model
                model = EEGModel.convolutional_network(self.shape)
                history = model.fit_generator(generator=train_generator, validation_data=validation_generator,
                                              epochs=epochs, use_multiprocessing=True, verbose=verbose,
                                              workers=os.cpu_count())
                history_list[ii] = history
                # Save model and obtain predictions for test data
                if save:
                    model.save('ICU-EEG-CNN-%d.h5' % ii, save_format='h5')
                predict = model.predict_generator(test_generator, use_multiprocessing=True, verbose=verbose,
                                                  workers=os.cpu_count())
                predict = np.argmax(predict, axis=1)
                # Post-process the model predictions and obtain labels
                predict = EEGEvaluator.postprocess_outputs(predict, length=self.length)
                labels = test_generator.get_labels()
                # Compute evaluation metrics
                metrics = EEGEvaluator.evaluate_metrics(labels, predict)
                metric_list[ii] = metrics
            # Display results
            EEGEvaluator.training_curve_cv(history_list)
            EEGEvaluator.test_results_cv(metric_list)
        else:
            # Initialize generators for training, validation and testing data
            train_patients, validation_patients, test_patients = self.split_data(0.8, 0.1)
            train_generator = EEGDataGenerator(train_patients, batch_size=1e4, shuffle=True)
            validation_generator = EEGDataGenerator(validation_patients, batch_size=1e4, shuffle=True)
            test_generator = EEGDataGenerator(test_patients, batch_size=1e4, shuffle=True)
            # Load and train the model
            model = EEGModel.convolutional_network(self.shape)
            history = model.fit_generator(generator=train_generator, validation_data=validation_generator,
                                          epochs=epochs, use_multiprocessing=True, verbose=verbose, workers=os.cpu_count())
            # Save model and obtain predictions for test data
            if save:
                model.save('ICU-EEG-CNN.h5', save_format='h5')
            predict = model.predict_generator(test_generator, use_multiprocessing=True, verbose=verbose,
                                              workers=os.cpu_count())
            predict = np.argmax(predict, axis=1)
            # Post-process the model predictions and obtain labels
            predict = EEGEvaluator.postprocess_outputs(predict, length=self.length)
            labels = test_generator.get_labels()
            # Compute evaluation metrics
            metrics = EEGEvaluator.evaluate_metrics(labels, predict)
            # Display results
            EEGEvaluator.training_curve(history)
            EEGEvaluator.test_results(metrics)
        return

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
