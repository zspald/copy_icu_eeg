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
import h5py
import numpy as np
import os
import random
from sklearn.model_selection import KFold
import pickle

# A class that runs a 2D convolutional neural network
class EEGLearner:

    # The constructor for the EEGLearner class
    # Attributes
    #   length: the length of each EEG sample
    #   loss_weights: the weights for non-seizure/seizure loss
    #   patient_list: list of all patient IDs
    #   name: the name of the training model
    #   shape: the shape of the EEG data
    def __init__(self, patient_list, sz_list=None, nsz_list=None):
        # Reset the image ordering so that channels occupy the first dimension
        set_image_data_format('channels_first')
        # Initialize fields
        self.name = ''
        self.patient_list = patient_list
        self.sz_list = sz_list
        self.nsz_list = nsz_list
        self.loss_weights = {0: 1.0, 1: 2.0}
        # Check the shape of the EEG data
        file = h5py.File('data/%s_data_wt.h5' % patient_list[0], 'r')
        self.shape = file['maps'][0].shape
        self.length = file['labels'][0, 2] - file['labels'][0, 1]
        file.close()

    # Trains a CNN model based on the EEG dataset
    # Inputs
    #   epochs: number of epochs during the training process
    #   batch_size: number of samples in each batch
    #   control: the maximum amount of time allowed for control samples, in seconds
    #   cross_val: whether to use cross validation on the EEG dataset
    #   save: whether to save the trained model
    #   verbose: level of verbosity of the training process
    #   visualize: whether to visualize the train/validation/test results
    # Outputs
    #   model: the trained CNN model
    def train_cnn(self, epochs, batch_size=25, control=None, cross_val=False, k_fold=False, save=True, verbose=1, visualize=True):
        model = EEGModel.convolutional_network(self.shape)
        if k_fold:
            model = self.train_model_kfold_cv(epochs, model, 'conv', folds=5, batch_size=batch_size, control=control, save=save, use_seq=False, 
                                              verbose=verbose, visualize=visualize)
        else:
            model = self.train_model(epochs, model, 'conv', batch_size=batch_size, control=control, cross_val=cross_val,
                                     save=save, use_seq=False, verbose=verbose, visualize=visualize)
        return model

    # Trains a convolutional GRU model based on the EEG dataset
    # Inputs
    #   epochs: number of epochs during the training process
    #   batch_size: number of samples in each batch
    #   control: the maximum amount of time allowed for control samples, in seconds
    #   cross_val: whether to use cross validation on the EEG dataset
    #   save: whether to save the trained model
    #   seq_len: length of the input sequence, in seconds
    #   verbose: level of verbosity of the training process
    #   visualize: whether to visualize the train/validation/test results
    # Outputs
    #   model: the trained convolutional GRU model
    def train_convolutional_gru(self, epochs, batch_size=10, control=None, cross_val=False, save=True, seq_len=20,
                                verbose=1, visualize=True):
        model = EEGModel.convolutional_gru_network((seq_len,) + self.shape)
        model = self.train_model(epochs, model, 'cnn-gru', batch_size=batch_size, control=control, cross_val=cross_val,
                                 save=save, seq_len=seq_len, use_seq=True, verbose=verbose, visualize=visualize)
        return model

    # Trains a ConvLSTM model based on the EEG dataset
    # Inputs
    #   epochs: number of epochs during the training process
    #   batch_size: number of samples in each batch
    #   control: the maximum amount of time allowed for control samples, in seconds
    #   cross_val: whether to use cross validation on the EEG dataset
    #   save: whether to save the trained model
    #   seq_len: length of the input sequence, in seconds
    #   verbose: level of verbosity of the training process
    #   visualize: whether to visualize the train/validation/test results
    # Outputs
    #   model: the trained Conv-LSTM model
    def train_conv_lstm(self, epochs, batch_size=10, control=None, cross_val=False, save=True, seq_len=20,
                        verbose=1, visualize=True):
        model = EEGModel.conv_lstm_network((seq_len,) + self.shape)
        model = self.train_model(epochs, model, 'conv-lstm', batch_size=batch_size, control=control, cross_val=cross_val
                                 , save=save, seq_len=seq_len, use_seq=True, verbose=verbose, visualize=visualize)
        return model

    # Trains a deep neural network model based on the EEG data
    # Inputs
    #   epochs: number of epochs during the training process
    #   batch_size: number of samples in each batch
    #   control: the maximum amount of time allowed for control samples, in seconds
    #   cross_val: whether to use cross validation on the EEG dataset
    #   save: whether to save the trained model
    #   seq_len: length of the input sequence, in seconds
    #   use_seq: whether to use a sequence-based model
    #   verbose: level of verbosity of the training process
    #   visualize: whether to visualize the train/validation/test results
    # Outputs
    #   model: the trained model
    def train_model(self, epochs, model, name, batch_size=25, control=None, cross_val=False, save=True,
                    seq_len=20, use_seq=False, verbose=1, visualize=True):
        # Save the name of the training model
        self.name = name
        if cross_val:
            # Initialize list containers for training history and metrics
            history_list = [None for _ in range(len(self.patient_list))]
            metric_list = [None for _ in range(len(self.patient_list))]
            # Iterate over all patients for cross validation (leave one out)
            for ii in range(len(self.patient_list)):
                print('Iteration: {ii}')
                # Initialize generators for training, validation and testing data
                train_patients, valid_patients, test_patients = self.split_data_fix(self.patient_list, 0.9, ii)
                train_generator = EEGDataGenerator(train_patients, batch_size=batch_size, control=control,
                                                   sample_len=self.length, seq_len=seq_len, use_seq=use_seq)
                validation_generator = EEGDataGenerator(valid_patients, batch_size=batch_size, control=control,
                                                        sample_len=self.length, seq_len=seq_len, shuffle=False,
                                                        use_seq=use_seq)
                test_generator = EEGDataGenerator(test_patients, batch_size=batch_size, sample_len=self.length,
                                                  seq_len=seq_len, shuffle=False, use_seq=use_seq)
                # Load and train the model
                history = model.fit_generator(generator=train_generator, validation_data=validation_generator,
                                              class_weight=self.loss_weights, epochs=epochs, shuffle=True,
                                              verbose=verbose)
                history_list[ii] = history
                # Save model and obtain predictions for test data
                if save:
                    model.save('ICU-EEG-%s-%d.h5' % (name, ii), save_format='h5')
                predict = model.predict_generator(test_generator, verbose=0)
                predict = np.argmax(predict, axis=1)
                # Obtain annotations from the test generator
                labels = test_generator.get_annots()
                # Post-process the model predictions and obtain labels
                predict = EEGEvaluator.postprocess_outputs(predict, length=self.length)
                # Compute evaluation metrics for post-processed outputs
                metrics_postprocess = EEGEvaluator.evaluate_metrics(labels[:, 0], predict)
                metric_list[ii] = metrics_postprocess
            # Display results
            if visualize:
                EEGEvaluator.training_curve_cv(history_list)
                EEGEvaluator.test_results_cv(metric_list)
        else:
            # Initialize generators for training, validation and testing data
            # test = ["RID0062", "RID252_68561f5b", "CNT685", "CNT688"]
            test = ["ICUDataRedux_0062", "ICUDataRedux_0085", "CNT685", "CNT688"]
            train_patients, validation_patients, test_patients = self.split_data_test(self.patient_list, 0.9, test)
            print('Training Data: ', train_patients)
            print('Validation Data: ', validation_patients)
            print('Test Data: ', test_patients)
            train_generator = EEGDataGenerator(train_patients, batch_size=batch_size, control=control,
                                               sample_len=self.length, seq_len=seq_len, use_seq=use_seq)
            print('Number of training data: ', batch_size * train_generator.length)
            train_labels = train_generator.get_annots()
            print('Proportion of seizures: ', np.sum(train_labels[:, 0], axis=0) / np.size(train_labels, axis=0))
            validation_generator = EEGDataGenerator(validation_patients, batch_size=batch_size, control=control,
                                                    sample_len=self.length, seq_len=seq_len, use_seq=use_seq)
            print('Number of validation data: ', batch_size * validation_generator.length)
            test_generator = EEGDataGenerator(test_patients, batch_size=batch_size, sample_len=self.length,
                                              seq_len=seq_len, shuffle=False, use_seq=use_seq)
            print('Number of test data: ', batch_size * test_generator.length)
            # Load and train the model
            history = model.fit_generator(generator=train_generator, validation_data=validation_generator,
                                          class_weight=self.loss_weights, epochs=epochs, shuffle=True,
                                          verbose=verbose)
            # Save model and obtain predictions for test data
            if save:
                model.save('ICU-EEG-%s-%d(7).h5' % (name, epochs), save_format='h5')
            predict = model.predict_generator(test_generator, verbose=0)
            predict = np.argmax(predict, axis=1)
            # Obtain annotations from the test data generator
            labels = test_generator.get_annots()
            print('Predictions/Labels Shape: ', np.shape(predict), np.shape(labels[:, 0]))
            # Compute evaluation metrics for raw outputs
            metrics_raw = EEGEvaluator.evaluate_metrics(labels[:, 0], predict)
            # Post-process the model predictions and obtain labels
            predict = EEGEvaluator.postprocess_outputs(predict, length=self.length)
            # Compute evaluation metrics for smoothed outputs
            metrics_postprocess = EEGEvaluator.evaluate_metrics(labels[:, 0], predict)
            # Display results
            if visualize:
                EEGEvaluator.training_curve(history)
                print('========== Raw Output Metrics ==========')
                EEGEvaluator.test_results(metrics_raw)
                print('========== Smoothed Output Metrics ==========')
                EEGEvaluator.test_results(metrics_postprocess)
        return model

    def train_model_kfold_cv(self, epochs, model, name, folds=5, batch_size=25, control=None, save=True,
                    seq_len=20, use_seq=False, verbose=1, visualize=True):
        # Save the name of the training model
        self.name = name

        # Initialize list containers for training history and metrics
        history_list = [None for _ in range(len(self.patient_list))]
        metric_list = [None for _ in range(len(self.patient_list))]

        sz_split, nsz_split = EEGLearner.get_fold_splits(self.sz_list, self.nsz_list, folds)

        # dictionary to track test patients for separate model folds
        pts_by_fold = {}

        # create directory for model data
        if save:
            # create proper directory for model without overwriting
            new_dir = 'model-%s' % name
            new_dir = EEGLearner.make_save_dir(new_dir)

        # Iterate over all patients for cross validation (kfold)
        for i in range(folds):
            print(f'Fold: {i}')
            # Initialize generators for training, validation and testing data
            train_patients, validation_patients, test_patients = self.split_data_kfold(self.sz_list, self.nsz_list, sz_split, nsz_split, i, train_split=0.9)
            train_generator = EEGDataGenerator(train_patients, batch_size=batch_size, control=control,
                                                sample_len=self.length, seq_len=seq_len, use_seq=use_seq)
            validation_generator = EEGDataGenerator(validation_patients, batch_size=batch_size, control=control,
                                                    sample_len=self.length, seq_len=seq_len, shuffle=False,
                                                    use_seq=use_seq)
            test_generator = EEGDataGenerator(test_patients, batch_size=batch_size, sample_len=self.length,
                                                seq_len=seq_len, shuffle=False, use_seq=use_seq)
            # Load and train the model
            history = model.fit_generator(generator=train_generator, validation_data=validation_generator,
                                            class_weight=self.loss_weights, epochs=epochs, shuffle=True,
                                            verbose=verbose)
            history_list[i] = history
            # save test patients for corresponding model
            pts_by_fold[i] = test_patients
            # Save model and obtain predictions for test data
            if save:
                model.save('cnn_models\%s\%s-fold-%d.h5' % (new_dir, name, i), save_format='h5')
            predict = model.predict_generator(test_generator, verbose=0)
            predict = np.argmax(predict, axis=1)
            # Obtain annotations from the test generator
            labels = test_generator.get_annots()
            # Post-process the model predictions and obtain labels
            predict = EEGEvaluator.postprocess_outputs(predict, length=self.length)
            # Compute evaluation metrics for post-processed outputs
            metrics_postprocess = EEGEvaluator.evaluate_metrics(labels[:, 0], predict)
            metric_list[i] = metrics_postprocess

        # save fold -> test patients mapping as pickle file in model folder
        if save:
            test_pts_filename = 'cnn_models\%s\cnn_test_pts_by_fold.pkl' % new_dir
            with open(test_pts_filename, 'wb') as pkl_file:
                pickle.dump(pts_by_fold, pkl_file)

        # Display results
        if visualize:
            EEGEvaluator.training_curve_cv(history_list)
            EEGEvaluator.test_results_cv(metric_list)


    # Divides training, validation and testing data
    # Inputs
    #   patient_list: list of all patient IDs
    #   train_split: proportion of training data
    #   validation_split: proportion of validation data
    # Outputs
    #   train_patients: list of patient IDs to be used for training
    #   validation_patients: list of patient IDs to be used for validation
    #   test_patients: list of patient IDs to be used for testing
    @staticmethod
    def split_data(patient_list, train_split, validation_split):
        # Reject any invalid input
        if train_split + validation_split > 1:
            print("Please provide an appropriate split input.")
            return
        # Shuffle the list of patient IDs
        num = len(patient_list)
        new_patient_list = random.sample(patient_list, num)
        # Obtain the list of patients used for training, validating and testing
        train_patients = new_patient_list[:int(train_split * num)]
        validation_patients = new_patient_list[int(train_split * num):int((train_split + validation_split) * num)]
        test_patients = new_patient_list[int((train_split + validation_split) * num):]
        return train_patients, validation_patients, test_patients

    # Divides training and validation data with a specified test patient
    # Inputs
    #   patient_list: list of all patient IDs
    #   train_split: proportion of training data
    #   idx: index of the testing patient
    # Outputs
    #   train_patients: list of patient IDs to be used for training
    #   validation_patients: list of patient IDs to be used for validation
    #   test_patients: list of patient IDs to be used for testing
    @staticmethod
    def split_data_fix(patient_list, train_split, idx):
        # Reject any invalid input
        if train_split > 1:
            print("Please provide an appropriate split input.")
            return
        # Shuffle the list of patient IDs except for the designated test patient
        new_patient_list = [patient for patient in patient_list if patient != patient_list[idx]]
        random.shuffle(new_patient_list)
        num = len(new_patient_list)
        # Obtain the list of patients used for training, validating and testing
        train_patients = new_patient_list[:int(train_split * num)]
        validation_patients = new_patient_list[int(train_split * num):]
        test_patients = [patient_list[idx]]
        return train_patients, validation_patients, test_patients

    # Divides training and validation data with a list of specified test patients
    # Inputs
    #   patient_list: list of all patient IDs
    #   train_split: proportion of training data
    #   test_patients: list of patient IDs to be used for testing
    # Outputs
    #   train_patients: list of patient IDs to be used for training
    #   validation_patients: list of patient IDs to be used for validation
    #   test_patients: list of patient IDs to be used for testing
    @staticmethod
    def split_data_test(patient_list, train_split, test_patients):
        # Reject any invalid input
        if train_split > 1:
            print("Please provide an appropriate split input.")
            return
        # Shuffle the list of patient IDs except for the designated test patient
        new_patient_list = list(set(patient_list) - set(test_patients))
        num = len(new_patient_list)
        # Obtain the list of patients used for training, validating and testing
        train_patients = new_patient_list[:int(train_split * num)]
        validation_patients = new_patient_list[int(train_split * num):]
        return train_patients, validation_patients, test_patients

    @staticmethod
    def get_fold_splits(sz_list, nsz_list, n_folds):
        # create kfold objects to split sz and nsz data
        kf_sz = KFold(n_splits=n_folds, shuffle=True)
        kf_nsz = KFold(n_splits=n_folds, shuffle=True)

        # get split indices for all 5 folds
        sz_split = list(kf_sz.split(sz_list))
        nsz_split = list(kf_nsz.split(nsz_list))

        return sz_split, nsz_split

    @staticmethod
    def split_data_kfold(pt_list_sz, pt_list_nsz, sz_split, nsz_split, curr_fold, train_split=0.9):
        # get train split indices for ith fold
        train_idx_sz, test_idx_sz = sz_split[curr_fold]
        train_idx_nsz, test_idx_nsz = nsz_split[curr_fold]

        # construct training set from splits of sz and nsz list
        train_sz = pt_list_sz[train_idx_sz]
        train_nsz = pt_list_nsz[train_idx_nsz]
        train_list = np.r_[train_sz, train_nsz]

        train_patients = train_list[:int(train_split*len(train_list))]
        validation_patients = train_list[int(train_split*len(train_list)):]

        # construct testing set from sz and nsz splits to save with model
        test_sz = pt_list_sz[test_idx_sz]
        test_nsz = pt_list_nsz[test_idx_nsz]
        test_patients = np.r_[test_sz, test_nsz]

        return train_patients, validation_patients, test_patients

    @staticmethod
    def make_save_dir(save_dir):

        # check for previous model directories to prevent overwriting
        counter = 0
        while os.path.exists(os.path.join('cnn_models', save_dir)):

            # update counter
            counter += 1

            # update directory to check for
            check_name = save_dir + '_' + str(counter)

        # create new directory with proper suffix
        if counter > 0:
            new_dir = save_dir + '_' + str(counter) 
        else:
            new_dir = save_dir 
        os.makedirs('cnn_models\%s' % new_dir)

        return new_dir