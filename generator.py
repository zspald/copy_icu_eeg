############################################################################
# Generates training/validation data to be fed into the training loop
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
import h5py
import numpy as np
import tensorflow as tf


# A class that generates EEG training/validation data in batches
class EEGDataGenerator:

    # The constructor for the EEGDataGenerator class
    # Fields
    #   patient_list: list of all patient IDs
    #   batch_size: the number of samples in each batch
    #   shuffle: whether to shuffle order of training/validation batches
    def __init__(self, patient_list, batch_size=1e4, shuffle=True):
        self.patient_list = patient_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.patient_list))
        self.batch_list = list()
        # Iterate over all patients to determine the total number of samples and batches
        for patient in patient_list:
            with open('data/%s_data.h5' % patient, 'r') as file:
                self.batch_list.append(np.ceil(file['maps'].shape[0] / batch_size))
        self.on_epoch_end()

    # Returns the total number of batches required for training each epoch
    def __len__(self):
        return sum(self.batch_list)

    # Re-shuffles the training/validation data for every epoch
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        return

    # Returns the next batch of data and labels
    # Inputs
    #   idx: current batch index
    # Outputs
    #   output_data: EEG maps returned by the generator
    #   output_labels: labels corresponding to the generated EEG maps
    def __getitem__(self, idx):
        patient_num = 0
        batch_sum = self.batch_list[0]
        while batch_sum < idx:
            patient_num += 1
            batch_sum += self.batch_list[self.indices[patient_num]]
        batch_idx = idx - batch_sum + self.batch_list[self.indices[patient_num]]
        output_data, output_labels = self.__data_generation(self.indices[patient_num], batch_idx)
        return output_data, output_labels

    # Generates data and labels for the specified batch
    # Inputs
    #   patient_idx: index of current patient, ordered identically to patient list
    #   batch_idx: index of current batch, starting at 0
    # Outputs
    #   output_data: EEG maps returned by the generator
    #   output_labels: labels corresponding to the generated EEG maps
    def __data_generation(self, patient_idx, batch_idx):
        with open('data/%s_data.h5' % self.patient_list[patient_idx], 'r') as file:
            if batch_idx == self.batch_list[patient_idx] - 1:
                output_data = file['maps'][batch_idx * self.batch_size:, :, :, :]
                output_labels = file['labels'][batch_idx * self.batch_size:, 0]
            else:
                output_data = file['maps'][batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size, :, :, :]
                output_labels = file['labels'][batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size, 0]
        output_labels = tf.keras.utils.to_categorical(output_labels, num_classes=2)
        return output_data, output_labels
