############################################################################
# Generates training/validation data to be fed into the training loop
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
import copy
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence


# A class that generates EEG training/validation data in batches
class EEGDataGenerator(Sequence):

    # The constructor for the EEGDataGenerator class
    # Attributes
    #   batch_list: a list that contains number of batches in each dataset
    #   batch_size: the number of samples in each batch
    #   indices: list of all patient indices hashed from IDs
    #   patient_list: list of all patient IDs
    #   shuffle: whether to shuffle order of training/validation batches
    def __init__(self, patient_list, batch_size, shuffle=True):
        self.patient_list = patient_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.patient_list))
        self.batch_list = np.zeros(len(self.patient_list))
        # Iterate over all patients to determine the total number of samples and batches
        for idx, patient_id in enumerate(patient_list):
            file = h5py.File('data/%s_data.h5' % patient_id, 'r')
            self.batch_list[idx] = np.ceil(file['maps'].shape[0] / batch_size)
            file.close()
        self.length = int(sum(self.batch_list))
        self.on_epoch_end()

    # Returns the total number of batches required for training each epoch
    def __len__(self):
        return self.length

    # Re-shuffles the training/validation data for every epoch
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        return

    # Returns labels from the given list of patient IDs
    def get_labels(self):
        labels = list()
        # Iterate over all batches to extract associated labels
        for idx in range(self.length):
            _, label = self.__getitem__(idx)
            labels.extend(list(np.argmax(label, axis=1)))
        # Convert to numpy array
        labels = np.asarray(labels)
        return labels

    # Returns the next batch of data and labels
    # Inputs
    #   idx: current batch index
    # Outputs
    #   output_data: EEG maps returned by the generator
    #   output_labels: labels corresponding to the generated EEG maps
    def __getitem__(self, idx):
        patient_num = 0
        batch_sum = self.batch_list[self.indices[patient_num]]
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
        file = h5py.File('data/%s_data.h5' % self.patient_list[patient_idx], 'r')
        if batch_idx == self.batch_list[patient_idx] - 1:
            output_data = file['maps'][-1 * int(self.batch_size):, :, :, :]
            output_labels = file['labels'][-1 * int(self.batch_size):, 0]
        else:
            output_data = file['maps'][int(batch_idx * self.batch_size):int((batch_idx + 1) * self.batch_size), :, :, :]
            output_labels = file['labels'][int(batch_idx * self.batch_size):int((batch_idx + 1) * self.batch_size), 0]
        output_data = copy.deepcopy(output_data)
        output_labels = copy.deepcopy(output_labels)
        file.close()
        output_labels = tf.keras.utils.to_categorical(output_labels, num_classes=2)
        return output_data, output_labels
