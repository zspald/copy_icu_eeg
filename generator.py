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
    #   sample_len: length of each EEG sample
    #   shuffle: whether to shuffle order of training/validation batches
    #   use_seq: whether the data should be generated for sequential models
    # ### the parameters below are only used if seq==True ###
    #   seq_len: length of the input sequence, in seconds
    #   seq_disp: displacement of the sequence, in seconds
    def __init__(self, patient_list, batch_size, sample_len, shuffle=True, use_seq=False, seq_len=20, seq_disp=5):
        # Initialize attributes with user inputs
        self.patient_list = patient_list
        self.batch_size = batch_size
        self.sample_len = sample_len
        self.shuffle = shuffle
        self.indices = np.arange(len(self.patient_list))
        self.use_seq = use_seq
        self.seq_len = seq_len
        self.on_epoch_end()
        # Use eager execution to pre-determine data to be extracted
        if use_seq:
            # Initialize batch_info, which contains indices of the heads of every sequence in each batch and patient
            self.batch_info = [[] for _ in range(len(self.patient_list))]
            self.length = self.initialize_batch_seq(seq_len, seq_disp)
        else:
            # Initialize the cross-patient batch list and obtain the number of all batches
            self.batch_list = np.zeros(len(self.patient_list))
            self.length = self.initialize_batch()

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
        if self.use_seq:  # Extract data in form of (batch_size x seq_len x data)
            patient_num = 0
            batch_sum = len(self.batch_info[self.indices[patient_num]])
            while batch_sum <= idx:
                patient_num += 1
                batch_sum += len(self.batch_info[self.indices[patient_num]])
            batch_idx = idx - batch_sum + len(self.batch_info[self.indices[patient_num]])
            output_data, output_labels = self.__data_generation_seq(self.indices[patient_num], batch_idx)
        else:  # Extract data in form of (batch_size x data)
            patient_num = 0
            batch_sum = self.batch_list[self.indices[patient_num]]
            while batch_sum <= idx:
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

    # Generates data and labels for the specified batch in a sequential manner
    # Inputs
    #   patient_idx: index of current patient, ordered identically to patient list
    #   batch_idx: index of current batch, starting at 0
    # Outputs
    #   output_data: EEG maps returned by the generator
    #   output_labels: labels corresponding to the generated EEG maps
    def __data_generation_seq(self, patient_idx, batch_idx):
        file = h5py.File('data/%s_data.h5' % self.patient_list[patient_idx], 'r')
        dataset = file['maps']
        output_data = np.zeros((self.batch_size, self.seq_len) + dataset.shape[1:])
        output_labels = np.zeros(self.batch_size)
        seq_indices = self.batch_info[patient_idx][batch_idx]
        for idx, seq_idx in enumerate(seq_indices):
            output_data[idx, :, :, :, :] = copy.deepcopy(dataset[seq_idx:seq_idx + self.seq_len, :, :, :])
            output_labels[idx] = max(file['labels'][seq_idx:seq_idx + self.seq_len, 0])
        file.close()
        output_labels = tf.keras.utils.to_categorical(output_labels, num_classes=2)
        return output_data, output_labels

    # Computes the number of batches in each patient for non-sequential data generation
    # Outputs
    #   num_batches: total number of batches in the dataset
    def initialize_batch(self):
        # Iterate over all patients to determine the total number of samples and batches
        for idx, patient_id in enumerate(self.patient_list):
            file = h5py.File('data/%s_data.h5' % patient_id, 'r')
            self.batch_list[idx] = np.ceil(file['maps'].shape[0] / self.batch_size)
            file.close()
        num_batches = int(sum(self.batch_list))
        return num_batches

    # Records the indices of samples in each batch to perform sequential data extraction from
    # Inputs
    #   seq_len: length of the sequence, in seconds
    #   seq_disp: displacement of the sequence, in seconds
    # Outputs
    #   num_batches: total number of batches in the dataset
    def initialize_batch_seq(self, seq_len, seq_disp):
        # Re-define inputs in terms of number of samples
        seq_len, seq_disp = int(seq_len / self.sample_len), int(seq_disp / self.sample_len)
        # Iterate over all patients to populate the batch_info attribute
        num_batches = 0
        for idx, patient_id in enumerate(self.patient_list):
            # Initialize batch/sample counters
            batch_idx = 0  # index of the batch within the patient dataset
            sample_idx = 0  # index of the sample within the batch
            seq_pos = 0  # current position of the head of the sequence
            # Open the file and read the labels
            file = h5py.File('data/%s_data.h5' % patient_id, 'r')
            data_labels = file['labels']
            data_len = data_labels.shape[0]
            # Use a sliding-window approach for each patient
            while seq_pos + seq_len <= data_len:
                # Check whether the data in the sequence are contingent
                if self.contingent(data_labels[seq_pos:seq_pos + seq_len]):
                    if sample_idx == 0:
                        self.batch_info[idx].append([seq_pos])
                        sample_idx += 1
                    else:
                        self.batch_info[idx][batch_idx].append(seq_pos)
                        sample_idx += 1
                        if sample_idx == self.batch_size:
                            batch_idx += 1
                            sample_idx = 0
                    if data_labels[seq_pos, 0] == 0:
                        seq_pos += seq_disp
                    else:
                        seq_pos += max(1, int(seq_disp / 2))
                # If not, then proceed forward by one sample
                else:
                    seq_pos += 1
            # Ensure that the last batch has self.batch_size samples
            if sample_idx > 0:
                count = self.batch_size - len(self.batch_info[idx][batch_idx])
                self.batch_info[idx][batch_idx].extend(self.batch_info[idx][0][:count])
            # Close the file and update total number of batches
            file.close()
            num_batches += batch_idx + 1
        return num_batches

    # Checks whether the labels are contiguous segments
    # Inputs
    #   labels: the labels to be checked for contingency
    #   threshold: the maximum amount of time allowed for disjoint segments
    # Outputs
    #   whether the labels are from continguous segments
    @staticmethod
    def contingent(labels, threshold=2):
        prev_timepoint = labels[0, 1]
        for label in labels:
            curr_timepoint = label[1]
            if curr_timepoint - prev_timepoint > threshold:
                return False
            prev_timepoint = label[2]
        return True
