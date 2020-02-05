############################################################################
# Preprocesses EEG data, annotations from ICU patients at HUP
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
from artifacts import Artifacts
from load_dataset import IEEGDataLoader
import numpy as np

# Artifact rejection method
ARTIFACT_METHOD = 'default'


# A class that preprocesses IEEG data for the specified patient
# Inherits fields and methods from the IEEGDataLoader class
class IEEGDataProcessor(IEEGDataLoader):

    # The constructor for the IEEGDataProcessor class
    def __init__(self, dataset_id, user, pwd):
        super(IEEGDataProcessor, self).__init__(dataset_id, user, pwd)
        self.data_storage = list()
        self.label_storage = list()

    # A function that processes data from IEEG.org in multiple batches
    # Inputs
    #   num: number of EEG batches to process
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   use_filter: whether to apply a bandpass filter to the EEG
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    #   channels_to_filter: a list of EEG channels to filter
    # Outputs
    #   output_data: a set of processed EEG segments with shape N* x C x D, where N* is the number
    #                of clean EEG segments from the patient's dataset, C is the number of clean
    #                channels and D is the number of samples within each segment (S = fs * s_length)
    #   output_labels: a modified list of seizure annotations of the given patient with length N*
    def process_data(self, num, start, length, use_filter=True, eeg_only=True, channels_to_filter=None):
        input_data = IEEGDataLoader.load_data_batch(self, num, start, length, use_filter, eeg_only, channels_to_filter)
        input_data = np.swapaxes(input_data, 1, 2)
        input_labels = IEEGDataLoader.load_annots(self)
        sample_freq = IEEGDataLoader.sampling_frequency(self)
        output_data, output_labels, _ = self.clean_data(input_data, input_labels, sample_freq, ARTIFACT_METHOD)
        return output_data, output_labels

    # A function that processes batches in multiple iterations from IEEG.org
    # Inputs
    #   num_iter: number of iterations over the dataset
    #   num_batch: number of EEG segments within each batch
    #   start: initial starting point, in seconds
    #   length: duration of each segment, in seconds
    #   use_filter: whether to apply a bandpass filter to the EEG
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    #   channels_to_filter: a list of EEG channels to filter
    def process_data_iter(self, num_iter, num_batch, start, length, use_filter=True, eeg_only=True,
                          channels_to_filter=None):
        for ii in range(num_iter):
            data, labels = self.process_data(num_batch, start, length, use_filter, eeg_only, channels_to_filter)
            self.data_storage.append(data)
            self.label_storage.append(labels)
        return

    # A function that performs artifact rejection over all EEG recordings within the given dataset
    # and updates the annotation file in accordance with the processed EEG data
    # Inputs
    #   input_data: EEG sample of shape N x C x D, where N is the number of EEG batches, C is the number of
    #               valid channels and D is the total number of samples within the given patient's EEG
    #   input_labels: a list of seizure annotations of the given patient, with length (Q / fs)
    #                 i.e. sampled at 1 Hz
    #   fs: sampling frequency of the patient's EEG recording
    #   s_length: length of each EEG segment to be inspected, in seconds
    #   artifact_rejection: specific method for removing artifact-like segments.
    #                       'default' removes all segments that exceed a certain threshold under
    #                       1) Range 2) Line Length 3) Bandpower in beta frequency band (25 - 60 Hz)
    #                       other methods to be incorporated
    # Outputs
    #   output_data: a set of processed EEG segments with shape N* x C x D, where N* is the number
    #                of valid EEG segments from the patient's dataset, C is the number of valid
    #                channels and D is the number of samples within each segment (S = fs * s_length)
    #   output_labels: a modified list of seizure annotations of the given patient with length N*
    #   verifier: a list indicating whether each EEG segment should be removed, with length N
    @staticmethod
    def clean_data(input_data, input_labels, fs, artifact_rejection='default'):
        s_length = int(np.size(input_data, axis=-1) / fs)
        # Determine batches to remove
        remove = Artifacts.remove_artifacts(input_data, fs, method=artifact_rejection)
        # Remove artifact data
        indices_to_keep = [idx for idx in remove if remove[idx] == 0]
        output_data = input_data[indices_to_keep]
        # Update labels to match the size of the clean recordings
        output_labels = []
        for ii in range(len(remove)):
            if remove[ii] == 0:
                labels = np.amax(input_labels[int(ii * s_length): int((ii + 1) * s_length)])
                output_labels.append(labels)
        output_labels = np.array(output_labels)
        return output_data, output_labels, remove
