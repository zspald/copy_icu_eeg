###################################################################################################
# Loads EEG data, annotations from ICU patients at HUP (Hospital at the University of Pennsylvania)
# from IEEG, a free online EEG database portal.
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
###################################################################################################

# Import libraries
from ieeg.auth import Session
from ieeg.ieeg_api import IeegConnectionError
import math
import numpy as np
from scipy.signal import butter, filtfilt

# List of all scalp EEG channels
EEG_CHANNELS = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz',
                'T3', 'T4', 'T5', 'T6']


# A class that loads IEEG data for the specified patient
class IEEGDataLoader:

    # The constructor for the IEEGDataLoader class
    # Fields
    #   dataset: the IEEG dataset object which contains all information about the given dataset
    #   channel_labels: a list of strings containing channel names
    #   channel_indices: a list of integers containing channel indices
    #   details: a dictionary containing
    #   fs: sampling frequency of the recording
    #   montage: the current montage being used
    def __init__(self, dataset_id, user, pwd):
        # Open IEEG Session with the specified ID and password
        self.dataset = Session(user, pwd).open_dataset(dataset_id)
        self.channel_labels = self.dataset.get_channel_labels()
        self.channel_indices = self.dataset.get_channel_indices(self.channel_labels)
        self.details = self.dataset.get_time_series_details(self.channel_labels[0])
        self.fs = self.details.sample_rate
        self.montage = self.dataset.get_current_montage()

    # Loads data from IEEG.org for the specified patient
    # Inputs
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   use_filter: whether to apply a bandpass filter to the EEG
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    #   channels_to_filter: a list of EEG channels to filter
    # Outputs: data_pulled: a numpy array of shape D x C, where D is the number of samples
    #                       and C is the number of valid channels
    def load_data(self, start, length, use_filter=True, eeg_only=True, channels_to_filter=None):
        # Determine channels to use for EEG extraction
        if eeg_only:
            channels_to_use = self.filter_channels(channels_to_exclude=channels_to_filter)
        else:
            channels_to_use = self.channel_indices
        # Check whether too much data is requested for storage
        try:
            np.zeros((length, len(channels_to_use)))
        except MemoryError:
            print('Too much data is requested at once - please consider using smaller numbers of batches'
                  'and increasing the number of iterations')
            return
        # Convert from seconds to microseconds
        start, length = start * 1e6, length * 1e6
        data_pulled = self.dataset.get_data(start, length, channels_to_use)
        # Filter from 0.5 to 20 Hz if selected
        if use_filter:
            coeff = butter(4, [0.5 / (self.fs / 2), 20 / (self.fs / 2)], 'bandpass')
            data_pulled = filtfilt(coeff[0], coeff[1], data_pulled, axis=0)
        return data_pulled

    # Loads data from IEEG.org in multiple batches
    # Inputs
    #   num: number of EEG batches to load
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   use_filter: whether to apply a bandpass filter to the EEG
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    #   channels_to_filter: a list of EEG channels to filter
    # Outputs
    #   batch_data: numpy array of shape N x D x C, where N is the number of segments,
    #               D is the number of samples and C is the number of valid channels
    def load_data_batch(self, num, start, length, use_filter=True, eeg_only=True, channels_to_filter=None):
        try:
            raw_data = self.load_data(start, num * length, use_filter, eeg_only, channels_to_filter)
        except IeegConnectionError:  # Too much data is loaded from IEEG in this case
            data_left = self.load_data_batch(math.floor(num / 2), start, length, use_filter, channels_to_filter)
            data_right = self.load_data_batch(math.ceil(num / 2), start + math.floor(num / 2) * length, length,
                                              use_filter, channels_to_filter)
            raw_data = np.r_[data_left, data_right]
        batch_data = np.reshape(raw_data, (num, int(length * self.fs), np.size(raw_data, axis=-1)))
        return batch_data

    # Loads annotations from IEEG.org for the specified patient
    # Inputs
    #   annot_idx - index of the annotation layer to use
    # Outputs
    #   type - a string indicating which class of annotations to extract
    def load_annots(self, annot_idx=0, type='seizure'):
        annot_layers = self.dataset.get_annotation_layers()
        layer_name = list(annot_layers.keys())[annot_idx]
        annots = self.dataset.get_annotations(layer_name)
        for annot in annots:
            if type == 'seizure':
                # TBD - possibly use regular expression to match seizure output format
                continue
        return None

    # Filters channels by allowing the user the specify which channels to remove
    # Inputs
    #   channels_to_exclude: a list of channels to exclude, consists of strings
    # Outputs
    #   channel_indices: a list of indices for channels that will be included
    def filter_channels(self, channels_to_exclude=None):
        channels = EEG_CHANNELS
        if channels_to_exclude is not None:
            for ch in channels_to_exclude:
                if ch in channels:
                    channels.remove(ch)
        channel_indices = self.dataset.get_channel_indices(channels)
        return channel_indices

    # Returns the sampling frequency
    # Outputs
    #   self.fs: sampling frequency of the recording
    def sampling_frequency(self):
        return self.fs
