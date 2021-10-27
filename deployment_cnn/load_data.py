###################################################################################################
# Loads EEG data, annotations from ICU patients at HUP (Hospital at the University of Pennsylvania)
# from IEEG, a free online EEG database portal.
# Deployment version of load_dataset.py
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
###################################################################################################

# Import libraries
from ieeg.auth import Session
from ieeg.ieeg_api import IeegConnectionError
import math
import numpy as np

# List of all scalp EEG channels
EEG_CHANNELS = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz',
                'T3', 'T4', 'T5', 'T6']


# A class that loads IEEG data for the specified patient
class IEEGDataLoader:

    # The constructor for the IEEGDataLoader class
    # Attributes
    #   id: ID of the patient dataset
    #   session: the IEEG session object for opening the associated dataset
    #   dataset: the IEEG dataset object which contains all information about the given dataset
    #   channel_labels: a list of strings containing channel names
    #   channel_indices: a list of integers containing channel indices
    #   details: a dictionary containing the time series details of the recording
    #   fs: sampling frequency of the recording
    #   montage: the current montage being used
    def __init__(self, dataset_id, user, pwd):
        # Open IEEG Session with the specified ID and password
        self.id = dataset_id
        self.session = Session(user, pwd)
        self.dataset = self.session.open_dataset(dataset_id)
        self.channel_labels = self.dataset.get_channel_labels()
        self.channel_indices = self.dataset.get_channel_indices(self.channel_labels)
        self.details = self.dataset.get_time_series_details(self.channel_labels[0])
        self.fs = self.details.sample_rate
        self.montage = self.dataset.get_current_montage()

    # Returns the sampling frequency
    # Outputs
    #   self.fs: sampling frequency of the recording
    def sampling_frequency(self):
        return self.fs

    # Filters channels by allowing the user the specify which channels to remove
    # Inputs
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    # Outputs
    #   channel_indices: a list of indices for channels that will be included
    def filter_channels(self, eeg_only):
        channels = EEG_CHANNELS if eeg_only else self.channel_labels
        channel_indices = self.dataset.get_channel_indices(channels)
        return channel_indices

    # Loads data from IEEG.org for the specified patient
    # Inputs
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    # Outputs
    #   data_pulled: a numpy array of shape D x C, where D is the number of samples
    #                and C is the number of valid channels
    def load_data(self, start, length, eeg_only=True):
        # Determine channels to use for EEG extraction
        channels_to_use = self.filter_channels(eeg_only)
        # Check whether too much data is requested for storage
        try:
            np.zeros((length, len(channels_to_use)))
        except MemoryError:
            print('Too much data is requested at once - please consider using smaller numbers of batches'
                  'and increasing the number of iterations')
            return
        # Convert from seconds to microseconds
        start, length = int(start * 1e6), int(length * 1e6)
        data_pulled = self.dataset.get_data(start, length, channels_to_use)
        return data_pulled

    # Loads data from IEEG.org in multiple batches
    # Inputs
    #   num: number of EEG batches to load
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   use_filter: whether to apply a bandpass filter to the EEG
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    # Outputs
    #   batch_data: numpy array of shape N x D x C, where N is the number of segments,
    #               D is the number of samples and C is the number of valid channels
    def load_data_batch(self, num, start, length, eeg_only=True):
        try:
            raw_data = self.load_data(start, num * length, eeg_only)
        except IeegConnectionError:  # Too much data is loaded from IEEG
            data_left = self.load_data_batch(math.floor(num / 2), start, length)
            data_right = self.load_data_batch(math.ceil(num / 2), start + math.floor(num / 2) * length, length)
            raw_data = np.r_[data_left, data_right]
        batch_data = np.reshape(raw_data, (num, int(length * self.fs), np.size(raw_data, axis=-1)))
        return batch_data

    # Crawls the patient recordings to locate the first timepoint that does not contain NaNs
    # Inputs
    #   start: starting point, in seconds
    #   interval_length: length of each segment to inspect, in seconds
    #   threshold: time duration to search before rejecting patient data, in seconds
    #   channels_to_use:  list of the channels to pull data from
    # Outputs
    #   the first point that does not contain NaNs, in seconds, None if searching time exceeds threshold
    def crawl_data(self, start, interval_length, threshold, channels_to_use):
        contains_nan_prev = False
        while True:
            data = self.dataset.get_data(start * 1e6, interval_length * 1e6, channels_to_use)
            contains_nan = np.isnan(data).any()
            if contains_nan_prev and not contains_nan:
                return self.search_nans(start - interval_length, interval_length, channels_to_use)
            elif contains_nan:
                contains_nan_prev = contains_nan
                start += interval_length
                if start > threshold:
                    return
            else:
                return 0

    # Performs a binary search for the first timepoint that does not contain NaNs
    # Inputs
    #   start: starting point, in seconds
    #   duration: length of data pulled, in seconds
    #   channels_to_use: a list of the channels to pull data from
    # Outputs
    #   begin_time: the first point that does not contain NaNs, in seconds
    def search_nans(self, start, duration, channels_to_use):
        if duration > 1:
            start_usec, duration_usec = start * 1e6, duration * 1e6
            data_left = self.dataset.get_data(start_usec, int(duration_usec / 2), channels_to_use)
            data_right = self.dataset.get_data(start_usec + int(duration_usec / 2), math.ceil(duration_usec / 2),
                                               channels_to_use)
            left_contains, right_contains = np.isnan(data_left).any(), np.isnan(data_right).any()
            if left_contains and right_contains:
                return self.search_nans(start + int(duration / 2), math.ceil(duration / 2), channels_to_use)
            elif left_contains and not right_contains:
                return self.search_nans(start, int(duration / 2), channels_to_use)
        begin_time = start + 1
        return begin_time
