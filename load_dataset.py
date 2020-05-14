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
import pandas as pd
import pickle
import re

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

    # Loads annotations from IEEG.org for the specified batch of EEG
    # Inputs
    #   num: number of EEG batches to load
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   annot_idx: index of the annotation layer to use
    #   use_file: whether to use a given .csv file as the annotation
    # Outputs
    #   annotations: a numpy array of length 'num' that contains annotations for each EEG segment
    def load_annots(self, num, start, length, annot_idx=0, use_file=True):
        sz_intervals = self.load_annots_source(annot_idx, use_file)
        annotations = [(0, 0, 0) for _ in range(num)]
        for ii in range(num):
            begin = start + ii * length
            if IEEGDataLoader.search_interval(sz_intervals, start, length):
                annotations[ii] = (1, begin, begin + length)
            else:
                annotations[ii] = (0, begin, begin + length)
            start += num * length
        return annotations

    # Loads seizure intervals from IEEG.org for the specified patient
    # Inputs
    #   annot_idx: index of the annotation layer to use
    #   use_file: whether to use a given .csv file as the annotation
    # Outputs
    #   sz_intervals: a numpy array of shape R x 2 where R is the number of seizure intervals.
    #                 Each row contains the start and stop time, measured in seconds
    def load_annots_source(self, annot_idx=0, use_file=True):
        if use_file:
            # Read the .csv file and return the seizure intervals
            with open('dataset/' + self.id + '.pkl', 'rb') as file:
                dataframe = pickle.load(file)
            dataframe = dataframe[dataframe.event == 'seizure']
            sz_start = np.asarray(dataframe)[:, 1]
            sz_stop = np.asarray(dataframe)[:, 2]
            sz_intervals = np.c_[sz_start, sz_stop]
        else:
            # Define regular expressions for seizure onset/stop
            start_pattern = re.compile('osz', re.IGNORECASE)
            stop_pattern = re.compile('sze|szstop', re.IGNORECASE)
            sz_count = 0  # A counter to ensure that start and end times alternate
            # Obtain annotation layers from IEEG
            annot_layers = self.dataset.get_annotation_layers()
            layer_name = list(annot_layers.keys())[annot_idx]
            annots = self.dataset.get_annotations(layer_name)
            # Initialize list of seizure type, start and stop times
            sz_list, sz_start, sz_stop = list(), list(), list()
            for annot in annots:
                if re.match(start_pattern, annot.type) and sz_count == 0:
                    sz_list.append('sz')
                    sz_start.append(int(annot.start_time_offset_usec / 1e6))
                    sz_count += 1
                elif re.match(stop_pattern, annot.type) and sz_count == 1:
                    sz_stop.append(int(annot.start_time_offset_usec / 1e6))
                    sz_count -= 1
            # Create a pandas dataframe and save it
            sz_data = {'event': sz_list, 'start': sz_start, 'stop': sz_stop}
            dataframe = pd.DataFrame(data=sz_data)
            dataframe.to_pickle(r'./dataset/%s.pkl' % self.id)
            # Output the seizure intervals
            sz_intervals = np.c_[sz_start, sz_stop]
        return sz_intervals

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

    # Performs a binary search over a list of intervals to decide whether an EEG segment is
    # contained within a seizure interval
    # Inputs
    #   intervals: a numpy array of shape R x 2 where R is the number of seizure intervals.
    #              Each row contains the start and stop time, measured in seconds
    #   timepoint: the starting point of the EEG segment of interest, in seconds
    #   length: length of the EEG segment, in seconds
    # Outputs
    #   a boolean indicating whether the given EEG segment should be labeled as 'seizure'
    @staticmethod
    def search_interval(intervals, timepoint, length):
        idx = int(np.size(intervals, axis=0) / 2)
        if timepoint + length >= intervals[idx][0] and timepoint < intervals[idx][1]:
            return True
        elif idx == 0:
            return False
        elif timepoint + length < intervals[idx][0]:
            return IEEGDataLoader.search_interval(intervals[:idx, :], timepoint, length)
        elif timepoint >= intervals[idx][1]:
            return IEEGDataLoader.search_interval(intervals[idx:, :], timepoint, length)
        else:
            return False
