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
import re
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
        self.id = dataset_id
        self.dataset = Session(user, pwd).open_dataset(dataset_id)
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
        except IeegConnectionError:  # Too much data is loaded from IEEG
            data_left = self.load_data_batch(math.floor(num / 2), start, length, use_filter, channels_to_filter)
            data_right = self.load_data_batch(math.ceil(num / 2), start + math.floor(num / 2) * length, length,
                                              use_filter, channels_to_filter)
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
        annotations = np.zeros(num)
        for ii in range(num):
            annotations[ii] = 1 if IEEGDataLoader.search_interval(sz_intervals, start, length) else 0
            start += num
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
            dataframe = pd.read_csv(self.id + '_annots.csv')
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
            sz_data = {'seizure_type': sz_list, 'start': sz_start, 'stop': sz_stop}
            dataframe = pd.DataFrame(data=sz_data)
            dataframe.to_csv(r'./%s_annots.csv' % self.id)
            # Output the seizure intervals
            sz_intervals = np.c_[sz_start, sz_stop]
        return sz_intervals

    # Performs a binary search over a list of intervals to decide whether an EEG segment is
    # contained within a seizure interval
    # Inputs
    #   intervals: a numpy array of shape R x 2 where R is the number of seizure intervals.
    #              Each row contains the start and stop time, measured in seconds
    #   timepoint: the starting point of the EEG segment of interest, in seconds
    #   length: length of the EEG segment, in seconds
    @staticmethod
    def search_interval(intervals, timepoint, length):
        idx = int(np.size(intervals, axis=0) / 2)
        if timepoint + length >= intervals[idx][0] and timepoint < intervals[idx][1]:
            return True
        elif timepoint + length < intervals[idx][0]:
            return IEEGDataLoader.search_interval(intervals[:idx, :], timepoint, length)
        elif timepoint >= intervals[idx][1]:
            return IEEGDataLoader.search_interval(intervals[idx:, :], timepoint, length)
        else:
            return False
