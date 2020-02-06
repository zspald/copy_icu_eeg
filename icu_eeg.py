#################################################################################
# Loads and saves preprocessed EEG annotations, features from a specified patient
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
#################################################################################

# Import libraries
import h5py
import numpy as np
from preprocess_dataset import IEEGDataProcessor


# A class that processes and saves EEG features/annotations from IEEG.org
class IEEGData:

    # The constructor for the IEEGData class
    def __init__(self, dataset_id, user, pwd):
        self.processor = IEEGDataProcessor(dataset_id, user, pwd)
        self.feat_storage = list()
        self.label_storage = list()

    # Processes features over a multiple number of iterations to reduce storage space
    # Inputs
    #   num_iter: total number of iterations over the patient data
    #   num_batches: number of EEG segments within each iteration
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   use_filter: whether to apply a bandpass filter to the EEG
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    #   channels_to_filter: a list of EEG channels to filter
    def process_all_feats(self, num_iter, num_batches, start, length, use_filter=True, eeg_only=True,
                          channels_to_filter=None):
        for ii in range(num_iter):
            feats, labels = self.processor.get_features(num_batches, start, length, norm='off', use_filter=use_filter
                                                        , eeg_only=eeg_only, channels_to_filter=channels_to_filter)
            self.feat_storage.append(feats)
            self.label_storage.append(labels)
