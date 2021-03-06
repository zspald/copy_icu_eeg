#############################################################################
# Removes artifacts from EEG data obtained at IEEG.org
# Deployment version of artifacts.py
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
#############################################################################

# Import libraries
import numpy as np
import pandas as pd
import scipy.stats
from features import EEGFeatures


# A class that provides methods for artifact rejection
class Artifacts:

    # Decides whether each EEG segment is an artifact-free signal
    # Inputs
    #   input_data: EEG data of shape N x C x S, where N is the number of EEG segments from
    #               a patient's dataset, C is the number of valid EEG channels and S is the
    #               number of samples within the EEG segment
    #   fs: sampling frequency of the EEG recording
    #   channel_limit: number of channels to allow for artifacts
    #   method: specific method for removing artifact-like segments.
    #           'stats' - removes all segments that exceed a z-score of 5 under
    #           1) Range 2) Line Length 3) Bandpower in beta frequency band (12 - 20 Hz)
    #           'threshold' - removes all segments that exceed/fall below certain thresholds under
    #           1) Variance 2) Range 3) Line Length 4) Bandpower 5) Signal difference
    # Outputs
    #   indices_to_remove: a list with length N that indicates whether each EEG segment should be removed
    #   channels_to_remove: a N x C array that indicates whether each channel in each segment should be removed
    @staticmethod
    def remove_artifacts(input_data, fs, channel_limit=3, method='threshold'):
        # Remove segments that include NaN values first
        indices_to_remove, channels_to_remove = Artifacts.remove_nans(input_data)
        input_data = np.nan_to_num(input_data)
        if np.size(input_data, axis=0) == 0:
            return list()
        elif method == 'stats':
            indices_to_remove, channels_to_remove = \
                Artifacts.remove_artifacts_stats(input_data, fs, channel_limit, indices_to_remove, channels_to_remove)
        elif method == 'threshold':
            indices_to_remove, channels_to_remove = \
                Artifacts.remove_artifacts_thresholds(input_data, fs, channel_limit, indices_to_remove,
                                                      channels_to_remove)
        return indices_to_remove, channels_to_remove

    # Removes NaN recordings from an input EEG dataset
    # Inputs
    #   input_data: EEG data of shape N x C x S, where N is the number of EEG segments from
    #               a patient's dataset, C is the number of valid EEG channels and S is the
    #               number of samples within the EEG segment
    #   channel_limit: the maximum number of channels that may contain artifacts within each segment
    # Outputs
    #   segments_to_remove: a list with length N that indicates whether each EEG segment should be removed
    #   channels_to_remove: a N x C array that indicates whether each channel in each segment should be removed
    @staticmethod
    def remove_nans(input_data, channel_limit=3):
        # Create a N x C array that indicates whether every channel contains NaN values
        mask = np.amax(np.array(np.isnan(input_data), dtype=int), axis=-1)
        mask_sum = np.sum(mask, axis=-1)
        # Replace NaNs with zeros
        input_data = np.nan_to_num(input_data)
        segments_to_remove = np.zeros(np.size(input_data, axis=0), dtype=int)
        channels_to_remove = mask
        # Iterate over all segments, adding non-NaN recordings to the output
        for ii in range(np.size(input_data, axis=0)):
            if mask_sum[ii] > channel_limit:
                segments_to_remove[ii] = 1
                channels_to_remove[ii] = 1
        return segments_to_remove, channels_to_remove

    # Removes artifacts that include statistically outlying features
    # Inputs
    #   input_data: EEG data of shape N x C x S, where N is the number of EEG segments from
    #               a patient's dataset, C is the number of valid EEG channels and S is the
    #               number of samples within the EEG segment
    #   fs: sampling frequency of the EEG recording
    #   indices_to_remove: a list with length N that indicates whether each EEG segment should be removed
    # Outputs
    #   indices_to_remove: a list indicating whether each EEG segment should be removed, with length N*
    @staticmethod
    def remove_artifacts_stats(input_data, fs, channel_limit=3, indices_to_remove=None, channels_to_remove=None):
        if indices_to_remove is None:
            indices_to_remove = np.zeros(np.size(input_data, axis=0))
        # Calculate statistical features for artifact identification
        minmax = np.amax(input_data, axis=2) - np.amin(input_data, axis=2)
        range_z = scipy.stats.zscore(minmax, axis=None)
        llength = EEGFeatures.line_length(input_data)
        llength_z = scipy.stats.zscore(llength, axis=None)
        bdpower = EEGFeatures.bandpower(input_data, fs, 12, 20)
        bdpower_z = scipy.stats.zscore(bdpower, axis=None)
        diff = EEGFeatures.diff_signal(input_data, fs, window_size=0.03)
        diff_z = scipy.stats.zscore(diff, axis=None)
        z_threshold = 5
        # Check threshold compliance for every channel
        violations = (range_z > z_threshold) | (llength_z > z_threshold) | (bdpower_z > z_threshold) | \
                     (diff_z > z_threshold)
        violations = np.array(violations, dtype=int)
        violation_sum = np.sum(violations, axis=-1)
        # Iterate over each segment to check for channels that contain artifacts
        for ii in range(np.size(input_data, axis=0)):
            if violation_sum[ii] > channel_limit:
                indices_to_remove[ii] = 1
                channels_to_remove[ii] = 1
            else:
                channels_to_remove[ii] = violations[ii]
        # Remove noise-filled channels (threshold is 50% of all input samples)
        channels_to_filter = np.sum(channels_to_remove, axis=0) > int(0.5 * np.size(channels_to_remove, axis=0))
        channels_to_remove[:, channels_to_filter] = 1
        return indices_to_remove, channels_to_remove

    # Removes artifacts using threshold measures
    # Inputs
    #   input_data: EEG data of shape N x C x S, where N is the number of EEG segments from
    #               a patient's dataset, C is the number of valid EEG channels and S is the
    #               number of samples within the EEG segment
    #   fs: sampling frequency of the EEG recording
    #   indices_to_remove: a list with length N that indicates whether each EEG segment should be removed
    #   channels_to_remove: a N x C array that indicates whether each channel in each segment should be removed
    # Outputs
    #   indices_to_remove: a list with length N that indicates whether each EEG segment should be removed
    #   channels_to_remove: a N x C array that indicates whether each channel in each segment should be removed
    @staticmethod
    def remove_artifacts_thresholds(input_data, fs, channel_limit=3, indices_to_remove=None, channels_to_remove=None):
        if indices_to_remove is None:
            indices_to_remove = np.zeros(np.size(input_data, axis=0))
        # Feature-specific thresholds for removing artifacts
        var_threshold = 5
        minmax_threshold = 500
        llength_threshold = 1e4
        bdpower_threshold = 1e6
        diff_threshold = 100
        # Compute features for artifact rejection
        var = np.var(input_data, axis=-1)
        minmax = np.amax(input_data, axis=-1) - np.amin(input_data, axis=-1)
        llength = EEGFeatures.line_length(input_data)
        bdpower = EEGFeatures.bandpower(input_data, fs, 12, 20)
        diff = EEGFeatures.diff_signal(input_data, fs, window_size=0.03)
        # Check threshold compliance for every channel
        violations = (var < var_threshold) | (minmax > minmax_threshold) | (llength > llength_threshold) \
            | (bdpower > bdpower_threshold) | (diff > diff_threshold)
        violations = np.array(violations, dtype=int)
        violation_sum = np.sum(violations, axis=-1)
        # Iterate over each segment to check for channels that contain artifacts
        for ii in range(np.size(input_data, axis=0)):
            if violation_sum[ii] > channel_limit:
                indices_to_remove[ii] = 1
                channels_to_remove[ii] = 1
            else:
                channels_to_remove[ii] = violations[ii]
        # Remove noise-filled channels (threshold is 50% of all input samples)
        channels_to_filter = np.sum(channels_to_remove, axis=0) > int(0.5 * np.size(channels_to_remove, axis=0))
        channels_to_remove[:, channels_to_filter] = 1
        return indices_to_remove, channels_to_remove
