#############################################################################
# Removes artifacts from EEG data obtained at IEEG.org
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
#############################################################################

# Import libraries
import numpy as np
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
    #   method: specific method for removing artifact-like segments. 'default' removes all
    #           segments that exceed a certain threshold under
    #           1) Range 2) Line Length 3) Bandpower in beta frequency band (25 - 60 Hz)
    #           other methods to be incorporated soon
    # Outputs
    #   indices_to_remove: a list indicating whether each EEG segment should be removed, with length N
    @staticmethod
    def remove_artifacts(input_data, fs, method='default'):
        # Remove segments that include NaN values first
        output_data, indices_to_remove = Artifacts.remove_nans(input_data)
        if method == 'default':
            indices_to_remove = Artifacts.remove_artifacts_default(output_data, fs, indices_to_remove)
        return indices_to_remove

    # Removes NaN recordings from an input EEG dataset
    # Inputs
    #   input_data: EEG data of shape N x C x S, where N is the number of EEG segments from
    #               a patient's dataset, C is the number of valid EEG channels and S is the
    #               number of samples within the EEG segment
    # Outputs
    #   output_data: EEG data of shape N' x C x S, where N' is the number of EEG segments
    #                that do not contain NaN values
    #   remove: a list indicating whether each EEG segment should be removed, with length N
    @staticmethod
    def remove_nans(input_data):
        boolean_mask = np.isnan(input_data)
        output_data = []
        remove = np.zeros(np.size(input_data, axis=0), dtype=bool)
        # Iterate over all segments, adding non-NaN recordings to the output
        for ii in range(np.size(input_data, axis=0)):
            if not boolean_mask[ii].any():
                if len(output_data) == 0:
                    output_data = np.expand_dims(input_data[ii], axis=0)
                else:
                    output_data = np.r_[output_data, np.expand_dims(input_data[ii], axis=0)]
            else:
                remove[ii] = 1
        return output_data, remove

    # Removes artifacts that include statistically outlying features
    # Inputs
    #   input_data: EEG data of shape N x C x S, where N is the number of EEG segments from
    #               a patient's dataset, C is the number of valid EEG channels and S is the
    #               number of samples within the EEG segment
    #   fs: sampling frequency of the EEG recording
    #   indices_to_remove: a list indicating whether each EEG segment should be removed, with length N
    @staticmethod
    def remove_artifacts_default(input_data, fs, indices_to_remove=None):
        if indices_to_remove is None:
            indices_to_remove = np.zeros(np.size(input_data, axis=0))
        minmax = np.mean(np.amax(input_data, axis=2) - np.amin(input_data, axis=2), axis=1)
        range_z = scipy.stats.zscore(minmax)
        llength = np.mean(EEGFeatures.line_length(input_data), axis=1)
        llength_z = scipy.stats.zscore(llength)
        bdpower = np.mean(EEGFeatures.bandpower(input_data, fs, 25, 60), axis=1)
        bdpower_z = scipy.stats.zscore(bdpower)
        # Initialize iterator variables for update procedure
        cnt = 0
        idx = 0
        # Update the removal list whenever statistical outliers are found
        while cnt < np.size(input_data, axis=0):
            if indices_to_remove[idx] == 0:
                # Assign different numbers in case artifact information is needed
                if range_z[cnt] > 3:
                    indices_to_remove[idx] = 1
                elif llength_z[cnt] > 3:
                    indices_to_remove[idx] = 2
                elif bdpower_z[cnt] > 3:
                    indices_to_remove[idx] = 3
                cnt += 1
            idx += 1
        print('Number of rejected segments: ', np.sum(indices_to_remove))
        return indices_to_remove
