############################################################################
# Extracts statistical features from EEG data in time/frequency domains
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
import math
import numpy as np
import pywt
import scipy.stats
from scipy.signal import hilbert

# EEG electrodes
ALL = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz',
       'T3', 'T4', 'T5', 'T6']
LEFT = ['C3', 'F3', 'F7', 'Fp1', 'O1', 'P3', 'T3', 'T5']
RIGHT = ['C4', 'F4', 'F8', 'Fp2', 'O2', 'P4', 'T4', 'T6']

# List of statistical EEG features
EEG_FEATS = ['Line Length', 'Delta Power', 'Theta Power', 'Alpha Power', 'Beta Power',
             'Skewness', 'Kurtosis', 'Envelope']


# A class that contains methods for extracting statistical EEG features
class EEGFeatures:

    # Extracts statistical EEG features from the input EEG data
    # Inputs
    #   input_data: EEG data of shape N x C x S, where N is the number of EEG segments from
    #               a patient's dataset, C is the number of valid EEG channels and S is the
    #               number of samples within the EEG segment
    #   fs: sampling frequency
    #   normalize: normalization method to be used for the extracted features
    #              'off' does not perform any normalization
    #              'default' normalizes features from 0 to 1
    #              'zscore' normalizes features according to their z-scores
    #              'minmax' normalizes features from -1 to 1
    #   pool_region: whether to manually pool statistics from a given region of electrodes
    # Outputs
    #   output_feats: an array of shape N x G x F, where N is the number of EEG segments,
    #                 G is the number of feature groups and F is the number of features
    #                 (Note that G = C when pool_region is set to be false.)
    @staticmethod
    def extract_features(input_data, fs, normalize='off', pool_region=False):
        output_feats = None
        # Line length
        llength = EEGFeatures.line_length(input_data)
        # Delta bandpower
        bdpower_delta = EEGFeatures.bandpower(input_data, fs, 0.5, 4)
        # Theta bandpower
        bdpower_theta = EEGFeatures.bandpower(input_data, fs, 4, 8)
        # Alpha bandpower
        bdpower_alpha = EEGFeatures.bandpower(input_data, fs, 8, 12)
        # Beta bandpower
        bdpower_beta = EEGFeatures.bandpower(input_data, fs, 12, 20)
        # Skewness
        skew = scipy.stats.skew(input_data, axis=-1)
        # Kurtosis
        kurt = scipy.stats.kurtosis(input_data, axis=-1)
        # Envelope
        envp = EEGFeatures.envelope(input_data)
        # Aggregate all features and compute mean over specified channels
        all_feats = np.array([llength, bdpower_delta, bdpower_theta, bdpower_alpha,
                                   bdpower_beta, skew, kurt, envp])
        # Apply regional pooling over specified regions of scalp electrodes based on user input
        if pool_region:
            categories = [ALL, LEFT, RIGHT]
            # Iterate through different types of electrode
            for category in categories:
                # Determine indices of intersection and filter the input data
                indices_to_use = np.nonzero(np.in1d(ALL, category))[0]
                category_feats = all_feats[:, :, indices_to_use]
                category_feats = np.expand_dims(np.nanmean(category_feats, axis=-1), axis=-1)
                if output_feats is None:
                    output_feats = category_feats
                else:
                    output_feats = np.c_[output_feats, category_feats]
        # Otherwise pass all features into the set of output features
        else:
            output_feats = all_feats
        # Rearrange axes to match the desired output format
        output_feats = np.swapaxes(np.swapaxes(output_feats, 0, 1), 1, 2)
        # Normalize the features if a specific method is indicated
        if normalize != 'off':
            output_feats = EEGFeatures.normalize_feats(output_feats, option=normalize)
        return output_feats

    # Normalizes features
    # Inputs
    #   input_feats: an array of shape N x G x F, where N is the number of EEG segments,
    #                G is the number of feature groups and F is the number of features
    #   option: normalization method to be used for the extracted features
    #           'default' normalizes features from 0 to 1
    #           'zscore' normalizes features according to their z-scores
    #           'minmax' normalizes features from -1 to 1
    # Outputs
    #   output_feats: array of same shape with normalized features
    @staticmethod
    def normalize_feats(input_feats, option='default'):
        # Normalize features between 0 to 1
        output_feats = (input_feats - np.nanmin(input_feats, axis=0)) / \
                       (np.nanmax(input_feats, axis=0) - np.nanmin(input_feats, axis=0))
        # Further normalize features based on user option
        if option == 'zscore':
            output_feats = (input_feats - np.nanmean(input_feats, axis=0)) / np.nanstd(input_feats, axis=0)
        elif option == 'minmax':
            output_feats = output_feats * 2 - 1
        return output_feats

    # Computes mean and standard deviation of the given data over all channels and features
    # Inputs
    #   input_feats: an array of shape N x C x F, where N is the number of EEG segments,
    #                C is the number of channels and F is the number of features
    # Outputs
    #   feats_mean: the mean of all channels and features, with shape C x F as described above
    #   feats_std: the standard deviation of all channels and features, with shape C x F as described above
    @staticmethod
    def compute_stats(input_feats):
        feats_mean = np.nanmean(input_feats, axis=0)
        feats_std = np.nanstd(input_feats, axis=0)
        return feats_mean, feats_std

    # Computes the line length of every EEG segment over every channel for a batch of EEG data
    # Inputs
    #   input_data: EEG data of shape N x C x S, where N is the number of EEG segments from
    #               a patient's dataset, C is the number of valid EEG channels and S is the
    #               number of samples within the EEG segment
    # Outputs
    #   llength: an array of shape N x C containing the line length of each segment & channel
    @staticmethod
    def line_length(input_data):
        llength = np.abs(np.diff(input_data, axis=-1))
        llength = np.sum(llength, axis=-1)
        return llength

    # Computes the bandpower of every EEG segment over every channel for a batch of EEG data
    # Inputs
    #   input_data: EEG data of shape N x C x S, where N is the number of EEG segments from
    #               a patient's dataset, C is the number of valid EEG channels and S is the
    #               number of samples within the EEG segment
    #   fs: sampling frequency
    #   low: the lower frequency bound
    #   high: the upper frequency bound
    # Outputs
    #   bandpower_values: an array of shape N x C that stores the bandpower of each segment &
    #                     channel within the specified frequency range
    @staticmethod
    def bandpower(input_data, fs, low, high):
        fft_values = np.abs(np.fft.rfft(input_data, axis=-1))
        fft_values = np.transpose(fft_values, [2, 1, 0])  # Reorder axes to slice the bandpower values of interest
        fft_freqs = np.fft.rfftfreq(np.size(input_data, axis=-1), d=1.0 / fs)
        freq_idx = np.where((fft_freqs >= low) & (fft_freqs <= high))[0]
        bandpower_values = fft_values[freq_idx]
        bandpower_values = np.transpose(bandpower_values, [2, 1, 0])
        bandpower_values = np.sum(bandpower_values, axis=-1)  # Sum the extracted bandpower values for every channel
        return bandpower_values

    # Computes the rate of sharpest change of every EEG segment over every channel for a batch of EEG data
    # Inputs
    #   input_data: EEG data of shape N x C x S, where N is the number of EEG segments from
    #               a patient's dataset, C is the number of valid EEG channels and S is the
    #               number of samples within the EEG segment
    #   fs: sampling frequency
    #   window_size: size of the window to compute differences over
    # Outputs
    #   output_diff: an array of shape N x C that stores the sharpest transition for every segment & channel
    @staticmethod
    def diff_signal(input_data, fs, window_size=0.03):
        output_diff = None
        # Search over all possible differences
        for num in range(int(fs * window_size)):
            input_diff = np.amax(np.diff(input_data, n=num, axis=-1), axis=-1)
            if output_diff is None:
                output_diff = np.expand_dims(input_diff, axis=0)
            else:
                output_diff = np.r_[output_diff, np.expand_dims(input_diff, axis=0)]
        output_diff = np.amax(output_diff, axis=0)
        return output_diff

    # Computes the envelope of every EEG segment over every channel for a batch of EEG data
    # Inputs
    #   input_data: EEG data of shape N x C x S, where N is the number of EEG segments from
    #               a patient's dataset, C is the number of valid EEG channels and S is the
    #               number of samples within the EEG segment
    # Outputs
    #   envelope_values: an array of shape N x C that stores the median envelope of every segment & channel
    @staticmethod
    def envelope(input_data):
        analytic_signal = hilbert(input_data, axis=-1)
        envelope_output = np.abs(analytic_signal)
        envelope_values = np.median(envelope_output, axis=-1)
        return envelope_values
