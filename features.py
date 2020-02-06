############################################################################
# Extracts statistical features from EEG data in time/frequency domains
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
import numpy as np
import scipy.stats
from scipy.signal import hilbert


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
    # Outputs
    #   output_feats: an array of shape N x F, where N is the number of EEG segments
    #                 and F is the number of features
    @staticmethod
    def extract_features(input_data, fs, normalize='off'):
        # Line length
        llength = EEGFeatures.line_length(input_data)
        # Delta bandpower
        bdpower_delta = EEGFeatures.bandpower(input_data, fs, 0.5, 4)
        # Theta bandpower
        bdpower_theta = EEGFeatures.bandpower(input_data, fs, 4, 8)
        # Alpha bandpower
        bdpower_alpha = EEGFeatures.bandpower(input_data, fs, 8, 12)
        # Beta bandpower
        bdpower_beta = EEGFeatures.bandpower(input_data, fs, 12, 25)
        # Skewness
        skew = scipy.stats.skew(input_data, axis=-1)
        # Kurtosis
        kurt = scipy.stats.kurtosis(input_data, axis=-1)
        # Envelope
        envp = EEGFeatures.envelope(input_data)
        # Obtain all features
        output_feats = np.array([llength, bdpower_delta, bdpower_theta, bdpower_alpha,
                                 bdpower_beta, skew, kurt, envp])
        output_feats = np.swapaxes(np.swapaxes(output_feats, 0, 1), 1, 2)
        # Normalize the features if a specific method is indicated
        if normalize != 'off':
            output_feats = (output_feats - np.amin(output_feats, axis=0)) / \
                           (np.amax(output_feats, axis=0) - np.amin(output_feats, axis=0))
            # Normalize according to z-score
            if normalize == 'zscore':
                output_feats = scipy.stats.zscore(output_feats, axis=0)
            # Normalize from -1 to +1
            elif normalize == 'minmax':
                output_feats = output_feats * 2 - 1
        return output_feats

    # Computes the line length of every EEG segment over every channel for a batch of EEG data
    # Inputs
    #   input_data: EEG data of shape N x C x S, where N is the number of EEG segments from
    #               a patient's dataset, C is the number of valid EEG channels and S is the
    #               number of samples within the EEG segment
    # Outputs
    #   llength: an array of shape N x C containing the line length of each segment & channel
    @staticmethod
    def line_length(input_data):
        llength = np.abs(np.diff(input_data, axis=2))
        llength = np.sum(llength, axis=2)
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

    # Computes the envelope of every EEG segment over every channel for a batch of EEG data
    # Inputs
    #   input_data: EEG data of shape N x C x S, where N is the number of EEG segments from
    #               a patient's dataset, C is the number of valid EEG channels and S is the
    #               number of samples within the EEG segment
    # Outputs
    #   envelope_values: an array of shape N x C that stores the median envelope of each segment & channel
    @staticmethod
    def envelope(input_data):
        analytic_signal = hilbert(input_data, axis=-1)
        envelope_output = np.abs(analytic_signal)
        envelope_values = np.median(envelope_output, axis=-1)
        return envelope_values
