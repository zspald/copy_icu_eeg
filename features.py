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
    #   output_feats: an array of shape N x C x F, where N is the number of EEG segments,
    #                 C is the number of channels and F is the number of features
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

    # Computes a time-frequency representation of an EEG signal over multiple channels
    # by using the continuous wavelet transform
    # Inputs
    #   input_data: EEG data of shape N x C x S, where N is the number of EEG segments from
    #               a patient's dataset, C is the number of valid EEG channels and S is the
    #               number of samples within the EEG segment
    #   num_scales: the number of scales to be used in the wavelet transform. Note that lower scales
    #               correspond to higher frequencies and vice versa
    #   downsample_factor: a factor that indicates the extent of downsampling. For example, a 64 x 64
    #                      matrix with downsampling factor of 4 becomes a 16 x 16 matrix
    #   method: the method to use for compressing the image data
    #           'both' computes both images obtained by the minmax and average methods described below
    #           'minmax' computes the range of the values within each downsampling window
    #           'average' computes the mean of the values within each downsampling window
    # Outputs
    #   img_outputs: a multidimensional array of shape (2) x N x C x H x W, where H, W corresponds to
    #                the height and width of the new input (N, C are the same as the input).
    #                Contains the time-frequency representation of the EEG data
    @staticmethod
    def wavelet_image(input_data, num_scales, downsample_factor=2, method='minmax'):
        # Warn the user if the specified imaging method is not valid
        if method != 'minmax' and method != 'average' and method != 'both':
            print('Invalid method input detected. Please use either minmax or average')
            return None
        downsample_factor = int(downsample_factor)
        # Define scale and perform the continuous wavelet transform on the input dataset
        scale = np.arange(1, num_scales + 1)
        cwt_outputs, freqs = pywt.cwt(input_data, scale, wavelet='morl', axis=-1)
        # Output of pywt.cwt is of shape (# scales x # samples x # channels x # datapoints)
        spacing = int(np.size(input_data, axis=-1) / len(scale))
        cwt_outputs = cwt_outputs[:, :, :, 0:np.size(cwt_outputs, axis=-1):spacing]
        cwt_outputs = np.transpose(cwt_outputs, [1, 2, 0, 3])
        new_width = math.ceil(np.size(cwt_outputs, axis=2) / downsample_factor)
        new_length = math.ceil(np.size(cwt_outputs, axis=3) / downsample_factor)
        if method == 'both':
            img_outputs = (
                np.zeros((np.size(cwt_outputs, axis=0), np.size(cwt_outputs, axis=1), new_width, new_length)),
                np.zeros((np.size(cwt_outputs, axis=0), np.size(cwt_outputs, axis=1), new_width, new_length)))
        else:
            img_outputs = np.zeros((np.size(cwt_outputs, axis=0), np.size(cwt_outputs, axis=1),
                                    new_width, new_length))
        # Downsample the data
        for ii in range(np.size(cwt_outputs, axis=2)):
            for jj in range(np.size(cwt_outputs, axis=3)):
                if (ii % downsample_factor == 0) and (jj % downsample_factor == 0):
                    window_x = 0
                    window_y = 0
                    # Determine the range of the window
                    while window_x < downsample_factor and ii + window_x < np.size(cwt_outputs, axis=2) - 1:
                        window_x += 1
                    while window_y < downsample_factor and jj + window_y < np.size(cwt_outputs, axis=3) - 1:
                        window_y += 1
                    window_x = max([window_x, 1])
                    window_y = max([window_y, 1])
                    # Apply either the minimum-maximum method or the averaging method
                    if method == 'minmax':
                        img_outputs[:, :, int(ii / downsample_factor), int(jj / downsample_factor)] = \
                            np.amax(np.amax(cwt_outputs[:, :, ii:ii + window_x, jj:jj + window_y], axis=-1), axis=-1) - \
                            np.amin(np.amin(cwt_outputs[:, :, ii:ii + window_x, jj:jj + window_y], axis=-1), axis=-1)
                    elif method == 'average':
                        img_outputs[:, :, int(ii / downsample_factor), int(jj / downsample_factor)] = \
                            np.mean(np.mean(cwt_outputs[:, :, ii:ii + window_x, jj:jj + window_y], axis=-1), axis=-1)
                    else:
                        img_outputs[0][:, :, int(ii / downsample_factor), int(jj / downsample_factor)] = \
                            np.amax(np.amax(cwt_outputs[:, :, ii:ii + window_x, jj:jj + window_y], axis=-1), axis=-1) - \
                            np.amin(np.amin(cwt_outputs[:, :, ii:ii + window_x, jj:jj + window_y], axis=-1), axis=-1)
                        img_outputs[1][:, :, int(ii / downsample_factor), int(jj / downsample_factor)] = \
                            np.mean(np.mean(cwt_outputs[:, :, ii:ii + window_x, jj:jj + window_y], axis=-1), axis=-1)
        # Normalize the data (0 to 1)
        for ii in range(np.size(img_outputs, axis=-4)):
            for jj in range(np.size(img_outputs, axis=-3)):
                if method == 'both':
                    img_outputs[0][ii][jj] = (img_outputs[0][ii][jj] - np.amin(img_outputs[0][ii][jj])) / \
                                             (np.amax(img_outputs[0][ii][jj]) - np.amin(img_outputs[0][ii][jj]))
                    img_outputs[1][ii][jj] = (img_outputs[1][ii][jj] - np.amin(img_outputs[1][ii][jj])) / \
                                             (np.amax(img_outputs[1][ii][jj]) - np.amin(img_outputs[1][ii][jj]))
                else:
                    img_outputs[ii][jj] = (img_outputs[ii][jj] - np.amin(img_outputs[ii][jj])) / \
                                          (np.amax(img_outputs[ii][jj]) - np.amin(img_outputs[ii][jj]))
        return img_outputs
