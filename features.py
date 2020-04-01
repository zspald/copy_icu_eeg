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
from load_dataset import EEG_CHANNELS


# EEG electrodes
ALL = EEG_CHANNELS
LEFT = ['C3', 'F3', 'F7', 'Fp1', 'O1', 'P3', 'T3', 'T5']
RIGHT = ['C4', 'F4', 'F8', 'Fp2', 'O2', 'P4', 'T4', 'T6']


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
        output_feats = (input_feats - np.amin(input_feats, axis=0)) / \
                       (np.amax(input_feats, axis=0) - np.amin(input_feats, axis=0))
        # Further normalize features based on user option
        if option == 'zscore':
            output_feats = scipy.stats.zscore(output_feats, axis=0)
        elif option == 'minmax':
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
