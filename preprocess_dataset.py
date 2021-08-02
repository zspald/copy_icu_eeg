############################################################################
# Preprocesses EEG data, annotations from ICU patients at HUP
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
from random import random
from artifacts import Artifacts
from features import EEGFeatures, EEG_FEATS
from features_2d import EEGMap
from load_dataset import IEEGDataLoader
from scipy.signal import butter, filtfilt
import h5py
import numpy as np
import pandas as pd
import pickle

# Artifact rejection method
ARTIFACT_METHOD = 'threshold'

# Whether to use file for annots or not
USE_FILE_ANNOTS = True

# A class that preprocesses IEEG data for the specified patient
# Inherits fields and methods from the IEEGDataLoader class
class IEEGDataProcessor(IEEGDataLoader):

    # The constructor for the IEEGDataProcessor class
    # Attributes
    #   normal_count: the number of non-seizure samples accepted
    #   seizure_count: the number of seizure samples accepted
    #   sz_intervals: a numpy array that contains pairs of start/stop times of
    #                 distinct seizures from the EEG recording
    def __init__(self, dataset_id, user, pwd):
        super().__init__(dataset_id, user, pwd)
        self.normal_count = 0
        self.seizure_count = 0
        self.sz_intervals = None

    # Processes features over a multiple number of iterations to reduce storage space
    # Inputs
    #   num_iter: total number of iterations over the patient data
    #   num_batches: number of EEG segments within each iteration
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   use_filter: whether to apply a bandpass filter to the EEG
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    #   normalize: normalization method to be used for the extracted features
    #              'default' normalizes features from 0 to 1
    #              'zscore' normalizes features according to their z-scores
    #              'minmax' normalizes features from -1 to 1
    # Outputs
    #   patient_feats: a feature array that contains features for all EEG segments across all channels
    #                 with shape N* x C x F, where N* is the number of clean EEG segments from the
    #                 patient's dataset, C is the number of channels and F is the number of features
    #   patient_labels: a modified list of seizure annotations of the given patient with length N*
    #   returns None if no preprocessed data is available
    def process_all_feats(self, num_iter, num_batches, start, length, use_filter=True, eeg_only=True, 
    normalize=None, log_artifacts=False, bipolar=False, random_forest=False, use_label_times=True, pool=False):
        # Determine channels to be used
        channels_to_use = self.filter_channels(eeg_only)
        chan_length = len(channels_to_use)
        
        # Create a gzipped HDF file to store the processed features for specific montage type
        filename = "data/%s_data_wt" % self.id
        if bipolar:
            print("Using bipolar montage (double banana)")
            filename += "_bipolar"
            chan_length = 18
        else:
            print("Using referential montage")
        if pool:
            print("Pooling data by region")
            filename += "_pool"
            if bipolar:
                chan_length = 4 
            else:
                chan_length = 3
        if random_forest:
            print("Saving data in random forest format")
            filename += "_rf"
        filename += ".h5"

        file = h5py.File(filename, 'w')
        
        patient_feats = file.create_dataset('feats', (0, chan_length, len(EEG_FEATS)),
                                            maxshape=(None, chan_length, len(EEG_FEATS)),
                                            compression='gzip', chunks=True)
        patient_labels = file.create_dataset('labels', (0, 3), maxshape=(None, 3), chunks=True)
        patient_channels = file.create_dataset('channels', (0, len(channels_to_use)),
                                               maxshape=(None, len(channels_to_use)), chunks=True)

        #Create dataframe to log artifact rejection criteria
        rejection_log_df = pd.DataFrame(columns=['Iteration', 'Segment', 'NaN', 'Variance', 'Minmax', 'Line Length',
                                                 'Band Power', 'Signal Diff'])

        # Find the starting point for extracting EEG features
        if use_label_times:
            print('Using label times')
            f_pkl = open("dataset/patient_start_stop.pkl", 'rb')
            start_stop_df = pickle.load(f_pkl)
            patient_times = start_stop_df[start_stop_df['patient_id'] == self.id].values
            start = patient_times[0,1]
            stop = patient_times[-1,2]
            num_iter = int(np.floor((stop - start) / num_batches))
        start_origin = self.crawl_data(start, interval_length=600, threshold=1e4, channels_to_use=channels_to_use)
        if start_origin is None:
            print("Patient contains NaN recordings for over three hours!")
            return None, None, None
        start = start_origin
        print("The starting point is: %d seconds" % start)
        # Extract all seizure intervals of the patient, and check if there are any
        self.sz_intervals = self.load_annots_source(use_file=USE_FILE_ANNOTS)
        has_seizure = np.size(self.sz_intervals, axis=0) > 0
        # Proceed with seizure feature extraction if the patient has seizure
        if has_seizure:
            print("Extracting seizure data")
            # Iterate over all batches and extract seizure data
            for idx in range(num_iter):
                print("=====Iteration %d=====" % (idx + 1))
                # Extract seizure features using the IEEGDataProcessor object
                feats, labels, channels_to_remove, rejection_log_df = self.get_features(num_batches, start, length, idx + 1, rejection_log_df, use_filter=use_filter,
                                                                      method='sz', bipolar=bipolar, pool_region=pool)
                start += num_batches * length
                # Save the features, labels and channel info into the HDF file
                if feats is not None:
                    num_feats = np.size(feats, axis=0)
                    patient_feats.resize((patient_feats.shape[0] + num_feats, chan_length, len(EEG_FEATS)))
                    patient_feats[-num_feats:, :, :] = feats
                    patient_labels.resize((patient_labels.shape[0] + num_feats, 3))
                    patient_labels[-num_feats:, :] = labels
                    patient_channels.resize((np.size(patient_channels, axis=0) + num_feats, len(channels_to_use)))
                    patient_channels[-num_feats:, :] = channels_to_remove
                else:
                    print("No seizure data is available from batch #%d" % (idx + 1))
            print(f"Finished at {start}")
        else:
            print("No seziure data in considered patient range")
        # Set non-seizure data extraction method based on seizure occurrence
        method = 'norm-sz' if has_seizure else 'norm'
        start = start_origin
        print("Extracting non-seizure data")
        # Iterate over all batches and extract non-seizure data
        for idx in range(num_iter):
            print("=====Iteration %d=====" % (idx + 1))
            # Extract non-seizure features using the IEEGDataProcessor object
            feats, labels, channels_to_remove, rejection_log_df = self.get_features(num_batches, start, length, idx + num_iter + 1, rejection_log_df, use_filter=use_filter,
                                                                  eeg_only=eeg_only, method=method, bipolar=bipolar, pool_region=pool)
            start += num_batches * length
            # Add batch features and labels to patient-specific outputs
            if feats is not None:
                num_feats = np.size(feats, axis=0)
                patient_feats.resize((patient_feats.shape[0] + num_feats, chan_length, len(EEG_FEATS)))
                patient_feats[-num_feats:, :, :] = feats
                patient_labels.resize((patient_labels.shape[0] + num_feats, 3))
                patient_labels[-num_feats:, :] = labels
                patient_channels.resize((np.size(patient_channels, axis=0) + num_feats, len(channels_to_use)))
                patient_channels[-num_feats:, :] = channels_to_remove
            else:
                print("No data is available from batch #%d" % (idx + 1))

        print(f"Finished at {start}")
        #save artifact rejection log to file
        if log_artifacts:
            Artifacts.save_rejection_log(rejection_log_df, self.id, visualize=True)

        # Normalize patient data if indicated to do so
        if normalize:
            try:  # Normalize all patient data if possible
                patient_feats[:, :, :] = EEGFeatures.normalize_feats(patient_feats, option='zscore')
            except MemoryError:  # Normalize in hour-long intervals
                hour = 3600
                # Iterate over all hour-long batches of EEG features
                for idx in range(np.ceil(patient_feats.shape[0] / hour)):
                    hour_batch = patient_feats[idx * hour:min((idx + 1) * hour, patient_feats.shape[0]), :, :]
                    patient_feats[idx * hour:min((idx + 1) * hour, patient_feats.shape[0]), :, :] = \
                        EEGFeatures.normalize_feats(hour_batch, option='zscore')
        # Close the file and return outputs
        print("Shape of the features: ", patient_feats.shape)
        return patient_feats, patient_labels, patient_channels

    # Generates a map of EEG feature distribution over the brain surface
    # Inputs
    #   num_iter: total number of iterations over the patient data
    #   num_batches: number of EEG segments within each iteration
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   use_filter: whether to apply a bandpass filter to the EEG
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    #   normalize: whether to normalize the feature inputs for map generation
    # Outputs
    #   map_outputs: a multidimensional array of feature maps with shape (N* x F x W x H), where
    #                N* and F share the same definitions as above and W, H denote the image size
    def generate_map(self, num_iter, num_batches, start, length, use_filter=True, eeg_only=True, normalize=False, bipolar=False):
        # Indicate default normalization method
        normalize = 'zscore' if normalize else None
        # Process and return features for user-designated EEG intervals
        patient_feats, _, patient_channels = self.process_all_feats(num_iter, num_batches, start, length, use_filter,
                                                                    eeg_only, normalize, bipolar=bipolar)
        # Compute map renderings of the features across the patient's scalp
        map_outputs = EEGMap.generate_map(self.id, patient_feats, patient_channels, bipolar=bipolar)
        print("Shape of the feature maps: ", map_outputs.shape)
        return map_outputs

    # Extracts features for a specified interval of EEG recordings from IEEG
    # Inputs
    #   num: number of EEG batches to process
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   norm: normalization method to be used for the extracted features
    #         'off' does not perform any normalization
    #         'default' normalizes features from 0 to 1
    #         'zscore' normalizes features according to their z-scores
    #         'minmax' normalizes features from -1 to 1
    #   use_filter: whether to apply a bandpass filter to the EEG
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    #   method: refer to header for *clean_data* below
    # Outputs
    #   output_feats: a feature array that contains features for all EEG segments across all channels
    #                 with shape N* x C x F, where N* is the number of clean EEG segments from the
    #                 patient's dataset, C is the number of channels and F is the number of features
    #   output_labels: a modified list of seizure annotations of the given patient with length N*
    #   channels_to_remove: a N x C array that indicates whether each channel in each segment should be removed
    #   returns None if no preprocessed data is available
    def get_features(self, num, start, length, curr_iter, rejection_log_df, norm='off', use_filter=True, eeg_only=True, method=None, bipolar=False, pool_region=False):
        output_data, output_labels, _, channels_to_remove, rejection_log_df = self.process_data(num, start, length, curr_iter, rejection_log_df, use_filter, eeg_only,
                                                                              save_artifacts=False, method=method)
        fs = self.sampling_frequency()
        if output_data is None:
            return None, None, None, rejection_log_df
        output_feats = EEGFeatures.extract_features(output_data, fs, normalize=norm, pool_region=pool_region, bipolar=bipolar)
        return output_feats, output_labels, channels_to_remove, rejection_log_df

    # Processes data from IEEG.org in multiple batches
    # Inputs
    #   num: number of EEG batches to process
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   use_filter: whether to apply a bandpass filter to the EEG
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    #   save_artifacts: whether to save timepoints of EEG artifacts
    #   method: refer to header for *clean_data* below
    # Outputs
    #   output_data: a set of processed EEG segments with shape N* x C x D, where N* is the number
    #                of clean EEG segments from the patient's dataset, C is the number of clean
    #                channels and D is the number of samples within each segment (S = fs * s_length)
    #   output_labels: a modified list of seizure annotations of the given patient with length N*
    #   indices_to_remove: a list indicating whether each EEG segment should be removed, with length N
    #   channels_to_remove: a N* x C array that indicates whether each channel in each segment should be removed
    def process_data(self, num, start, length, curr_iter, rejection_log_df, use_filter=True, eeg_only=True, save_artifacts=False, method=None):
        # Load raw data and labels
        input_data = IEEGDataLoader.load_data_batch(self, num, start, length, eeg_only)
        input_data = np.swapaxes(input_data, 1, 2)
        input_labels = IEEGDataLoader.load_annots(self, num, start, length, use_file=USE_FILE_ANNOTS)
        fs = IEEGDataLoader.sampling_frequency(self)
        # Filter from 0.5 to 20 Hz if selected
        if use_filter:
            coeff = butter(4, [0.5 / (fs / 2), 20 / (fs / 2)], 'bandpass')
            input_data = filtfilt(coeff[0], coeff[1], input_data, axis=-1)
        # Perform artifact rejection over the input data
        output_data, output_labels, indices_to_remove, channels_to_remove, rejection_log_df = \
            self.clean_data(input_data, input_labels, fs, start, length, curr_iter, rejection_log_df, artifact=ARTIFACT_METHOD, method=method)
        # Postprocess data structure for channels removed
        if channels_to_remove is not None:
            channels_to_remove = np.array([channels_to_remove[ii] for ii in range(np.size(channels_to_remove, axis=0))
                                           if indices_to_remove[ii] == 0])
        # Save artifact information if required
        if save_artifacts:
            Artifacts.save_artifacts(self.id, indices_to_remove, start, length)
        return output_data, output_labels, indices_to_remove, channels_to_remove, rejection_log_df

    # Performs artifact rejection over all EEG recordings within the given dataset
    # and updates the annotation file in accordance with the processed EEG data
    # Inputs
    #   input_data: EEG sample of shape N x C x D, where N is the number of EEG batches, C is the number of
    #               valid channels and D is the total number of samples within the given patient's EEG
    #   input_labels: a list of seizure annotations of the given patient for every segment, with length N
    #   fs: sampling frequency of the patient's EEG recording
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   artifact: specific method for removing artifact-like segments.
    #             'stats' - removes all segments with z-scores above 5 under
    #             1) Range 2) Line Length 3) Bandpower in beta frequency band (12 - 20 Hz)
    #             'threshold' - removes all segments that exceed/fall below certain thresholds under
    #             1) Variance 2) Range 3) Line Length 4) Bandpower 5) Signal difference
    #   method: the method for extracting samples
    #           'none' - does not perform any additional check, sampling all seizure/non-seizure segments
    #           'sz' - only extracts seizure samples, removing all non-seizure samples
    #           'norm' - only extracts normal samples, removing all seizure samples
    #           'norm-sz' - only extracts normal samples with a count limit
    # Outputs
    #   output_data: a set of processed EEG segments with shape N* x C x D, where N* is the number
    #                of valid EEG segments from the patient's dataset, C is the number of valid
    #                channels and D is the number of samples within each segment (S = fs * s_length)
    #   output_labels: a modified list of seizure annotations of the given patient with length N*
    #   indices_to_remove: a list indicating whether each EEG segment should be removed, with length N
    #   channels_to_remove: a N x C array that indicates whether each channel in each segment should be removed
    #   returns None if no preprocessed data is available
    def clean_data(self, input_data, input_labels, fs, start, length, curr_iter, rejection_log_df, artifact='threshold', method='none'):
        # Perform artifact rejection
        indices_to_remove, channels_to_remove, rejection_log_df = Artifacts.remove_artifacts(input_data, fs, curr_iter, rejection_log_df, channel_limit=6,
                                                                           method=artifact)
        timepoints = [start + idx * length for idx in range(len(indices_to_remove))]
        # Remove non-seizure segments if indicated by the user
        if method == 'sz':
            indices_to_remove[input_labels[:, 0] == 0] = 1
            self.seizure_count += np.count_nonzero(np.multiply(input_labels[:, 0], 1 - indices_to_remove))
            print('Number of seizures: ', np.count_nonzero(input_labels[:, 0]))
        # Remove seizure segments and extract normal segments if indicated by the user
        elif method == 'norm':
            indices_to_remove[input_labels[:, 0] == 1] = 1
            indices_to_remove = self.extract_norm(indices_to_remove, timepoints, balance=False)
        # Remove seizure segments and extract normal segments constraining the amount of non-seizure segments added,
        # if indicated by the user
        elif method == 'norm-sz':
            indices_to_remove[input_labels[:, 0] == 1] = 1
            indices_to_remove = self.extract_norm(indices_to_remove, timepoints, balance=True)
        print('Number of accepted segments: ', np.sum(1 - indices_to_remove))
        print('Number of rejected segments: ', np.sum(indices_to_remove))
        # Remove cross-channel artifact data
        indices_to_keep = (1 - indices_to_remove).astype('float')
        indices_to_keep[indices_to_keep == 0] = np.nan
        output_data = indices_to_keep[:, None, None] * input_data # comment out for artifact clip visualization
        # Remove channel-specific artifact data
        channels_to_keep = (1 - channels_to_remove).astype('float')
        channels_to_keep[channels_to_keep == 0] = np.nan
        output_data = np.expand_dims(channels_to_keep, axis=-1) * output_data # comment out for artifact clip visualization
        # Return None if output data only contains NaNs
        if np.isnan(indices_to_keep).all():
            return None, None, None, None, rejection_log_df
        # Remove artifact portions of the EEG data
        output_data = output_data[(1-indices_to_remove).astype(bool), :, :] # comment out for artifact clip visualization
        # Update labels to match the size of the clean recordings
        output_labels = np.array([input_labels[idx] for idx, element in enumerate(indices_to_remove) if element == 0])
        return output_data, output_labels, indices_to_remove, channels_to_remove, rejection_log_df

    # Cherry-picks EEG non-seizure data that is appropriate for further use
    # Inputs
    #   indices_to_remove: a list indicating whether each EEG segment should be removed, with length N
    #   timepoints: a list of absolute timepoints for all EEG segments (in seconds), with length N
    #   balance: whether to balance the number of seizure and non-seizure data
    #   cont_length: the minimum length of contiguous non-seizure EEG segments, in seconds
    #   sep_length: the minimum length of separation between seizure and non-seizure EEG segments, in seconds
    #   ratio: the balance ratio between seizure and non-seizure data
    # Outputs
    #   indices_to_remove: a post-processed list indicating whether each EEG segment should be removed, with length N
    def extract_norm(self, indices_to_remove, timepoints, balance, cont_length=30, sep_length=1200, ratio=1.5):
        # Load seizure intervals and initialize parameters for the sliding window
        length = timepoints[1] - timepoints[0]
        window_size = int(cont_length / length)
        window_disp = int(window_size / 3)
        window_pos = 0
        # Check whether the given sequence of timepoints overlaps with a seizure interval
        if np.size(self.sz_intervals, axis=0) > 0 and self.search_interval(self.sz_intervals, timepoints[0], length):
            intervals = self.find_seizures(self.sz_intervals, timepoints, sep_length=sep_length)
            for interval in intervals:
                low = min(0, int((interval[0] - sep_length - timepoints[0]) / length))
                high = max(len(indices_to_remove) - 1, int((interval[1] + sep_length - timepoints[0]) / length))
                indices_to_remove[low:high] = 1
        # Slide the window and check whether each group of EEG samples are sufficiently contiguous
        time_limit = 10800
        while window_pos + window_size <= len(timepoints):
            checker = self.normal_count < int(self.seizure_count * ratio) if balance \
                else self.normal_count < int(time_limit / length)
            if np.sum(indices_to_remove[window_pos:window_pos + window_size]) == 0 and checker:
                self.normal_count += window_size
                window_pos += window_size
            else:
                indices_to_remove[window_pos:window_pos + window_disp] = 1
                window_pos += window_disp
                if window_pos + window_size > len(timepoints):
                    indices_to_remove[window_pos:] = 1
        return indices_to_remove

    # Finds all seizure intervals overlapping with the given sequence of timepoints
    # Inputs
    #   intervals: a numpy array of shape R x 2 where R is the number of seizure intervals.
    #              Each row contains the start and stop time, measured in seconds.
    #   timepoints: a list of absolute timepoints for all EEG segments (in seconds), with length N
    #   sep_length: the minimum length of separation between seizure and non-seizure EEG segments, in seconds
    # Outputs
    #   seizure_outputs: identical format to the input *intervals*, but only intervals overlapping with the
    #                    given sequence of timepoints
    @staticmethod
    def find_seizures(intervals, timepoints, sep_length=1200):
        seizure_outputs = []
        # Iterate over all seizure intervals in the annotations
        for interval in intervals:
            if interval[0] <= timepoints[0] + sep_length and timepoints[0] - sep_length <= interval[1]:
                seizure_outputs.append(interval)
        seizure_outputs = np.array(seizure_outputs)
        return seizure_outputs
