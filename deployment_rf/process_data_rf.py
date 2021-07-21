############################################################################
# Preprocesses EEG data, annotations from ICU patients at HUP
# Deployment version of preprocess_dataset.py
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
from artifacts_rf import Artifacts
from features_rf import EEGFeatures, EEG_FEATS
from load_data_rf import IEEGDataLoader
from scipy.signal import butter, filtfilt
import math
import numpy as np

# Artifact rejection method
ARTIFACT_METHOD = 'threshold'


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
        self.mean = None
        self.std = None

    # Generates a map of EEG feature distribution over the brain surface
    # Inputs
    #   num_batches: number of EEG segments
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   initial_pass: whether the feature extraction process is in the initial phase
    # Outputs
    #   map_outputs: a multidimensional array of feature maps with shape (N* x F x W x H), where
    #                N* and F share the same definitions as above and W, H denote the image size
    #   map_indices: a list indicating whether each EEG segment has been used, with length N
    # def generate_map(self, num_batches, start, length, initial_pass=False):
    #     # Process and return features for user-designated EEG intervals
    #     patient_feats, patient_indices, patient_channels = self.get_features(num_batches, start, length)
    #     # Check whether features are entirely artifacts
    #     if patient_feats is None:
    #         return None, None
    #     # Save patient-specific EEG statistics and apply normalization
    #     if initial_pass:
    #         self.mean, self.std = EEGFeatures.compute_stats(patient_feats)
    #     patient_feats = (patient_feats - self.mean) / self.std
    #     # Compute map renderings of the features across the patient's scalp
    #     map_outputs = EEGMap.generate_map(patient_feats, patient_channels)
    #     map_indices = 1 - patient_indices
    #     return map_outputs, map_indices

    # replacement of generate_map for random_forest model (method header TODO)
    def process_feats(self, num, start, length, bipolar=False):
        patient_feats, patient_indices, _ = self.get_features(num, start, length, bipolar=bipolar)
        
        if patient_feats is None:
            return None, None
        
        patient_feats = (patient_feats - self.mean) / self.std

        feat_indices = 1 - patient_indices
        
        return patient_feats, feat_indices

    # Pre-computes the patient-specific EEG statistics with the given set of EEG features
    # Inputs
    #   num_batches: number of EEG segments
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    # Outputs
    #   self.mean: the mean of all channels and features, with shape C x F as described above
    #   self.std: the standard deviation of all channels and features, with shape C x F as described above
    def initialize_stats(self, num_batches, start, length):
        patient_feats, _, _ = self.get_features(num_batches, start, length)
        self.mean, self.std = EEGFeatures.compute_stats(patient_feats)
        return self.mean, self.std

    # Extracts features for a specified interval of EEG recordings from IEEG
    # Inputs
    #   num: number of EEG batches to process
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    # Outputs
    #   output_feats: a feature array that contains features for all EEG segments across all channels
    #                 with shape N* x C x F, where N* is the number of clean EEG segments from the
    #                 patient's dataset, C is the number of channels and F is the number of features
    #   indices_to_remove: a list indicating whether each EEG segment should be removed, with length N
    #   channels_to_remove: a N* x C array that indicates whether each channel in each segment should be removed
    #   returns None if no preprocessed data is available
    def get_features(self, num, start, length, bipolar=False):
        output_data, indices_to_remove, channels_to_remove = self.process_data(num, start, length)
        fs = self.sampling_frequency()
        if output_data is None:
            return None, None, None
        output_feats = EEGFeatures.extract_features(output_data, fs, pool_region=False, bipolar=bipolar)
        return output_feats, indices_to_remove, channels_to_remove

    # Processes data from IEEG.org in multiple batches
    # Inputs
    #   num: number of EEG batches to process
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    # Outputs
    #   output_data: a set of processed EEG segments with shape N* x C x D, where N* is the number
    #                of clean EEG segments from the patient's dataset, C is the number of clean
    #                channels and D is the number of samples within each segment (S = fs * s_length)
    #   indices_to_remove: a list indicating whether each EEG segment should be removed, with length N
    #   channels_to_remove: a N* x C array that indicates whether each channel in each segment should be removed
    def process_data(self, num, start, length):
        # Load raw data and labels
        input_data = IEEGDataLoader.load_data_batch(self, num, start, length, eeg_only=True)
        input_data = np.swapaxes(input_data, 1, 2)
        fs = IEEGDataLoader.sampling_frequency(self)
        # Filter from 0.5 to 20 Hz
        coeff = butter(4, [0.5 / (fs / 2), 20 / (fs / 2)], 'bandpass')
        input_data = filtfilt(coeff[0], coeff[1], input_data, axis=-1)
        # Perform artifact rejection over the input data
        output_data, indices_to_remove, channels_to_remove = self.clean_data(input_data, fs, artifact=ARTIFACT_METHOD)
        # Postprocess data structure for channels removed
        if channels_to_remove is not None:
            channels_to_remove = np.array([channels_to_remove[ii] for ii in range(np.size(channels_to_remove, axis=0))
                                           if indices_to_remove[ii] == 0])
        return output_data, indices_to_remove, channels_to_remove

    # Performs artifact rejection over all EEG recordings within the given dataset
    # and updates the annotation file in accordance with the processed EEG data
    # Inputs
    #   input_data: EEG sample of shape N x C x D, where N is the number of EEG batches, C is the number of
    #               valid channels and D is the total number of samples within the given patient's EEG
    #   fs: sampling frequency of the patient's EEG recording
    #   artifact: specific method for removing artifact-like segments.
    #             'stats' - removes all segments with z-scores above 5 under
    #             1) Range 2) Line Length 3) Bandpower in beta frequency band (12 - 20 Hz)
    #             'threshold' - removes all segments that exceed/fall below certain thresholds under
    #             1) Variance 2) Range 3) Line Length 4) Bandpower 5) Signal difference
    # Outputs
    #   output_data: a set of processed EEG segments with shape N* x C x D, where N* is the number
    #                of valid EEG segments from the patient's dataset, C is the number of valid
    #                channels and D is the number of samples within each segment (S = fs * s_length)
    #   indices_to_remove: a list indicating whether each EEG segment should be removed, with length N
    #   channels_to_remove: a N x C array that indicates whether each channel in each segment should be removed
    #   returns None if no preprocessed data is available
    @staticmethod
    def clean_data(input_data, fs, artifact='threshold'):
        # Perform artifact rejection
        indices_to_remove, channels_to_remove = Artifacts.remove_artifacts(input_data, fs, channel_limit=6,
                                                                           method=artifact)
        # Remove cross-channel artifact data
        indices_to_keep = (1 - indices_to_remove).astype('float')
        indices_to_keep[indices_to_keep == 0] = np.nan
        output_data = indices_to_keep[:, None, None] * input_data
        # Remove channel-specific artifact data
        channels_to_keep = (1 - channels_to_remove).astype('float')
        channels_to_keep[channels_to_keep == 0] = np.nan
        output_data = np.expand_dims(channels_to_keep, axis=-1) * output_data
        # Return None if output data only contains NaNs
        if np.isnan(indices_to_keep).all():
            return None, None, None
        # Remove artifact portions of the EEG data
        output_data = output_data[(1-indices_to_remove).astype(bool), :, :]
        return output_data, indices_to_remove, channels_to_remove

    # Post-processes seizure predictions obtained from the model using a sliding window
    # Inputs
    #   predictions: a list of seizure predictions given as a 1D numpy array
    #   length: the length of every EEG sample
    #   sz_length: the minimum length of a typical seizure
    #   threshold: the minimum value for accepting a prediction as seizure
    # Outputs
    #   predictions_outputs: a list of post-processed seizure predictions
    @staticmethod
    def postprocess_outputs(predictions, length, sz_length=30, threshold=0.8):
        # Initialize outputs and parameters for the sliding window
        predictions_outputs = np.array([pred for pred in predictions])
        window_size = int(sz_length / length)
        window_disp = int(window_size / 3)
        window_pos = 0
        # Slide the window and fill in regions frequently predicted as seizure
        while window_pos + window_size <= np.size(predictions_outputs, axis=0):
            if np.sum(predictions[window_pos:window_pos + window_size]) >= window_size * threshold:
                predictions_outputs[window_pos:window_pos + window_size] = 1
            window_pos += window_disp
        # Initialize a smaller window to be run over the predictions
        window_size = int(window_size / 6)
        window_pos = 0
        # Slide another window and remove outlying predictions
        while window_pos < np.size(predictions_outputs, axis=0):
            if window_pos < window_size and np.sum(predictions[:window_pos]) <= window_pos * threshold:
                predictions_outputs[:window_pos] = 0
            elif window_pos > np.size(predictions_outputs, axis=0) - window_size and \
                    np.sum(predictions[window_pos:]) <= len(predictions[window_pos:]) * threshold:
                predictions_outputs[window_pos:] = 0
            elif np.sum(predictions[window_pos - window_size:window_pos + window_size]) <= 2 * window_size * threshold:
                predictions_outputs[window_pos - window_size:window_pos + window_size] = 0
            window_pos += window_size
        return predictions_outputs

    # Fills seizure detection outputs by accounting for segments removed from artifacts
    # Inputs
    #   predictions: a list of seizure predictions given as a 1D numpy array, with length N*
    #   indices_to_use: a list indicating whether each EEG segment has been used, with length N
    # Outputs
    #   predictions_fill: a list of seizure predictions given as a 1D numpy array, with length N
    #                     predictions for artifact segments given as np.nan
    @staticmethod
    def fill_predictions(predictions, indices_to_use):
        predictions_fill = np.zeros(len(indices_to_use))
        count = 0
        # Iterate over the indices and check for artifacts
        for idx, elem in enumerate(indices_to_use):
            if elem == 0:
                predictions_fill[idx] = np.nan
            else:
                predictions_fill[idx] = predictions[count]
                count += 1
        return predictions_fill

    # Saves the prediction outputs into a set of lists that contains onset/end info of EEG events
    # Inputs
    #   predictions: a list of seizure predictions given as a 1D numpy array
    #   timepoint: the reference timepoint at the onset of the predictions, in seconds
    #   length: the length of each EEG segment, in seconds
    #   include_artifact: whether to save artifact timepoints
    # Outputs
    #   event_list: a list of lists that contains the triplet 'event-type', 'onset time' and 'end time'
    @staticmethod
    def write_events(predictions, timepoint, length=1, include_artifact=False):
        event_list = list()
        event_start, event_end = 0, 0
        prev_output = 0
        predictions = np.append(predictions, np.array([0]))
        # Iterate over the predictions of the model and
        for idx, pred in enumerate(predictions):
            if pred == 1 and prev_output != 1:
                if include_artifact and math.isnan(prev_output):
                    event_end = timepoint + idx * length
                    event_list.append(['artifact', event_start, event_end])
                event_start = timepoint + idx * length
                prev_output = pred
            elif pred != 1 and prev_output == 1:
                event_end = timepoint + idx * length
                event_list.append(['seizure', event_start, event_end])
                prev_output = pred
                if include_artifact and math.isnan(pred):
                    event_start = timepoint + idx * length
            elif include_artifact and math.isnan(pred) and not math.isnan(prev_output):
                event_start = timepoint + idx * length
                prev_output = pred
            elif include_artifact and not math.isnan(pred) and math.isnan(prev_output):
                event_end = timepoint + idx * length
                event_list.append(['artifact', event_start, event_end])
                prev_output = pred
        return event_list
