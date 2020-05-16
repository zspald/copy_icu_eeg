############################################################################
# Preprocesses EEG data, annotations from ICU patients at HUP
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
from artifacts import Artifacts
from features import EEGFeatures, EEG_FEATS
from features_2d import EEGMap
from load_dataset import IEEGDataLoader
from scipy.signal import butter, filtfilt
import h5py
import numpy as np

# Artifact rejection method
ARTIFACT_METHOD = 'threshold'


# A class that preprocesses IEEG data for the specified patient
# Inherits fields and methods from the IEEGDataLoader class
class IEEGDataProcessor(IEEGDataLoader):

    # The constructor for the IEEGDataProcessor class
    def __init__(self, dataset_id, user, pwd):
        super().__init__(dataset_id, user, pwd)

    # Processes features over a multiple number of iterations to reduce storage space
    # Inputs
    #   num_iter: total number of iterations over the patient data
    #   num_batches: number of EEG segments within each iteration
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   use_filter: whether to apply a bandpass filter to the EEG
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    # Outputs
    #   patient_feats: a feature array that contains features for all EEG segments across all channels
    #                 with shape N* x C x F, where N* is the number of clean EEG segments from the
    #                 patient's dataset, C is the number of channels and F is the number of features
    #   patient_labels: a modified list of seizure annotations of the given patient with length N*
    #   returns None if no preprocessed data is available
    def process_all_feats(self, num_iter, num_batches, start, length, use_filter=True, eeg_only=True):
        # Determine channels to be used
        channels_to_use = self.filter_channels(eeg_only)
        # Create a gzipped HDF file to store the processed features
        file = h5py.File('data/%s_data.h5' % self.id, 'w')
        patient_feats = file.create_dataset('feats', (0, len(channels_to_use), len(EEG_FEATS)),
                                            maxshape=(None, len(channels_to_use), len(EEG_FEATS)),
                                            compression='gzip', chunks=True)
        patient_labels = file.create_dataset('labels', (0, 3), maxshape=(None, 3), chunks=True)
        patient_channels = file.create_dataset('channels', (0, len(channels_to_use)),
                                               maxshape=(None, len(channels_to_use)), chunks=True)
        # Iterate over all batches
        for ii in range(num_iter):
            # Search for the first point (in seconds) where non-NaN values occur
            if ii == 0:
                start = self.crawl_data(start, interval_length=600, threshold=3600, channels_to_use=channels_to_use)
                if start is None:
                    print("Patient contains NaN recordings for over an hour!")
                    return
                print("The starting point is: %d seconds" % start)
            print("===Iteration %d===" % (ii + 1))
            # Extract features using the given IEEGDataProcessor object
            feats, labels, channels_to_remove = self.get_features(num_batches, start, length, norm='off',
                                                                  use_filter=use_filter, eeg_only=eeg_only)
            num_feats = np.size(feats, axis=0)
            start += num_batches * length
            # Add batch features and labels to patient-specific outputs
            if feats is not None:
                patient_feats.resize((np.size(patient_feats, axis=0) + num_feats, len(channels_to_use), len(EEG_FEATS)))
                patient_feats[-num_feats:, :, :] = feats
                patient_labels.resize((np.size(patient_labels, axis=0) + num_feats, 3))
                patient_labels[-num_feats:, :] = labels
                patient_channels.resize((np.size(patient_channels, axis=0) + num_feats, len(channels_to_use)))
                patient_channels[-num_feats:, :] = channels_to_remove
            else:
                print("No data is available from batch #%d" % (ii + 1))
        # Close the file and return outputs
        print("Shape of the features: ", np.shape(patient_feats))
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
    def generate_map(self, num_iter, num_batches, start, length, use_filter=True, eeg_only=True, normalize=False):
        # Process and return features for user-designated EEG intervals
        patient_feats, _, patient_channels = self.process_all_feats(num_iter, num_batches, start, length,
                                                                    use_filter, eeg_only)
        # Compute map renderings of the features across the patient's scalp
        map_outputs = EEGMap.generate_map(self.id, patient_feats, patient_channels, normalize=normalize)
        return map_outputs

    # Extracts features for a specified interval of EEG recordings from IEEG
    # Inputs
    #   num: number of EEG batches to process
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   use_filter: whether to apply a bandpass filter to the EEG
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    # Outputs
    #   output_feats: a feature array that contains features for all EEG segments across all channels
    #                 with shape N* x C x F, where N* is the number of clean EEG segments from the
    #                 patient's dataset, C is the number of channels and F is the number of features
    #   output_labels: a modified list of seizure annotations of the given patient with length N*
    #   channels_to_remove: a N x C array that indicates whether each channel in each segment should be removed
    #   returns None if no preprocessed data is available
    def get_features(self, num, start, length, norm='off', use_filter=True, eeg_only=True):
        output_data, output_labels, _, channels_to_remove = self.process_data(num, start, length, use_filter, eeg_only,
                                                                              save_artifacts=False)
        fs = self.sampling_frequency()
        if output_data is None:
            return None, None
        output_feats = EEGFeatures.extract_features(output_data, fs, normalize=norm, pool_region=False)
        return output_feats, output_labels, channels_to_remove

    # Processes data from IEEG.org in multiple batches
    # Inputs
    #   num: number of EEG batches to process
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   use_filter: whether to apply a bandpass filter to the EEG
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    #   save_artifacts: whether to save timepoints of EEG artifacts
    # Outputs
    #   output_data: a set of processed EEG segments with shape N* x C x D, where N* is the number
    #                of clean EEG segments from the patient's dataset, C is the number of clean
    #                channels and D is the number of samples within each segment (S = fs * s_length)
    #   output_labels: a modified list of seizure annotations of the given patient with length N*
    #   indices_to_remove: a list indicating whether each EEG segment should be removed, with length N
    #   channels_to_remove: a N* x C array that indicates whether each channel in each segment should be removed
    def process_data(self, num, start, length, use_filter=True, eeg_only=True, save_artifacts=False):
        input_data = IEEGDataLoader.load_data_batch(self, num, start, length, eeg_only)
        input_data = np.swapaxes(input_data, 1, 2)
        input_labels = IEEGDataLoader.load_annots(self, num, start, length, use_file=True)
        fs = IEEGDataLoader.sampling_frequency(self)
        # Filter from 0.5 to 20 Hz if selected
        if use_filter:
            coeff = butter(4, [0.5 / (fs / 2), 20 / (fs / 2)], 'bandpass')
            input_data = filtfilt(coeff[0], coeff[1], input_data, axis=-1)
        # Perform artifact rejection over the input data
        output_data, output_labels, indices_to_remove, channels_to_remove = self.clean_data(input_data, input_labels,
                                                                                            fs, ARTIFACT_METHOD)
        # Postprocess data structure for channels removed
        channels_to_remove = np.array([channels_to_remove[ii] for ii in range(np.size(channels_to_remove, axis=0))
                                       if indices_to_remove[ii] == 0])
        # Save artifact information if required
        if save_artifacts:
            Artifacts.save_artifacts(self.id, indices_to_remove, start, length)
        return output_data, output_labels, indices_to_remove, channels_to_remove

    # Performs artifact rejection over all EEG recordings within the given dataset
    # and updates the annotation file in accordance with the processed EEG data
    # Inputs
    #   input_data: EEG sample of shape N x C x D, where N is the number of EEG batches, C is the number of
    #               valid channels and D is the total number of samples within the given patient's EEG
    #   input_labels: a list of seizure annotations of the given patient for every segment
    #   fs: sampling frequency of the patient's EEG recording
    #   artifact_rejection: specific method for removing artifact-like segments.
    #                       'stats' removes all segments with z-scores above 5 under
    #                       1) Range 2) Line Length 3) Bandpower in beta frequency band (12 - 20 Hz)
    #                       'threshold' removes all segments exceeding certain thresholds under
    #                       1) Variance 2) Range 3) Line Length 4) Bandpower 5) Signal difference
    # Outputs
    #   output_data: a set of processed EEG segments with shape N* x C x D, where N* is the number
    #                of valid EEG segments from the patient's dataset, C is the number of valid
    #                channels and D is the number of samples within each segment (S = fs * s_length)
    #   output_labels: a modified list of seizure annotations of the given patient with length N*
    #   indices_to_remove: a list indicating whether each EEG segment should be removed, with length N
    #   channels_to_remove: a N x C array that indicates whether each channel in each segment should be removed
    #   returns None if no preprocessed data is available
    @staticmethod
    def clean_data(input_data, input_labels, fs, artifact_rejection='none'):
        # Determine batches to remove
        indices_to_remove, channels_to_remove = Artifacts.remove_artifacts(input_data, fs, channel_limit=3,
                                                                           method=artifact_rejection)
        print('Number of rejected segments: ', np.sum(indices_to_remove))
        # Remove cross-channel artifact data
        indices_to_keep = (1 - indices_to_remove).astype('float')
        indices_to_keep[indices_to_keep == 0] = np.nan
        output_data = indices_to_keep[:, None, None] * input_data
        # Remove channel-specific artifact data
        channels_to_keep = (1 - channels_to_remove).astype('float')
        channels_to_keep[channels_to_keep == 0] = np.nan
        output_data = np.expand_dims(channels_to_keep, axis=-1) * output_data
        # Return None if output data is unavailable
        if len(output_data) == 0:
            return None, None, None
        # Remove completely NaN-filled portions of the EEG data
        output_data = output_data[(1-indices_to_remove).astype(bool), :, :]
        # Update labels to match the size of the clean recordings
        output_labels = np.array([input_labels[idx] for idx, element in enumerate(indices_to_remove) if element == 0])
        return output_data, output_labels, indices_to_remove, channels_to_remove
