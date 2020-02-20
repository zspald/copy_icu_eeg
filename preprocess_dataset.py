############################################################################
# Preprocesses EEG data, annotations from ICU patients at HUP
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
from artifacts import Artifacts
from features import EEGFeatures
from load_dataset import IEEGDataLoader
from scipy.signal import butter, filtfilt
import h5py
import numpy as np

# Artifact rejection method
ARTIFACT_METHOD = 'none'


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
    #   channels_to_filter: a list of EEG channels to filter
    #   save: whether to save the results to a .hdf5 file
    # Outputs
    #
    def process_all_feats(self, num_iter, num_batches, start, length, use_filter=True, eeg_only=True,
                          channels_to_filter=None, save=False):
        # Initialize patient-specific feature and label outputs
        patient_feats = None
        patient_labels = None
        # Iterate over all batches
        for ii in range(num_iter):
            # Search for the first point (in seconds) where non-NaN values occur
            if ii == 0:
                channels_to_use = self.filter_channels(eeg_only, channels_to_exclude=channels_to_filter)
                start = self.crawl_data(start, interval_length=600, threshold=3600, channels_to_use=channels_to_use)
                if start is None:
                    print("Patient contains NaN recordings for over an hour!")
                    return
                print("The starting point is: %d seconds" % start)
            print("===Iteration %d===" % (ii + 1))
            # Extract features using the given IEEGDataProcessor object
            feats, labels = self.get_features(num_batches, start, length, norm='off', use_filter=use_filter,
                                              eeg_only=eeg_only, channels_to_filter=channels_to_filter)
            start += num_batches * length
            # Add batch features and labels to patient-specific outputs
            if feats is None:
                print("No data is available from batch #%d" % (ii + 1))
            else:
                patient_feats = feats if patient_feats is None else np.r_[patient_feats, feats]
                patient_labels = labels if patient_labels is None else np.r_[patient_labels, labels]
        # Save features and labels if indicated by the user
        if save:
            with h5py.File('%s_feats.h5' % self.id, "w") as file:
                file.create_dataset('feats', data=patient_feats)
                file.create_dataset('labels', data=patient_labels)
        print("Shape of the features: ", np.shape(patient_feats))
        return patient_feats, patient_labels

    # Extracts features for a specified interval of EEG recordings from IEEG
    # Inputs
    #   num: number of EEG batches to process
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   use_filter: whether to apply a bandpass filter to the EEG
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    #   channels_to_filter: a list of EEG channels to filter
    # Outputs
    #   output_feats: a feature array that contains features for all EEG segments across all channels
    #                 with shape N* x C x F, where N* is the number of clean EEG segments from the
    #                 patient's dataset, C is the number of channels and F is the number of features
    #   output_labels: a modified list of seizure annotations of the given patient with length N*
    #   returns as None if no preprocessed data is available
    def get_features(self, num, start, length, norm='off', use_filter=True, eeg_only=True, channels_to_filter=None):
        output_data, output_labels, _ = self.process_data(num, start, length, use_filter, eeg_only, channels_to_filter,
                                                          save_artifacts=False)
        fs = self.sampling_frequency()
        if output_data is None:
            return None, None
        output_feats = EEGFeatures.extract_features(output_data, fs, normalize=norm)
        return output_feats, output_labels

    # Processes data from IEEG.org in multiple batches
    # Inputs
    #   num: number of EEG batches to process
    #   start: starting point, in seconds
    #   length: duration of each segment, in seconds
    #   use_filter: whether to apply a bandpass filter to the EEG
    #   eeg_only: whether to remove non-EEG channels (e.g. EKG)
    #   channels_to_filter: a list of EEG channels to filter
    #   save_artifacts: whether to save timepoints of EEG artifacts
    # Outputs
    #   output_data: a set of processed EEG segments with shape N* x C x D, where N* is the number
    #                of clean EEG segments from the patient's dataset, C is the number of clean
    #                channels and D is the number of samples within each segment (S = fs * s_length)
    #   output_labels: a modified list of seizure annotations of the given patient with length N*
    def process_data(self, num, start, length, use_filter=True, eeg_only=True, channels_to_filter=None,
                     save_artifacts=False):
        input_data = IEEGDataLoader.load_data_batch(self, num, start, length, eeg_only, channels_to_filter)
        input_data = np.swapaxes(input_data, 1, 2)
        input_labels = IEEGDataLoader.load_annots(self, num, start, length, use_file=True)
        sample_freq = IEEGDataLoader.sampling_frequency(self)
        # Filter from 0.5 to 20 Hz if selected
        if use_filter:
            coeff = butter(4, [0.5 / (sample_freq / 2), 20 / (sample_freq / 2)], 'bandpass')
            input_data = filtfilt(coeff[0], coeff[1], input_data, axis=-1)
        # Perform artifact rejection over the input data
        output_data, output_labels, timestamps = self.clean_data(input_data, input_labels, sample_freq, ARTIFACT_METHOD)
        # Save artifact information if required
        if save_artifacts:
            Artifacts.save_artifacts(self.id, timestamps, start, length)
        return output_data, output_labels, timestamps

    # Performs artifact rejection over all EEG recordings within the given dataset
    # and updates the annotation file in accordance with the processed EEG data
    # Inputs
    #   input_data: EEG sample of shape N x C x D, where N is the number of EEG batches, C is the number of
    #               valid channels and D is the total number of samples within the given patient's EEG
    #   input_labels: a list of seizure annotations of the given patient for every segment
    #   fs: sampling frequency of the patient's EEG recording
    #   s_length: length of each EEG segment to be inspected, in seconds
    #   artifact_rejection: specific method for removing artifact-like segments.
    #                       'default' removes all segments that exceed a certain threshold under
    #                       1) Range 2) Line Length 3) Bandpower in beta frequency band (25 - 60 Hz)
    #                       other methods to be incorporated
    # Outputs
    #   output_data: a set of processed EEG segments with shape N* x C x D, where N* is the number
    #                of valid EEG segments from the patient's dataset, C is the number of valid
    #                channels and D is the number of samples within each segment (S = fs * s_length)
    #   output_labels: a modified list of seizure annotations of the given patient with length N*
    #   indicator: a list indicating whether each EEG segment should be removed, with length N
    #   returns None if no preprocessed data is available
    @staticmethod
    def clean_data(input_data, input_labels, fs, artifact_rejection='none'):
        # Determine batches to remove
        indicator = Artifacts.remove_artifacts(input_data, fs, method=artifact_rejection)
        # Remove artifact data
        indices_to_keep = [idx for idx in range(len(indicator)) if indicator[idx] == 0]
        output_data = input_data[indices_to_keep]
        # Return None if output data is unavailable
        if len(output_data) == 0:
            return None, None, None
        # Update labels to match the size of the clean recordings
        output_labels = np.array([input_labels[idx] for idx, element in enumerate(indicator) if element == 0])
        return output_data, output_labels, indicator
