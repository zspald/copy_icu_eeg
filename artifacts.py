#############################################################################
# Removes artifacts from EEG data obtained at IEEG.org
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
#############################################################################

# Import libraries
import numpy as np
import pandas as pd
import scipy.stats
from features import EEGFeatures

import matplotlib.pyplot as plt
import seaborn as sns


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
    def remove_artifacts(input_data, fs, curr_iter, rejection_log_df, channel_limit=3, method='threshold'):
        # Remove segments that include NaN values first
        indices_to_remove, channels_to_remove, rejection_log_df = Artifacts.remove_nans(input_data, curr_iter, rejection_log_df)
        input_data = np.nan_to_num(input_data)
        if np.size(input_data, axis=0) == 0:
            return list()
        elif method == 'stats':
            indices_to_remove, channels_to_remove = \
                Artifacts.remove_artifacts_stats(input_data, fs, channel_limit, indices_to_remove, channels_to_remove)
        elif method == 'threshold':
            indices_to_remove, channels_to_remove, rejection_log_df = \
                Artifacts.remove_artifacts_thresholds(input_data, fs, curr_iter, rejection_log_df, channel_limit, indices_to_remove,
                                                      channels_to_remove)
        return indices_to_remove, channels_to_remove, rejection_log_df

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
    def remove_nans(input_data, curr_iter, rejection_log_df, channel_limit=3):
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
                rejection_log_df = Artifacts.log_nans(rejection_log_df, curr_iter, ii, mask_sum[ii])
        return segments_to_remove, channels_to_remove, rejection_log_df

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
    def remove_artifacts_thresholds(input_data, fs, curr_iter, rejection_log_df, channel_limit=3, indices_to_remove=None, channels_to_remove=None):
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
        violation_criteria = np.array([(var < var_threshold), (minmax > minmax_threshold),
                                       (llength > llength_threshold), (bdpower > bdpower_threshold),
                                       (diff > diff_threshold)])
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
                # rejection_log_df = Artifacts.log_violations(rejection_log_df, curr_iter, ii, violation_criteria)
            else:
                channels_to_remove[ii] = violations[ii]
        # Remove noise-filled channels (threshold is 50% of all input samples)
        channels_to_filter = np.sum(channels_to_remove, axis=0) > int(0.5 * np.size(channels_to_remove, axis=0))
        channels_to_remove[:, channels_to_filter] = 1
        return indices_to_remove, channels_to_remove, rejection_log_df

    # Obtains time intervals of all EEG segments labeled as artifacts
    # Inputs
    #   patient_id: the id of the patient dataset
    #   indicator: a list indicating the artifact type of each EEG segment
    #   start: absolute starting time point of the EEG batch
    #   length: length of each EEG segment
    # Outputs
    #   a pandas dataframe storing the timepoints of detected artifacts
    @staticmethod
    def save_artifacts(patient_id, indicator, start, length):
        if indicator is not None:
            print('Saving artifact info...')
            start = [start + ii * length for ii in range(len(indicator)) if indicator[ii] > 0]
            stop = np.array(start) + length
            artifact_data = {'start': start, 'stop': stop}
            dataframe = pd.DataFrame(data=artifact_data)
            dataframe.to_pickle(r'./dataset/%s_artifacts.pkl' % patient_id)
        return

    @staticmethod
    def log_nans(rejection_log_df, curr_iter, seg_num, nans_in_seg):
        #create dictionary from segment information and append to log dataframe
        to_append = {'Iteration': curr_iter, 'Segment': seg_num, 'NaN': nans_in_seg,
                    'Variance': 0, 'Minmax': 0, 'Line Length': 0, 'Band Power': 0, 
                    'Signal Diff': 0}
        rejection_log_df = rejection_log_df.append(to_append, ignore_index=True)

        return rejection_log_df

    @staticmethod
    def log_violations(rejection_log_df, curr_iter, seg_num, violation_criteria):
        #calculate counts of violation by type (var, band power, etc.) for current segment
        segment_viols = violation_criteria[:, seg_num, :]
        viol_by_criteria = np.sum(segment_viols, axis=1)

        # #update artifact rejection logs if iteration, segment pair exists
        # if (rejection_log_df[['Iteration','Segment']].values == [curr_iter, seg_num]).all(axis=1).any():
        #     rejection_log_df.loc[(rejection_log_df['Iteration'] == curr_iter) & \
        #     (rejection_log_df['Segment'] == seg_num), \
        #     ['Variance', 'Minmax', 'Line Length', 'Band Power', 'Signal Diff']] = viol_by_criteria
        # else:
        #     #create dictionary from segment information and append to log dataframe
        #     to_append = {'Iteration': curr_iter, 'Segment': seg_num, 'NaN': 0, 'Variance': viol_by_criteria[0],
        #                 'Minmax': viol_by_criteria[1], 'Line Length': viol_by_criteria[2],
        #                 'Band Power': viol_by_criteria[3], 'Signal Diff': viol_by_criteria[4]}
        #     rejection_log_df = rejection_log_df.append(to_append, ignore_index=True)

        #create dictionary from segment information and append to log dataframe if iteration, segment pair not present
        if not (rejection_log_df[['Iteration','Segment']].values == [curr_iter, seg_num]).all(axis=1).any():
            #create dictionary from segment information and append to log dataframe
            to_append = {'Iteration': curr_iter, 'Segment': seg_num, 'NaN': 0, 'Variance': viol_by_criteria[0],
                        'Minmax': viol_by_criteria[1], 'Line Length': viol_by_criteria[2],
                        'Band Power': viol_by_criteria[3], 'Signal Diff': viol_by_criteria[4]}
            rejection_log_df = rejection_log_df.append(to_append, ignore_index=True)

        return rejection_log_df

    @staticmethod
    def save_rejection_log(rejection_log_df, patient_id, visualize=False):
        #calculate rejection percentages by type (for visualization later)
        rejection_log_df[['NaN %', 'Variance %', 'Minmax %', 'Line Length %', 'Band Power %', 'Signal Diff %']] = \
                rejection_log_df[['NaN', 'Variance', 'Minmax', 'Line Length', 'Band Power', 'Signal Diff']].div(\
                rejection_log_df[['NaN', 'Variance', 'Minmax', 'Line Length', 'Band Power', 'Signal Diff']].sum(axis=1), 
                axis=0)

        #reorder dataframe columns
        rejection_log_df = rejection_log_df[['Iteration', 'Segment', 'NaN', 'Variance', 'Minmax', 'Line Length',
                                             'Band Power', 'Signal Diff', 'NaN %', 'Variance %', 'Minmax %', 
                                             'Line Length %', 'Band Power %', 'Signal Diff %']]

        #save dataframe to csv file
        rejection_log_df.to_csv('artifact_rejection_logs/%s_logs.csv' % patient_id)

        
        #plot data to give information on reaaons for segment rejection for a given dataset
        if visualize:
            #calculate total number of rejections
            num_rejected = rejection_log_df.shape[0]

            #reorganize dataframe to be better plotted with seaborn
            plot_df = pd.melt(rejection_log_df, id_vars=['Iteration', 'Segment'], value_vars = \
                    ['NaN %', 'Variance %', 'Minmax %', 'Line Length %', 'Band Power %', 'Signal Diff %'])

            #plot bar chart of rejection percentages per iteration/segment (i.e. percentages of violations contributing 
            #to rejection for each sample)
            plt.figure()
            sns.barplot(x='variable', y='value', data=plot_df)
            plt.ylabel('% of Total Rejections (Grouped by Iteration/Segment)')
            plt.xlabel('Rejection Criteria')
            plt.title('%s Artifact Rejection Criteria (%d Total Rejections)' % (patient_id, num_rejected))
            
            #save figure as pdf
            plt.savefig('artifact_rejection_logs/visualizations/%s_rejections_visual.pdf' % patient_id,
                        bbox_inches="tight", dpi=100)
        

