# %% Load in data and save artifact information

# Import libraries
from getpass import getpass
from preprocess_dataset import IEEGDataProcessor
from features import EEGFeatures
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Define IEEG username and password here
username = 'zspald'
password = getpass("Enter the IEEG password: ")
num_seg = 1000
start = 60 #seconds
length = 15 #seconds

patient_list = ['CNT685', 'CNT691', 'CNT700', 'ICUDataRedux_0054', 'ICUDataRedux_0063', 
                'ICUDataRedux_0068', 'ICUDataRedux_0074', 'ICUDataRedux_0089']

for patient_id in patient_list:
    rejection_log_df = pd.DataFrame() #empty rejection log to make function work

    # Load processed EEG recordings with 2000 segments and segment length of 5 seconds. Save artifact info.
    dataset = IEEGDataProcessor(patient_id, username, password)
    data, labels, indices_to_remove, channels_to_remove, rejection_log_df = dataset.process_data(num=num_seg,
                                                                            start=start, length=length,
                                                                            curr_iter=0,
                                                                            rejection_log_df=rejection_log_df,
                                                                            use_filter=True, eeg_only=True)

    #get non rejected data
    normal_data = data[indices_to_remove == 0]

    #remove segments with no artifact channel rejections/all kept channels
    normal_inds = np.where(~(1-channels_to_remove).all(axis=1) == True)[0]
    channels_to_remove = channels_to_remove[~(1-channels_to_remove).all(axis=1)]

    #filter removed segments from above out of data
    normal_data = normal_data[normal_inds]
    normal_timepoints = [idx * length + start for idx in normal_inds]
    # normal_data = normal_data[~np.isnan(normal_data).any(axis=2)].reshape(normal_data.shape[0], -1, normal_data.shape[2])
    print(normal_data.shape)

    #get artifact segments
    artifact_data = data[indices_to_remove == 1]
    artifact_inds = [idx for idx, elem in enumerate(indices_to_remove) if elem == 1]
    artifact_timepoints = [idx * length + start for idx, elem in enumerate(indices_to_remove) if elem == 1]
    # artifact_data = artifact_data[~np.isnan(artifact_data).any(axis=2)].reshape(artifact_data.shape[0], -1, artifact_data.shape[2])
    print(artifact_data.shape)

    # data for plots
    channel_list = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fz' 'Fp1', 'Fp2', 'T3', 'T4', 'T5', 'T6', 'Pz',
       'P3', 'P4', 'O1', 'O2']  
    num_plots = 10

    # Plot Artifact data for rejected segments
    artifact_used_list = []
    for j in range(num_plots if len(artifact_inds) >= num_plots else len(artifact_inds)):
        if len(artifact_inds) >= num_plots:
            artifact_seg = random.choice(range(len(artifact_inds)))
            while artifact_seg in artifact_used_list:
                artifact_seg = random.choice(range(len(artifact_inds)))
        else:
            artifact_seg = j
        artifact_used_list.append(artifact_seg)
        artifact_seg_label = artifact_inds[artifact_seg]
        time_bound = artifact_timepoints[artifact_seg]
        x = np.linspace(time_bound, time_bound + length, artifact_data.shape[-1])
        num_channels = artifact_data.shape[1]
        f, axs = plt.subplots(nrows=num_channels, ncols=1, sharex=True, figsize=(12,8))
        for i in range(num_channels):
            axs[i].plot(x, artifact_data[artifact_seg, i, :], color='r')
            axs[i].set_ylabel(channel_list[i], labelpad=15, rotation=0)
        f.suptitle(f'Rejected Artifact Segment (ID:{patient_id}, Segment: {artifact_seg_label})')
        f.text(0.5, 0.04, 'Time in Recording (s)', ha='center')
        # f.text(0.04, 0.5, 'Channels', va='center', rotation=0)
        plt.savefig('artifact_rejection_clips/rejected_seg/%s_%s_rejection_clip_seg.pdf' % \
                                (patient_id, artifact_seg_label),
                                bbox_inches="tight", dpi=100)
        plt.show()


    # Plot artifact data for kept segments with rejected channels
    normal_used_list = []
    for j in range(num_plots if len(normal_inds) >= num_plots else len(normal_inds)):
        if len(normal_inds) >= num_plots:
            normal_seg = random.choice(range(len(normal_inds)))
            while normal_seg in normal_used_list:
                normal_seg = random.choice(range(len(normal_inds)))
        else:
            normal_seg = j
        normal_used_list.append(normal_seg)
        normal_seg_label = normal_inds[normal_seg]
        time_bound = normal_timepoints[normal_seg]
        x = np.linspace(time_bound, time_bound + length, normal_data.shape[-1])
        num_channels = normal_data.shape[1]
        f, axs = plt.subplots(nrows=num_channels, ncols=1, sharex=True, figsize=(12,8))
        for i in range(num_channels):
            if channels_to_remove[normal_seg, i] == 1:
                color = 'r'
            else:
                color = 'b'
            axs[i].plot(x, normal_data[normal_seg, i, :], color=color)
            axs[i].set_ylabel(channel_list[i], labelpad=15, rotation=0)
        f.suptitle(f'Kept Segment with Rejected Channels (ID:{patient_id}, Segment: {normal_seg_label})')
        f.text(0.5, 0.04, 'Time in Recording (s)', ha='center')
        # f.text(0.04, 0.5, 'Channels', va='center', rotation=0)
        plt.savefig('artifact_rejection_clips/rejected_channels/%s_%s_rejection_clip_chan.pdf' % \
                            (patient_id, normal_seg_label),
                            bbox_inches="tight", dpi=100)

        plt.show()

# %%
