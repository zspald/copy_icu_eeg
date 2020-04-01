############################################################################
# Demonstrates the use of the IEEGDataProcessor class
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import the IEEGDataProcessor to test its functionality
from preprocess_dataset import IEEGDataProcessor
from features_2d import EEGMap
import numpy as np

# Define IEEG username and password here
username = 'danieljkim0118'
password = 'kjm39173917#'

# Load processed EEG recordings with batch size 100 and segment length of 5 seconds. Save artifact info.
# print("==========Example #1==========")
# dataset1 = IEEGDataProcessor('RID0060', username, password)
# data, labels = dataset1.process_data(num=100, start=600, length=5, use_filter=True, eeg_only=True,
#                                      channels_to_filter=['Pz'], save_artifacts=True)
#
# print('shape of preprocessed data: ', data.shape)
# print('shape of preprocessed labels: ', labels.shape)

# Extract features 50 times with batch size 20 and segment length of 5 seconds.
print("==========Example #2==========")
dataset2 = IEEGDataProcessor('RID0061', username, password)
feats, labels, channel_info = dataset2.process_all_feats(num_iter=5, num_batches=200, start=0, length=5, use_filter=True,
                                                         eeg_only=True, save=False)
map_outputs = EEGMap.generate_map(feats, channel_info)
print(np.shape(map_outputs))
# print(feats.shape)
# print(np.isnan(feats).any())
# print(labels.shape)

# Extract features 100 times with batch size 30 and segment length of 1 second each, starting at 10 mins.
# Exclude the F8 and Pz channels.
# print("==========Example #3==========")
# dataset3 = IEEGDataProcessor('RID0061', username, password)
# dataset3.process_all_feats(num_iter=3, num_batches=300, start=0, length=5, use_filter=True,
#                            eeg_only=True, channels_to_filter=['F8', 'Pz'], save=False)

# Extract features 10 times with batch size 50 and segment length of 3 seconds each. Save the results.
# print("==========Example #4==========")
# dataset4 = IEEGDataProcessor('RID0065', username, password)
# dataset4.process_all_feats(num_iter=10, num_batches=50, start=0, length=3, use_filter=True,
#                            eeg_only=True, channels_to_filter=None, save=True)
