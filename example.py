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

# Extract features 50 times with batch size 20 and segment length of 5 seconds. Generate the corresponding map.
print("==========Example==========")
dataset = IEEGDataProcessor('RID0061', username, password)
print(dataset.id)
map_outputs = dataset.generate_map(num_iter=5, num_batches=20, start=0, length=5, use_filter=True, eeg_only=True,
                                   normalize=False)
EEGMap.visualize_map(map_outputs, 3)
