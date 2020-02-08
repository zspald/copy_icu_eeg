############################################################################
# Demonstrates the use of the IEEGDataProcessor class
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import the IEEGDataProcessor to test its functionalities
from preprocess_dataset import IEEGDataProcessor

# Define IEEG username and password here
username = 'danieljkim0118'
password = 'kjm39173917#'

# Load processed EEG recordings

# Extract features 50 times with batch size 20 and segment length of 5 seconds each.
print("==========Loading Example #2==========")
dataset2 = IEEGDataProcessor('RID252_68561f5b', username, password)
dataset2.process_all_feats(num_iter=10, num_batches=20, start=0, length=10, use_filter=True,
                           eeg_only=True, channels_to_filter=None, save=False)

# # Extract features, but all data is filtered out
# print("==========Loading Example #2==========")
# dataset3 = IEEGDataProcessor('RID0061', username, password)
# dataset3.process_all_feats(num_iter=10, num_batches=20, start=600, length=5, use_filter=True, eeg_only=True)

