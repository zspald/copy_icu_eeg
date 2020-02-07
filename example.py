# A file that demonstrates how to use the IEEGData-related classes

from preprocess_dataset import IEEGDataProcessor

# Define IEEG username and password here
username = 'danieljkim0118'
password = 'kjm39173917#'

# Example of
dataset = IEEGDataProcessor('RID0061', username, password)
dataset.process_all_feats(num_iter=10, num_batches=10, start=600, length=5, use_filter=True, eeg_only=True)
