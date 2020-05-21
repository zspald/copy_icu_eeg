############################################################################
# Demonstrates the use of the IEEGDataProcessor class
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import the IEEGDataProcessor to test its functionality
from preprocess_dataset import IEEGDataProcessor
from features_2d import EEGMap
from train import EEGLearner
import h5py
import numpy as np

# Define IEEG username and password here
username = 'danieljkim0118'
password = 'kjm39173917#'

# Extract features 50 times with batch size 20 and segment length of 5 seconds.
# Generate the corresponding map.
print("==========Example==========")
dataset = IEEGDataProcessor('RID0061', username, password)
print(dataset.id)
map_outputs = dataset.generate_map(num_iter=10, num_batches=200, start=0, length=5, use_filter=True, eeg_only=True,
                                   normalize=False, has_seizure=True)

# Train a CNN model on sample patient data
patient_list = ['RID0061', 'RID0062', 'RID0063', 'RID0064']
train_module = EEGLearner(patient_list)
train_module.train_cnn(epochs=10, cross_val=False, save=False)
