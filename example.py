############################################################################
# Demonstrates the use of the IEEGDataProcessor class
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import the IEEGDataProcessor and EEGLearner to test their functionalities
from getpass import getpass
from preprocess_dataset import IEEGDataProcessor
from train import EEGLearner

# Define IEEG username and password here
username = input("Enter the IEEG username: ")
password = getpass("Enter the IEEG password: ")
dataset_id = input("Enter the dataset ID: ")

# Extract features 50 times with batch size 20 and segment length of 5 seconds.
# Generate the corresponding map.
print("========== Pre-processing example ==========")
dataset = IEEGDataProcessor(dataset_id, username, password)
print("Dataset ID: ", dataset.id)
map_outputs = dataset.generate_map(num_iter=20, num_batches=1000, start=0, length=1,
                                   use_filter=True, eeg_only=True, normalize=True)

# Train a CNN model on sample patient data
patient_list = ["RID0061", "RID0062", "RID0063", "RID0064", "RID0065", "RID0066", "RID0067", "RID0068", "RID0069"]
print("========== Training example ==========")
train_module = EEGLearner(patient_list)
train_module.train_cnn(epochs=50, control=1800, cross_val=False, save=True, verbose=1)
train_module.train_convolutional_gru(epochs=20, batch_size=100, control=1800, save=True, seq_len=20, verbose=1)
train_module.train_conv_lstm(epochs=20, cross_val=False, save=True, seq_len=20, verbose=1)
