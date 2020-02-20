# Import libraries
from preprocess_dataset import IEEGDataProcessor
from features import EEGFeatures
from sklearn.cluster import KMeans
import numpy as np

# Define IEEG username and password here
username = 'danieljkim0118'
password = 'kjm39173917#'

# Load processed EEG recordings with batch size 100 and segment length of 5 seconds. Save artifact info.
dataset = IEEGDataProcessor('RID0060', username, password)
data, labels, timestamps = dataset.process_data(num=2000, start=60, length=5, use_filter=True, eeg_only=True,
                                                channels_to_filter=['Pz'])
fs = dataset.sampling_frequency()
timepoints = [idx * 5 for idx, elem in enumerate(timestamps) if elem == 0]
print(len(timepoints))

minmax = np.amax(np.amax(data, axis=2) - np.amin(data, axis=2), axis=1)
llength = np.amax(EEGFeatures.line_length(data), axis=1)
bdpower = np.amax(EEGFeatures.bandpower(data, fs, 12, 20), axis=1)
feats = np.c_[minmax, llength, bdpower]
print(feats.shape)


clusters = KMeans(n_clusters=2).fit_predict(feats)
np.save('artifact_labels.npy', clusters)

time_info = [[], []]
for idx, index in enumerate(clusters):
    time_info[index].append(timepoints[idx])


print(time_info[0][:10])
print(time_info[1][:10])
