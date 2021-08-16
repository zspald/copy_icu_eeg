# %% Imports
import h5py
import os
import pickle
import pandas as pd
import glob 
import numpy as np
from evaluate import EEGEvaluator

# %%
# pt_id = "ICUDataRedux_0085"
# filename = "data/%s_data_rf.h5" % pt_id

# f = h5py.File(filename, 'r')
# print(filename)
# data = f['feats'][:]
# labels = f['labels'][:]
# # print(f['maps'].shape[0])
# f.close()


for filename in glob.glob("data/*_data_wt_rf.h5"):
    f = h5py.File(filename, 'r')
    if f['feats'][:].shape[0] == 0:
        print(f"{filename}: {f['feats'][:].shape}")
    # try:
    #     print(f"{filename}: {f['maps'].shape[0]}")
    #     f.close()
    # except KeyError:
    #     print(f"{filename}: Maps empty")
    #     f.close()

# %% Re-organize prediction data

for file_path in glob.glob("deployment_rf/pred_data/*.npy"):
    if file_path.endswith("__0_3s.npy"):
        filename = os.path.basename(file_path)
        filename = filename[:-10] + '_3s_0.25.npy'
        # print(filename)
        os.rename(file_path, 'deployment_rf/pred_data/%s' % filename)



# %%

# sort_inds = np.argsort(labels[:,1])
# sorted_labels = labels[sort_inds, :]
# length = 1
# start = sorted_labels[0,1]
# stop = sorted_labels[-1,2]
# preds = sorted_labels[:,0]

# print("Predictions:")
# print(preds)
# print(f"Shape of predictions: {preds.shape}")

# filename_pick = 'dataset/%s.pkl' % pt_id
# f_pick = open(filename_pick, 'rb')
# annots = pickle.load(f_pick)
# annots = annots.sort_values(by=['start'], ignore_index=True)
# print(annots)
# annots['event'] = annots['event'].apply(lambda x: 1 if x == 'seizure' else 0)
# # print(annots)
# f_pick.close()

# pkl_labels = EEGEvaluator.annots_pkl_to_1D(filename_pick, start, stop, pred_length=length)
# pkl_labels = pkl_labels[:preds.shape[0]]
# print("Labels:")
# print(pkl_labels)
# print(f"Shape of labels: {pkl_labels.shape}")

# %%
# stats_sz = EEGEvaluator.sz_sens(pt_id, preds, pred_length=length)
# stats_non_sz = EEGEvaluator.data_reduc(pt_id, preds, pred_length=length)
# EEGEvaluator.compare_outputs_plot(pt_id, preds, length=(stop-start)/60, pred_length=length)

# %% Get list of start and stop times from pkl 
# annotations to use in processing data

start_stop_times = pd.DataFrame(columns=['patient_id', 'start', 'stop'])
dir = glob.glob("dataset/from_json/*_from_json.pkl")
for pkl_filename in dir:
    if pkl_filename == "dataset/patient_start_stop.pkl":
        continue
    f_pick = open(pkl_filename, 'rb')
    annots = pickle.load(f_pick)
    annots = annots.sort_values(by=['start'], ignore_index=True)
    times = annots.values
    start = times[0,1]
    stop = times[-1,2]
    to_append = {'patient_id': pkl_filename[18:-14], 'start': start, 'stop': stop}
    start_stop_times = start_stop_times.append(to_append, ignore_index=True)
    f_pick.close()
print(start_stop_times)
# start_stop_times.to_pickle("dataset/patient_start_stop.pkl")


# 
# %%
