# %%
import h5py
import os
import pickle
import pandas as pd
import glob 

# %%
filename = 'data/ICUDataRedux_0085_data_bipolar_rf.h5'

f = h5py.File(filename, 'r')
print(filename)
data = f['feats']
labels = f['labels']
# print(f['maps'].shape[0])
# f.close()


# for filename in os.listdir("data"):
#     h5_filename = "data/" + filename
#     f = h5py.File(h5_filename, 'r')
#     try:
#         print(f"{filename}: {f['maps'].shape[0]}")
#         f.close()
#     except KeyError:
#         print(f"{filename}: Maps empty")
#         f.close()


# %% Get list of start and stop times from pkl 
# annotations to use in processing data

# start_stop_times = pd.DataFrame(columns=['patient_id', 'start', 'stop'])
# dir = glob.glob("dataset\*.pkl")
# for pkl_filename in dir:
#     if pkl_filename == "dataset\patient_start_stop.pkl":
#         continue
#     f_pick = open(pkl_filename, 'rb')
#     annots = pickle.load(f_pick)
#     annots = annots.sort_values(by=['start'], ignore_index=True)
#     times = annots.values
#     start = times[0,1]
#     stop = times[-1,2]
#     to_append = {'patient_id': pkl_filename[8:-4], 'start': start, 'stop': stop}
#     start_stop_times = start_stop_times.append(to_append, ignore_index=True)
#     f_pick.close()
# print(start_stop_times)
# start_stop_times.to_pickle("dataset/patient_start_stop.pkl")


# %%
