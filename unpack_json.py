# %% Imports
import json
import pandas as pd
import numpy as np
import sys
import pickle

# %% Save data as pkl files from json
filename = 'annotations.json'

f = open(filename)
tot_data = json.load(f)

for num in tot_data.keys():
    # get patient dictionary and patient id
    pt_data = tot_data[num]
    pt_id = pt_data['patient']
    print(f'Patient: {pt_id}')

    # save interictal start and stop data
    ii_start = np.array(pt_data['ii_start'], dtype=float)
    ii_stop = np.array(pt_data['ii_stop'], dtype=float)
    num_ii = ii_start.shape[0]

    # save seizure start and stop data
    try:
        sz_start = np.array(pt_data['sz_start'], dtype=float)
        sz_stop = np.array(pt_data['sz_stop'], dtype=float)
        num_sz = sz_start.shape[0]
    except KeyError: # for CNT patients with no sz data in json
        sz_start = np.array([])
        sz_stop = np.array([])
        num_sz = 0

    #combine ii and sz data to put into dataframe
    num_segs = num_ii + num_sz
    all_start = np.concatenate((sz_start, ii_start))
    all_start = all_start.reshape(num_segs,)
    all_stop = np.concatenate((sz_stop, ii_stop))
    all_stop = all_stop.reshape(num_segs,)

    # create labels
    sz_list = ['seizure']*num_sz
    ii_list = ['interictal']*num_ii
    event_vec = sz_list + ii_list

    #combine data and create dataframe
    comb_data = {'event': event_vec, 'start': all_start, 'stop': all_stop}
    annot_df = pd.DataFrame(comb_data, columns=['event', 'start', 'stop'])
    # print(annot_df)

    # save dataframe to dataset folder
    annot_df.to_pickle('dataset/from_json/%s_from_json.pkl' % pt_id)

    ## use input to check process without running all code
    # user_input = input("Press 1 to continue or any other key to exit")
    # if not user_input == '1':
    #     sys.exit(0)

f.close()
# %% Read in pkl files saved from annotations.json and check format

# define pt name
pt = 'ICUDataRedux_0003'

# load data saved from json file
json_path = 'dataset/from_json/%s_from_json.pkl' % pt
json_pkl = open(json_path, 'rb')
json_data = pickle.load(json_pkl)
print('JSON data for %s' % pt)
print(json_data)

# load previously used data
try:
    reg_path = 'dataset/%s.pkl' % pt
    reg_pkl = open(reg_path, 'rb')
    reg_data = pickle.load(reg_pkl)
    print('Regular data for %s' % pt)
    print(reg_data)
except FileNotFoundError:
    print("File not present in non-json data")


# %% Get list of patients with and without seizures
filename = 'annotations.json'

f = open(filename)
tot_data = json.load(f)

sz_list = []
non_sz_list = []
for num in tot_data.keys():
    # get patient dictionary and patient id
    pt_data = tot_data[num]
    pt_id = pt_data['patient']
    # print(f'Patient: {pt_id}')
    if 'sz_start' in pt_data.keys():
        sz_list.append(pt_id)
    else:
        non_sz_list.append(pt_id)
    

# %%
