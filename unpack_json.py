# %% 
import json
import pandas as pd
import numpy as np

filename = 'annotations.json'

f = open(filename)
tot_data = json.load(f)

#TODO
# read in data from each patient, use Daniel's code to interpret sz_start/stop 
# and ii_start/stop, put data into a dataframe, and save that data as a pkl
# file for each patient

for num in tot_data.keys():
    patient_data = tot_data[num]
    start = patient_data['data_start']
    stop = patient_data['data_stop']
    test = patient_data['']

f.close()
# %%
