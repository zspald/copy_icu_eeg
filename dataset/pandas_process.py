import numpy as np
import pandas as pd
import h5py
import scipy.io

patient_no = 28

file1 = h5py.File(('dataset/annot_{}.mat').format(patient_no),'r')

#patient_name = np.array2string(file1['/annot_1/patient'])

patient_name = bytes(file1.get(('/annot_{}/patient').format(patient_no))[:]).decode('utf-16')

ii_start = np.array(file1[('/annot_{}/ii_start').format(patient_no)])
ii_stop = np.array(file1[('/annot_{}/ii_stop').format(patient_no)])
sz_start = np.array(file1[('/annot_{}/sz_start').format(patient_no)])
sz_stop = np.array(file1[('/annot_{}/sz_stop').format(patient_no)])

num_sz = len(sz_start)
num_ii = len(ii_start)

num_segs = num_sz+num_ii

all_start = np.vstack((sz_start,ii_start))

all_start = all_start.reshape(num_segs,)

all_stop = np.vstack((sz_stop,ii_stop))

all_stop = all_stop.reshape(num_segs,)

sz_list = ['seizure']*num_sz
ii_list = ['interictal']*num_ii

event_vec = sz_list+ii_list

#print(event_vec)

data3 = {'event':  event_vec,
        'start': all_start,
        'stop': all_stop}

df = pd.DataFrame (data3, columns = ['event','start','stop'])

print(df)
print(patient_name)

df.to_pickle(("dataset/{}.pkl").format(patient_name))