# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import h5py

# Patients to train random forest classifer on
patient_list = ["CNT684", "CNT687", "CNT689", "CNT690", "CNT691", "CNT692", "CNT694", "CNT695",
                "CNT698", 
                # "CNT700", 
                "CNT701", "CNT702", "CNT705", "CNT706", 
                # "ICUDataRedux_0054", "ICUDataRedux_0061",
                "ICUDataRedux_0063", "ICUDataRedux_0064", 
                # "ICUDataRedux_0065", 
                "ICUDataRedux_0068", "ICUDataRedux_0069", "ICUDataRedux_0072", "ICUDataRedux_0073", "ICUDataRedux_0074",
                "ICUDataRedux_0078", "ICUDataRedux_0082", "ICUDataRedux_0083", "ICUDataRedux_0084",
                "ICUDataRedux_0086", "ICUDataRedux_0087", "ICUDataRedux_0089", "ICUDataRedux_0090", "ICUDataRedux_0091"]

# True if using bipolar montage, false for referential montage
bipolar = False

# Initialize model with warm_start = True to train by patient
rfc = RandomForestClassifier(warm_start=True, n_estimators=0)
print(rfc.get_params())

# iterate over desired patients
for patient in patient_list:
    # load data from proper montage
    if bipolar:
        filename = "data/" + patient + "_data_bipolar_rf.h5"
    else:
        filename = "data/" + patient + "_data_rf.h5"

    # access h5 file with data
    with h5py.File(filename) as patient_data:
        # load in features and labels for current patient
        feats = (patient_data['feats'])[:]
        feats = feats.reshape(feats.shape[0], -1)
        labels = (patient_data['labels'])[:,0]
        # print(patient)
        # print(np.argwhere(np.isnan(feats)))

        # train model on current patient
        rfc.n_estimators += 10
        rfc.fit(feats, labels)
print(rfc.get_params())

test_list = ["CNT685", "CNT688", "ICUDataRedux_0062", "ICUDataRedux_0085"]

# iterate over desired patients
for patient in test_list:
    # load data from proper montage
    if bipolar:
        filename = "data/" + patient + "_data_bipolar_rf.h5"
    else:
        filename = "data/" + patient + "_data_rf.h5"

    # access h5 file with data
    with h5py.File(filename) as patient_data:
        # load in features and labels for current patient
        feats = (patient_data['feats'])[:]
        feats = feats.reshape(feats.shape[0], -1)
        labels = (patient_data['labels'])[:,0]
        
        # make predictions on current patient
        preds = rfc.predict(feats)
        num_correct = 0
        for i in range(len(preds)):
            if preds[i] == labels[i]:
                num_correct += 1
        # print(preds)
        print(f"{patient} Accuracy: {num_correct / len(preds)}")

# %%
