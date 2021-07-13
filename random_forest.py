# %% Imports and Parameters
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
import h5py
import pickle
from evaluate import EEGEvaluator

# Patients to train random forest classifer on
patient_list = ["CNT684", "CNT687", "CNT689", "CNT690", "CNT691", 
                "CNT692", "CNT694", "CNT695", "CNT698", "CNT700", 
                "CNT701", "CNT702", "CNT705", "CNT706", "ICUDataRedux_0054", 
                "ICUDataRedux_0061","ICUDataRedux_0063", "ICUDataRedux_0064", 
                "ICUDataRedux_0065", "ICUDataRedux_0068", "ICUDataRedux_0069", 
                "ICUDataRedux_0072", "ICUDataRedux_0073", "ICUDataRedux_0074", 
                "ICUDataRedux_0078", "ICUDataRedux_0082", "ICUDataRedux_0083", 
                "ICUDataRedux_0084", "ICUDataRedux_0086", "ICUDataRedux_0087", 
                "ICUDataRedux_0089", "ICUDataRedux_0090", "ICUDataRedux_0091"]

# True if using bipolar montage, false for referential montage
bipolar = False

# %% Model Training

# create rf model pipeline with feature selection step using recursive feature elimnation
# rfc = Pipeline([
#     ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False))),
#     ('rf_classifier', RandomForestClassifier(n_estimators=500, verbose=1))
# ])

rfc = Pipeline([
    ('pca_feature_selection', PCA(n_components=10)),
    ('rf_classifier', RandomForestClassifier(n_estimators=500, verbose=1))
])
# print(rfc.get_params())

# iterate over desired patients
tot_feats = None
tot_labels = None
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
        feats = np.nan_to_num(feats)
        if tot_feats is not None:
            tot_feats = np.r_[tot_feats, feats]
        else:
            tot_feats = feats
        # print(tot_feats.shape)

        labels = (patient_data['labels'])[:,0]
        if tot_labels is not None:
            tot_labels = np.r_[tot_labels, labels]
        else:
            tot_labels = labels

        # print(tot_labels.shape)

        # train model on current patient
        # rfc['rf_classifier'].n_estimators += 10

# pca = PCA()
# pca.fit(tot_feats)
# print(np.cumsum(pca.explained_variance_ratio_))
rfc.fit(tot_feats, tot_labels)


# %% Test Predictions
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
        feats = np.nan_to_num(feats)
        labels = (patient_data['labels'])[:,0]
        
        # make predictions on current patient
        preds = rfc.predict(feats)
        print(preds)
        num_correct = 0
        for i in range(len(preds)):
            if preds[i] == labels[i]:
                num_correct += 1
        # print(preds)
        print(f"{patient} Accuracy: {num_correct / len(preds)}")

# %% Model Cross Validation 

# hyper parameter tuning with randomized search CV over the parameter space below
random_grid = {'rf_classifier__bootstrap': [True, False],
               'rf_classifier__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
               'rf_classifier__min_samples_leaf': [1, 2, 4],
               'rf_classifier__min_samples_split': [2, 5, 10],
               'rf_classifier__n_estimators': [130, 180, 230]}

param_search = RandomizedSearchCV(rfc, random_grid)
param_search.fit(tot_feats, tot_labels)

# %% Model Evalutations

patient_id = "ICUDataRedux_0085"
length=1

#placeholder
start = 0
end = 1

if patient_id == "ICUDataRedux_0062":
    start = 500
    end = 24000
elif patient_id == "ICUDataRedux_0085":
    start = 79
    end = 15164
elif "CNT" in patient_id:
    start = 100
    end = 10100

filename_pick = 'dataset/%s.pkl' % patient_id
f_pick = open(filename_pick, 'rb')
annots = pickle.load(f_pick)
annots = annots.sort_values(by=['start'], ignore_index=True)
# print(annots)
annots['event'] = annots['event'].apply(lambda x: 1 if x == 'seizure' else 0)
# print(annots)
f_pick.close()

print("Results for predictions from %s" % patient_id)
# metrics = EEGEvaluator.evaluate_metrics(labels, preds)
# EEGEvaluator.test_results(metrics)
stats_sz = EEGEvaluator.sz_sens(patient_id, preds, pred_length=length)
stats_non_sz = EEGEvaluator.data_reduc(patient_id, preds, pred_length=length)
EEGEvaluator.compare_outputs_plot(patient_id, preds, length=(end-start)/60, pred_length=length)

# %%
