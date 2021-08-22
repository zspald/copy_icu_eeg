# %% Imports and Parameters
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from tqdm import tqdm
import h5py
import pickle
import sys
import h5py
from evaluate import EEGEvaluator

# Patients to train random forest classifer on
pt_list_nsz = np.array(['CNT684', 'CNT685', 'CNT687', 'CNT688', 'CNT689', 'CNT690', 'CNT691', 
                        'CNT692', 'CNT694', 'CNT695', 'CNT698', 'CNT700', 'CNT701', 'CNT702', 
                        'CNT705', 'CNT706', 'CNT708', 'CNT710', 'CNT711', 'CNT713', 'CNT715', 
                        'CNT720', 'CNT723', 'CNT724', 'CNT725', 'CNT726', 'CNT729', 'CNT730', 
                        'CNT731', 'CNT732', 'CNT733', 'CNT734', 'CNT737', 'CNT740', 'CNT741', 
                        'CNT742', 'CNT743', 'CNT748', 'CNT750', 'CNT757', 'CNT758', 'CNT765', 
                        'CNT773', 'CNT774', 'CNT775', 'CNT776', 'CNT778', 'CNT782', 
                        'ICUDataRedux_0023', 'ICUDataRedux_0026', 'ICUDataRedux_0029',
                        'ICUDataRedux_0030', 'ICUDataRedux_0034', 'ICUDataRedux_0035', 
                        'ICUDataRedux_0043', 'ICUDataRedux_0044', 'ICUDataRedux_0047', 
                        'ICUDataRedux_0048'])

pt_list_sz = np.array(['ICUDataRedux_0060', 'ICUDataRedux_0061', 'ICUDataRedux_0062',
                       'ICUDataRedux_0063', 'ICUDataRedux_0064', 'ICUDataRedux_0065',
                       'ICUDataRedux_0066', 'ICUDataRedux_0067', 'ICUDataRedux_0068',
                    #    'ICUDataRedux_0069', 
                       'ICUDataRedux_0072', 'ICUDataRedux_0073',
                       'ICUDataRedux_0074', 'ICUDataRedux_0054', 'ICUDataRedux_0078',
                    #    'ICUDataRedux_0082', 
                       'ICUDataRedux_0083', 'ICUDataRedux_0084',
                       'ICUDataRedux_0085', 'ICUDataRedux_0086', 'ICUDataRedux_0087',
                       'ICUDataRedux_0089', 'ICUDataRedux_0090', 
                    #    'ICUDataRedux_0091',
                       'ICUDataRedux_0003', 'ICUDataRedux_0004', 
                    #    'ICUDataRedux_0006',
                       'CNT929', 'ICUDataRedux_0027', 'ICUDataRedux_0028', 'ICUDataRedux_0033',
                    #    'ICUDataRedux_0036', 
                       'ICUDataRedux_0040', 'ICUDataRedux_0042',
                       'ICUDataRedux_0045', 'ICUDataRedux_0049', 'ICUDataRedux_0050'])

# True if combining referential and bipolar montage, false for single montage
ref_and_bip = True

# True if using bipolar montage, false for referential montage
bipolar = False

# True if using data pooled by region, false for data in channel format
pool = False

# penalty for sz misclassification
penalty = 500 # (fron John B's matlab RF)
# class_weight = {0: 1, 1: penalty}
# class_weight = 'balanced'
class_weight = 'balanced_subsample'
n_folds = 5

# threshold for window smoothing (see postprocess_outputs in evaluate.py)
sz_thresh = 0.45

# filename for saving model
model_filename = "rf_models_wt_3s"
if ref_and_bip:
    model_filename += "_refbip"
elif bipolar:
    model_filename += "_bipolar"
if pool:
    model_filename += "_pool"
model_filename += ".npy"

# filename for saving patients associated to each model (by fold for CV)
test_pts_filename = "model_test_pts_wt_3s"
if ref_and_bip:
    test_pts_filename += "_refbip"
elif bipolar:
    test_pts_filename += "_bipolar"
if pool:
    test_pts_filename += "_pool"
test_pts_filename += ".pkl"

# format for patient data filenames
data_file_format = "data/%s_data_wt"
if ref_and_bip:
    data_file_format += "_refbip"
elif bipolar:
    data_file_format += "_bipolar"
if pool:
    data_file_format += "_pool"
data_file_format += "_rf.h5"

# %% 5-Fold CV for Model
if ref_and_bip:
    print("Combining referential and bipolar montage")
elif bipolar:
    print("Using bipolar montage")
else:
    print("Using referential montage")

if pool:
    print("Using region-pooled features")
else:
    print("Using channel-based features")

# model (hyperparameters from tuning section below)
rfc = Pipeline([
    # ('pca_feature_selection', PCA(n_components=10)),
    # ('rf_classifier', RandomForestClassifier(n_estimators=2000,
    #                                          min_samples_split=2,
    #                                          min_samples_leaf=4,
    #                                          max_features='log2',
    #                                          max_depth=100,
    #                                          class_weight=class_weight,
    #                                         #  n_jobs=-1,
    #                                          verbose=1))
    ('rf_classifier', RandomForestClassifier(n_estimators=400,
                                             class_weight=class_weight,
                                            #  n_jobs=-1,
                                             verbose=1))
    ])
# print(rfc.get_params(())

# create kfold objects to split sz and nsz data
kf_sz = KFold(n_splits=n_folds, shuffle=True)
kf_nsz = KFold(n_splits=n_folds, shuffle=True)

# get split indices for all 5 folds
sz_split = list(kf_sz.split(pt_list_sz))
nsz_split = list(kf_nsz.split(pt_list_nsz))

# iterate through folds
models = np.zeros(n_folds, dtype=object)
pts_by_fold = {}
for i in tqdm(range(n_folds), desc='Training Fold', file=sys.stdout):
    # get train split indices for ith fold
    train_idx_sz, test_idx_sz = sz_split[i]
    train_idx_nsz, test_idx_nsz = nsz_split[i]

    # construct training set from splits of sz and nsz list
    train_sz = pt_list_sz[train_idx_sz]
    train_nsz = pt_list_nsz[train_idx_nsz]
    pt_train_list = np.r_[train_sz, train_nsz]

    # construct testing set from sz and nsz splits to save with model
    test_sz = pt_list_sz[test_idx_sz]
    test_nsz = pt_list_nsz[test_idx_nsz]
    pt_test_list = np.r_[test_sz, test_nsz]

    # Initialize variable to collect feats and labels
    tot_feats = None
    tot_labels = None

    for pt in pt_train_list:
        # print(f'Patient: {pt}')
        # get data filename for current patient
        filename = data_file_format % pt
        
        # access h5 file with data to extract features
        with h5py.File(filename) as pt_data:
            # load in features and labels for current patient
            feats = (pt_data['feats'])[:]
            feats = feats.reshape(feats.shape[0], -1)
            # zero rejected (nan) channels
            feats = np.nan_to_num(feats)

            # add feats from current pt to total for fold
            if tot_feats is not None:
                tot_feats = np.r_[tot_feats, feats]
            else:
                tot_feats = feats
            # print(tot_feats.shape)

            labels = (pt_data['labels'])[:,0]
            if tot_labels is not None:
                tot_labels = np.r_[tot_labels, labels]
            else:
                tot_labels = labels
    
    # Testing number of components for pca
    # pca = PCA()
    # pca.fit_transform(tot_feats)
    # expl_var = np.array(pca.explained_variance_ratio_).cumsum()
    # print(f"PCA Fold {i}: {expl_var}")
    # print(f"Number of PCs for > 99% Var Explained {(np.where(expl_var > 0.99)[0])[0] + 1}")

    # train and save model to np array
    rfc.fit(tot_feats, tot_labels)
    models[i] = rfc

    # save test patients for current model to dictionary
    pts_by_fold[i] = pt_test_list

# save models as h5 file
np.save(model_filename, models)

# save dictionary containing the test patients for the corresponding model
with open(test_pts_filename, 'wb') as pkl_file:
    pickle.dump(pts_by_fold, pkl_file)

# %% Test Predictions

# load model array
model_folds = np.load("rf_models_wt_3s_sub.npy", allow_pickle=True)

# load test patient lists by fold
test_pts = pickle.load(open("model_test_pts_wt_3s_sub.pkl", 'rb'))

output_dict = {}
for i in range(model_folds.shape[0]):
    # # get test split indices for ith fold
    # _, test_idx_sz = sz_split[i]
    # _, test_idx_nsz = nsz_split[i]

    # # construct testing set from splits of sz and nsz list
    # test_sz = pt_list_sz[test_idx_sz]
    # test_nsz = pt_list_nsz[test_idx_nsz]
    # pt_test_list = np.r_[test_sz, train_nsz]

    # get model and test patients for current fold
    model = model_folds[i]
    pt_test_list = test_pts[i]

    for pt in pt_test_list:
        # load data from proper montage and format
        filename = "data/%s_data_wt" % pt
        if bipolar:
            filename += "_bipolar"
        if pool:
            filename += "_pool"
        filename += "_rf.h5"
        print(f"On patient: {pt}")

        # access h5 file with data to extract features
        with h5py.File(filename) as pt_data:
            # load in features and labels for current patient
            feats = (pt_data['feats'])[:]
            feats = feats.reshape(feats.shape[0], -1)
            # zero rejected (nan) channels
            feats = np.nan_to_num(feats)

            # get labels for current pt
            labels = (pt_data['labels'])[:]

            # sort labels and feats in order of timepoints rather than sz and non-sz sections
            sort_inds = np.argsort(labels[:,1])
            sorted_labels = labels[sort_inds, 0]
            sorted_feats = feats[sort_inds, :]

            # get predictions for current pt
            preds = model.predict(sorted_feats)
            # post_processed_preds = EEGEvaluator.postprocess_outputs(preds, length=1, threshold=sz_thresh)

            # save pt output data to dict as an 2xN array where row 0 is model predictions and row 1 is 
            # the actual labels. The patient name is the key for this data in the output dict
            pt_output = np.array([preds, sorted_labels])
            output_dict[pt] = pt_output
    
with open("output_dict_ref_no_pool.pkl", 'wb') as pkl_file:
    pickle.dump(output_dict, pkl_file)

# %% Metrics for model evaluation by fold

tot_metrics = np.zeros((len(pt_list_sz) + len(pt_list_nsz), 5))
idx = 0
for pt, outputs in output_dict.items():
    preds = outputs[0]
    labels = outputs[1]
    print(f'Metrics for {pt}:')
    metrics = EEGEvaluator.evaluate_metrics(labels, preds)
    tot_metrics[idx, :] = metrics
    idx+=1
    EEGEvaluator.test_results(metrics, plot=False)
    print("")

mean_metrics = np.nanmean(tot_metrics, axis=0)
print('######## Average Metrics ########')
print("Test Accuracy: ", mean_metrics[0])
print("Recall (seizure): ", mean_metrics[1])
print("Precision (seizure): ", mean_metrics[2])
print("Recall (normal): ", mean_metrics[3])
print("Precision (normal): ", mean_metrics[4])

# %% Model Evalutations
# TODO -> move the sz sensitivity and data reduc evaluations to the
# deployment_rf folder to work on the model predictions from that output

pt_id = "ICUDataRedux_0064"
length=3

preds = (output_dict[pt_id])[0]

start_stop_pkl = open("dataset/patient_start_stop.pkl", 'rb')
start_stop_df = pickle.load(start_stop_pkl)
patient_times = start_stop_df[start_stop_df['patient_id'] == pt_id].values
start = patient_times[0,1]
stop = patient_times[-1,2]

filename_pick = 'dataset/from_json/%s_from_json.pkl' % pt_id
f_pick = open(filename_pick, 'rb')
annots = pickle.load(f_pick)
annots = annots.sort_values(by=['start'], ignore_index=True)
# print(annots)
annots['event'] = annots['event'].apply(lambda x: 1 if x == 'seizure' else 0)
# print(annots)
f_pick.close()

print("Results for predictions from %s" % pt_id)
# metrics = EEGEvaluator.evaluate_metrics(labels, preds)
# EEGEvaluator.test_results(metrics)
stats_sz = EEGEvaluator.sz_sens(pt_id, preds, pred_length=length)
stats_non_sz = EEGEvaluator.data_reduc(pt_id, preds, pred_length=length)
EEGEvaluator.compare_outputs_plot(pt_id, preds, length=(stop-start)/60, pred_length=length)

# %% Hyperparameter Tuning

hpt_model = RandomForestClassifier(n_estimators=400, class_weight=class_weight, n_jobs=-1, verbose=1)
hpt_rfc = Pipeline([
    # ('pca_feature_selection', PCA(n_components=100)),
    ('rf_classifier', hpt_model)
])
# print(hpt_rfc.get_params())

# iterate over desired patients
hpt_feats = None
hpt_labels = None
hpt_list = np.r_[pt_list_nsz, pt_list_sz]
for pt in hpt_list:
    filename = "data/%s_data_wt" % pt
    if bipolar:
        filename += "_bipolar"
    if pool:
        filename += "_pool"
    filename += "_rf.h5"

    # access h5 file with data to extract features
    with h5py.File(filename) as pt_data:
        # load in features and labels for current patient
        feats = (pt_data['feats'])[:]
        feats = feats.reshape(feats.shape[0], -1)
        # zero rejected (nan) channels
        feats = np.nan_to_num(feats)

        # add feats from current pt to total for fold
        if hpt_feats is not None:
            hpt_feats = np.r_[hpt_feats, feats]
        else:
            hpt_feats = feats

        labels = (pt_data['labels'])[:,0]
        if hpt_labels is not None:
            hpt_labels = np.r_[hpt_labels, labels]
        else:
            hpt_labels = labels

# # Testing number of components for pca
# pca = PCA()
# var_thresh = 0.95
# pca.fit_transform(hpt_feats)
# expl_var = np.array(pca.explained_variance_ratio_).cumsum()
# print(f"Number of PCs for > {str(var_thresh)[-2:]}% Var Explained {(np.where(expl_var > var_thresh)[0])[0] + 1}")


# hyper parameter tuning with randomized search CV over the parameter space below
random_grid = {'rf_classifier__max_depth': [25, 50, 75, 100, 125, 150, None],
               'rf_classifier__max_features': ['auto', 'sqrt', 'log2'],
               'rf_classifier__min_samples_leaf': [1, 2, 4, 8, 12],
               'rf_classifier__min_samples_split': [2, 5, 10, 15],
               'rf_classifier__n_estimators': [300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000]}

param_search = RandomizedSearchCV(hpt_rfc, random_grid)
print('Performing random hyperparameter search...')
param_search.fit(hpt_feats, hpt_labels)
print('...done')
print('Best Params:')
print(param_search.best_params_)

######### Best Params ##############
# n_estimators = 2000    
# min_samples_split = 2 
# min_samples_leaf = 4 
# max_features = 'log_2'
# max_depth = 100

# %% Tuned Model Training

# tuned_model = RandomForestClassifier(n_estimators=130, min_samples_split=2,
#                                     min_samples_leaf=4, max_depth=90,
#                                     bootstrap=True, class_weight=class_weight,
#                                     verbose=1)
# tuned_rfc = Pipeline([
#     ('pca_feature_selection', PCA(n_components=10)),
#     ('rf_classifier', tuned_model)
# ])
# # print(rfc.get_params())

# # iterate over desired patients
# tot_feats = None
# tot_labels = None
# for patient in patient_list:
#     # load data from proper montage
#     if bipolar:
#         filename = "data/" + patient + "_data_bipolar_rf.h5"
#     else:
#         filename = "data/" + patient + "_data_rf.h5"

#     # access h5 file with data
#     with h5py.File(filename) as patient_data:
#         # load in features and labels for current patient
#         feats = (patient_data['feats'])[:]
#         feats = feats.reshape(feats.shape[0], -1)
#         feats = np.nan_to_num(feats)
#         if tot_feats is not None:
#             tot_feats = np.r_[tot_feats, feats]
#         else:
#             tot_feats = feats
#         # print(tot_feats.shape)

#         labels = (patient_data['labels'])[:,0]
#         if tot_labels is not None:
#             tot_labels = np.r_[tot_labels, labels]
#         else:
#             tot_labels = labels

# tuned_rfc.fit(tot_feats, tot_labels)


