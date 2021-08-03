# %% Imports and parameters

from evaluate import EEGEvaluator
from load_dataset import IEEGDataLoader
import numpy as np
import pickle
from evaluate import EEGEvaluator
import pandas as pd

length = 1
save = True

bipolar = False
pool = False
random_forest = True

fig_filename = 'output_figs/%s_outputs_labels_wt'
if bipolar:
    fig_filename += '_bipolar'
if pool:
    fig_filename += '_pool'
if random_forest:
    fig_filename += '_rf'
fig_filename += '.pdf'

if random_forest:
    pred_filename = 'deployment_rf/pred_data/%s_predictions_rf_1s'
else:
    pred_filename = 'deployment/%s_predictions_ICU-EEG-conv-50'
if bipolar:
    pred_filename += '_bipolar'
if pool:
    pred_filename += '_pool'
pred_filename += '.npy'

# %%  Single patient testing

patient_id = "ICUDataRedux_0054"

start_stop_df = pickle.load(open("dataset/patient_start_stop.pkl", 'rb'))
patient_times = start_stop_df[start_stop_df['patient_id'] == patient_id].values
start = patient_times[0,1]
stop = patient_times[-1,2]

pred_file = open(pred_filename % patient_id, 'rb')
preds = np.load(pred_file)
if length == 60:
    preds = np.nanmax(preds, 1)
elif length == 1:
    preds = preds.flatten()
# preds = EEGEvaluator.postprocess_outputs(preds, length)
print("Predictions:")
print(preds)
print(f"Shape of predictions: {preds.shape}")
pred_file.close()

filename_pick = 'dataset/%s.pkl' % patient_id
f_pick = open(filename_pick, 'rb')
annots = pickle.load(f_pick)
annots = annots.sort_values(by=['start'], ignore_index=True)
# print(annots)
annots['event'] = annots['event'].apply(lambda x: 1 if x == 'seizure' else 0)
# print(annots)
f_pick.close()


labels = EEGEvaluator.annots_pkl_to_1D(filename_pick, start, stop, pred_length=length)
labels = labels[:preds.shape[0]]
print("Labels:")
print(labels)
print(f"Shape of labels: {labels.shape}")

# %%
# Evaluate predictions

print("Results for predictions from %s" % patient_id)
# metrics = EEGEvaluator.evaluate_metrics(labels, preds)
# EEGEvaluator.test_results(metrics)
stats_sz_sens = EEGEvaluator.sz_sens(patient_id, preds, pred_length=length)
stats_data_reduc = EEGEvaluator.data_reduc(patient_id, preds, pred_length=length)
stats_false_alerts = EEGEvaluator.false_alert_rate(patient_id, preds, pred_length=length)
EEGEvaluator.compare_outputs_plot(patient_id, preds, length=(stop-start)/60, pred_length=length)

# %% Looped Evalauations

pt_list = [
           'CNT684', 'CNT685', 'CNT687', 'CNT688',
           'CNT689', 'CNT690', 'CNT691', 'CNT692',
           'CNT694', 'CNT695', 'CNT698', 'CNT700',
           'CNT701', 'CNT702', 'CNT705', 'CNT706',
           'ICUDataRedux_0054', 'ICUDataRedux_0061', 'ICUDataRedux_0062',
           'ICUDataRedux_0063', 'ICUDataRedux_0064', 'ICUDataRedux_0065',
           'ICUDataRedux_0068', 'ICUDataRedux_0069', 'ICUDataRedux_0072',
           'ICUDataRedux_0073', 'ICUDataRedux_0074', 'ICUDataRedux_0078',
           'ICUDataRedux_0082', 'ICUDataRedux_0083', 'ICUDataRedux_0084',
           'ICUDataRedux_0085', 'ICUDataRedux_0086', 'ICUDataRedux_0087',
           'ICUDataRedux_0089', 'ICUDataRedux_0090', 'ICUDataRedux_0091'
           ]

start_stop_df = pickle.load(open('dataset/patient_start_stop.pkl', 'rb'))

for pt in pt_list:
    # get start and stop times
    patient_times = start_stop_df[start_stop_df['patient_id'] == pt].values
    start = patient_times[0,1]
    end = patient_times[-1,2]

    # load in predictions
    try:
        pred_file = open(pred_filename % pt, 'rb')
    except FileNotFoundError:
        print(f'{pt} predictions not found. Skipping patient.')
        continue
    preds = np.load(pred_file)
    if length == 60:
        preds = np.nanmax(preds, 1)
    elif length == 1:
        preds = preds.flatten()
    # preds = EEGEvaluator.postprocess_outputs(preds, length)
    # print("Predictions:")
    # print(preds)
    # print(f"Shape of predictions: {preds.shape}")
    pred_file.close()

    # get labels
    filename_pick = 'dataset/%s.pkl' % pt
    f_pick = open(filename_pick, 'rb')
    annots = pickle.load(f_pick)
    annots = annots.sort_values(by=['start'], ignore_index=True)
    # print(annots)
    annots['event'] = annots['event'].apply(lambda x: 1 if x == 'seizure' else 0)
    # print(annots)
    f_pick.close()

    # perform evaluations
    print("Results for predictions from %s" % pt)
    # metrics = EEGEvaluator.evaluate_metrics(labels, preds)
    # EEGEvaluator.test_results(metrics)
    stats_sz_sens = EEGEvaluator.sz_sens(pt, preds, pred_length=length)
    stats_data_reduc = EEGEvaluator.data_reduc(pt, preds, pred_length=length)
    stats_false_alerts = EEGEvaluator.false_alert_rate(pt, preds, pred_length=length)
    EEGEvaluator.compare_outputs_plot(pt, preds, length=(end-start)/60, pred_length=length, save=save, filename=fig_filename)
# %%
