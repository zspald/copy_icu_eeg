# %% Imports and parameters

from evaluate import EEGEvaluator
from load_dataset import IEEGDataLoader
import numpy as np
import pickle
from evaluate import EEGEvaluator
import pandas as pd

length = 3
save = True

bipolar = False
pool = False
random_forest = True

fig_filename = 'output_figs/%s_outputs_labels_wt_0.45proba05'
if bipolar:
    fig_filename += '_bipolar'
if pool:
    fig_filename += '_pool'
if random_forest:
    fig_filename += '_rf'
fig_filename += '.pdf'

if random_forest:
    pred_filename = 'deployment_rf/pred_data/%s_predictions_rf_3s_0.45proba05'
else:
    pred_filename = 'deployment/%s_predictions_ICU-EEG-conv-50'
if bipolar:
    pred_filename += '_bipolar'
if pool:
    pred_filename += '_pool'
pred_filename += '.npy'

# %%  Single patient testing

patient_id = "ICUDataRedux_0089"

start_stop_df = pickle.load(open("dataset/patient_start_stop.pkl", 'rb'))
patient_times = start_stop_df[start_stop_df['patient_id'] == patient_id].values
start = patient_times[0,1]
stop = patient_times[-1,2]
print(start)

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

filename_pick = 'dataset/from_json/%s_from_json.pkl' % patient_id
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

pt_list_nsz = np.array([
                        'CNT684', 'CNT685', 'CNT687', 'CNT688', 'CNT689', 'CNT690', 'CNT691', 
                        'CNT692', 'CNT694', 'CNT695', 'CNT698', 'CNT700', 'CNT701', 'CNT702', 
                        'CNT705', 'CNT706', 'CNT708', 'CNT710', 'CNT711', 'CNT713', 'CNT715', 
                        'CNT720', 'CNT723', 'CNT724', 'CNT725', 'CNT726', 'CNT729', 'CNT730', 
                        'CNT731', 'CNT732', 'CNT733', 'CNT734', 'CNT737', 'CNT740', 'CNT741', 
                        'CNT742', 'CNT743', 'CNT748', 'CNT750', 'CNT757', 'CNT758', 'CNT765', 
                        'CNT773', 'CNT774', 'CNT775', 'CNT776', 'CNT778', 'CNT782', 
                        'ICUDataRedux_0023', 'ICUDataRedux_0026', 'ICUDataRedux_0029',
                        'ICUDataRedux_0030', 'ICUDataRedux_0034', 'ICUDataRedux_0035', 
                        'ICUDataRedux_0043', 'ICUDataRedux_0044', 'ICUDataRedux_0047', 
                        'ICUDataRedux_0048'
                        ])

pt_list_sz = np.array([
                       'ICUDataRedux_0060', 'ICUDataRedux_0061', 'ICUDataRedux_0062',
                       'ICUDataRedux_0063', 'ICUDataRedux_0064', 'ICUDataRedux_0065',
                       'ICUDataRedux_0066', 'ICUDataRedux_0067', 'ICUDataRedux_0068',
                       'ICUDataRedux_0069', 'ICUDataRedux_0072', 'ICUDataRedux_0073',
                       'ICUDataRedux_0074', 'ICUDataRedux_0054', 'ICUDataRedux_0078',
                       'ICUDataRedux_0082', 'ICUDataRedux_0083', 'ICUDataRedux_0084',
                       'ICUDataRedux_0085', 'ICUDataRedux_0086', 'ICUDataRedux_0087',
                       'ICUDataRedux_0089', 'ICUDataRedux_0090', 'ICUDataRedux_0091',
                       'ICUDataRedux_0003', 'ICUDataRedux_0004', 'ICUDataRedux_0006',
                       'CNT929', 'ICUDataRedux_0027', 'ICUDataRedux_0028', 'ICUDataRedux_0033',
                       'ICUDataRedux_0036', 'ICUDataRedux_0040', 'ICUDataRedux_0042',
                       'ICUDataRedux_0045', 'ICUDataRedux_0049', 'ICUDataRedux_0050'
                       ])


pt_list = np.r_[pt_list_nsz, pt_list_sz]

start_stop_df = pickle.load(open('dataset/patient_start_stop.pkl', 'rb'))

sz_sens_arr = np.ones((pt_list.shape[0], 1)) * -1
data_reduc_arr = np.ones((pt_list.shape[0], 1)) * -1
false_alert_arr = np.ones((pt_list.shape[0], 1)) * -1
for i in range(pt_list.shape[0]):
    pt = pt_list[i]
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
    else:
        preds = preds.flatten()
    # preds = EEGEvaluator.postprocess_outputs(preds, length)
    # print("Predictions:")
    # print(preds)
    # print(f"Shape of predictions: {preds.shape}")
    pred_file.close()

    # get labels
    filename_pick = 'dataset/from_json/%s_from_json.pkl' % pt
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
    if not stats_sz_sens[2] is None:
        sz_sens_arr[i] = stats_sz_sens[2]
    stats_data_reduc = EEGEvaluator.data_reduc(pt, preds, pred_length=length)
    data_reduc_arr[i] = stats_data_reduc[2]
    stats_false_alerts = EEGEvaluator.false_alert_rate(pt, preds, pred_length=length)
    false_alert_arr[i] = stats_false_alerts[1]
    EEGEvaluator.compare_outputs_plot(pt, preds, length=(end-start)/60, pred_length=length, save=save, filename=fig_filename, show=False)

# remove non-sz patients from sz sens calculation
sz_sens_arr = sz_sens_arr[np.where(sz_sens_arr != -1)[0]]
data_reduc_arr = data_reduc_arr[np.where(data_reduc_arr != -1)[0]]
false_alert_arr = false_alert_arr[np.where(false_alert_arr != -1)[0]]

print(f'Mean sz sens: {np.mean(sz_sens_arr)} +\- {np.std(sz_sens_arr)} (SD)')
print(f'Mean data reduc: {np.mean(data_reduc_arr)} +\- {np.std(data_reduc_arr)} (SD)')
print(f'Mean false alert rate: {np.mean(false_alert_arr)} +\- {np.std(false_alert_arr)} (SD)')

# %%
