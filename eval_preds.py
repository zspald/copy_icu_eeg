# %% Imports and parameters

from evaluate import EEGEvaluator
from load_dataset import IEEGDataLoader
import numpy as np
import pickle
from evaluate import EEGEvaluator
import pandas as pd
import matplotlib.pyplot as plt

length = 3
save = True

bipolar = False
pool = False
ref_and_bip = False
random_forest = True
predict_proba = 0.10

fig_filename = 'output_figs/patient_prediction_outputs/%s_outputs_labels_wt_0.45'
if ref_and_bip:
    fig_filename += '_refbip'
elif bipolar:
    fig_filename += '_bipolar'
if pool:
    fig_filename += '_pool'
if random_forest:
    fig_filename += '_rf'
predict_proba_str = '%0.2f' % predict_proba
fig_filename += '_proba' + predict_proba_str
fig_filename += '.pdf'
print(fig_filename)

if random_forest:
    pred_filename = 'deployment_rf/pred_data/%s_predictions_rf_3s_0.45'
else:
    pred_filename = 'deployment/%s_predictions_ICU-EEG-conv-50'
if ref_and_bip:
    pred_filename += '_refbip'
elif bipolar:
    pred_filename += '_bipolar'
if pool:
    pred_filename += '_pool'
pred_filename += '_proba' + predict_proba_str
pred_filename += '.npy'
print(pred_filename)

def plot_stats_by_patient(sz_sens_arr, data_reduc_arr, save=False):
    # sort values in ascending order
    sorted_sz_sens = np.sort(sz_sens_arr)
    sorted_data_reduc = np.sort(data_reduc_arr)

    # create 2x2 figure (Figure 6 from ICU EEG skeleton paper)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=[15,12])

    # plot sz sens histogram
    ax1.hist(sz_sens_arr, color='lightskyblue', edgecolor='black')
    ax1.axvline(np.mean(sz_sens_arr), color='red', label='Mean Sz Sens')
    ax1.axvline(np.median(sz_sens_arr), color='blue', label='Median Sz Sens')
    ax1.set_xlabel('Detection Rate')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Seizure Detection Rate Distribution')
    ax1.legend()

    # plot sz sens data for all patients
    ax2.bar(np.arange(len(sorted_sz_sens)), sorted_sz_sens, color='lightskyblue', edgecolor='black')
    ax2.set_xlabel('Patient')
    ax2.set_ylabel('Detection Rate')
    ax2.set_title('Seizure Detection Rates for All Patients')

    # plot data reduc histogram
    ax3.hist(data_reduc_arr, color='lightskyblue', edgecolor='black')
    ax3.axvline(np.mean(data_reduc_arr), color='red', label='Mean Sz Sens')
    ax3.axvline(np.median(data_reduc_arr), color='blue', label='Median Sz Sens')
    ax3.set_xlabel('Reduction Ratio')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Data Reduction Ratio Distribution')
    ax3.legend()

    # plot sz sens data for all patients
    ax4.bar(np.arange(len(sorted_data_reduc)),sorted_data_reduc, color='lightskyblue', edgecolor='black')
    ax4.set_xlabel('Patient')
    ax4.set_ylabel('Reduction Ratio')
    ax4.set_title('Data Reduction Ratios for All Patients')

    if save:
        fig_title = 'output_figs/summary_stats/summary_stats'
        if ref_and_bip:
            fig_title += '_refbip'
        elif bipolar:
            fig_title += '_bipolar'
        if pool:
            fig_title += '_pool'
        if random_forest:
            fig_title += '_rf'
        predict_proba_str = '%0.2f' % predict_proba
        fig_title += '_proba' + predict_proba_str
        fig_title += '.pdf'
        plt.savefig(fig_title, bbox_inches='tight')
    plt.show()

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

# remove non-sz patients from sz sens calculation and any missing patients
sz_sens_arr = sz_sens_arr[np.where(sz_sens_arr != -1)[0]]
data_reduc_arr = data_reduc_arr[np.where(data_reduc_arr != -1)[0]]
false_alert_arr = false_alert_arr[np.where(false_alert_arr != -1)[0]]

print(f'Mean sz sens: {np.mean(sz_sens_arr)} +\- {np.std(sz_sens_arr)} (SD)')
print(f'Median sz sens: {np.median(sz_sens_arr)}')
print(f'Mean data reduc: {np.mean(data_reduc_arr)} +\- {np.std(data_reduc_arr)} (SD)')
print(f'Median data reduc: {np.median(data_reduc_arr)}')
# print(f'Mean false alert rate: {np.mean(false_alert_arr)} +\- {np.std(false_alert_arr)} (SD)')
# print(f'Mean false alert rate: {np.median(false_alert_arr)}')

print('Summary Stats Visualization:')
plot_stats_by_patient(sz_sens_arr.flatten(), data_reduc_arr.flatten(), save=True)

# %%
