# %% Imports and parameters

from evaluate import EEGEvaluator
from load_dataset import IEEGDataLoader
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

length = 3
batch_length = 60
save = True

bipolar = False
pool = False
ref_and_bip = False
random_forest = False
predict_proba = 0.40
# smooth_thresh = 0.45
# smooth_range = np.arange(0, 1.01, 0.05)
smooth_range = np.array([0.45])

if random_forest:
    fig_filename = 'output_figs/patient_prediction_outputs/rf_%s_outputs_labels_wt_0.45'
    if ref_and_bip:
        fig_filename += '_refbip'
    elif bipolar:
        fig_filename += '_bipolar'
    if pool:
        fig_filename += '_pool'
    if random_forest:
        fig_filename += '_rf'
else:
    fig_filename = 'output_figs/patient_prediction_outputs/cnn_%s_outputs_labels_wt_0.45'
predict_proba_str = '%0.2f' % predict_proba
fig_filename += '_proba' + predict_proba_str
fig_filename += '.pdf'
print(fig_filename)

if random_forest:
    pred_filename = 'deployment_rf/pred_data/%s_predictions_rf_3s_0.45'
    # pred_filename = 'deployment_rf/pred_data/%s_predictions_rf_1s'
    if ref_and_bip:
        pred_filename += '_refbip'
    elif bipolar:
        pred_filename += '_bipolar'
    if pool:
        pred_filename += '_pool'
else:
    pred_filename = 'deployment_cnn/pred_data/%s_predictions_3s_0.45'

pred_filename += '_proba' + predict_proba_str
pred_filename += '.npy'
print(pred_filename)

def plot_stats_by_patient(sz_sens_arr, data_reduc_arr, save=False, fig_name=None):
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
    ax3.axvline(np.mean(data_reduc_arr), color='red', label='Mean Data Reduc')
    ax3.axvline(np.median(data_reduc_arr), color='blue', label='Median Data Reduc')
    ax3.set_xlabel('Reduction Ratio')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Data Reduction Ratio Distribution')
    ax3.legend()

    # plot sz sens data for all patients
    ax4.bar(np.arange(len(sorted_data_reduc)),sorted_data_reduc, color='lightskyblue', edgecolor='black')
    ax4.set_xlabel('Patient')
    ax4.set_ylabel('Reduction Ratio')
    ax4.set_title('Data Reduction Ratios for All Patients')
    
    # set figure title
    if fig_name is not None:
        fig_title = fig_name.replace('_', ' ').title()
        plt.suptitle(fig_title)

    if save:
        if fig_name is None:
            fig_name = 'output_figs/summary_stats/summary_stats'
            if ref_and_bip:
                fig_name += '_refbip'
            elif bipolar:
                fig_name += '_bipolar'
            if pool:
                fig_name += '_pool'
            if random_forest:
                fig_name += '_rf'
            predict_proba_str = '%0.2f' % predict_proba
            fig_name += '_proba' + predict_proba_str

        fig_name = 'output_figs/summary_stats/' + fig_name
        fig_name += '.pdf'
        plt.savefig(fig_name, bbox_inches='tight')
    plt.show()

def plot_smoothing_thresh_curve(mean_sz_sens_to_data_reduc, median_sz_sens_to_data_reduc, save=False, fig_name=None):
    f, (ax1, ax2) = plt.subplots(1,2, figsize=[12,6])

    ax1.plot(mean_sz_sens_to_data_reduc[:,1], mean_sz_sens_to_data_reduc[:,0], marker='o')
    ax1.set_xlabel('Data Reduction (0-1)')
    ax1.set_ylabel('Seizure Sensitivity (0-1)')
    ax1.set_title('Mean Seizure Sensitivity vs Data Reduction')

    ax2.plot(median_sz_sens_to_data_reduc[:,1], median_sz_sens_to_data_reduc[:,0], marker='o')
    ax2.set_xlabel('Data Reduction (0-1)')
    ax2.set_ylabel('Seizure Sensitivity (0-1)')
    ax2.set_title('Median Seizure Sensitivity vs Data Reduction')

    if fig_name is not None:
        fig_title = fig_name.replace('_', ' ').title()
        plt.suptitle(fig_title)

    if save:
        if fig_name is None:
            fig_name = 'output_figs/summary_stats/smoothing_cruve'
            if ref_and_bip:
                fig_name += '_refbip'
            elif bipolar:
                fig_name += '_bipolar'
            if pool:
                fig_name += '_pool'
            if random_forest:
                fig_name += '_rf'
            predict_proba_str = '%0.2f' % predict_proba
            fig_name += '_proba' + predict_proba_str

        fig_name = 'output_figs/summary_stats/' + fig_name
        fig_name += '.pdf'
        plt.savefig(fig_name, bbox_inches='tight')
    plt.show()    

def get_best_smoothing_val(smooth_range, mean_sz_sens_to_data_reduc, median_sz_sens_to_data_reduc):

    sqrt_mean = np.sqrt(mean_sz_sens_to_data_reduc)
    sqrt_med = np.sqrt(median_sz_sens_to_data_reduc)

    mean_max = 0
    mean_max_ind = -1
    med_max = 0
    med_max_ind = -1
    for i in range(len(smooth_range)):
        mean_val = sqrt_mean[i,0] + sqrt_mean[i,1]
        med_val = sqrt_med[i,0] + sqrt_med[i,1]

        # print(f'Thresh: {smooth_range[i]} -> mean_val: {mean_val}')
        # print(f'Thresh: {smooth_range[i]} -> med_val: {med_val}')

        if mean_val > mean_max:
            mean_max = mean_val
            mean_max_ind = i
        if med_val > med_max:
            med_max = med_val
            med_max_ind = i

    print(f'Best Smoothing for Mean Sz Sens and Data Reduc: {smooth_range[mean_max_ind]}')
    print(f'Best Smoothing for Median Sz Sens and Data Reduc: {smooth_range[med_max_ind]}')

    return (smooth_range[mean_max_ind], mean_max_ind), (smooth_range[med_max_ind], med_max_ind)
# %%  Single patient testing

patient_id = "ICUDataRedux_0085"

start_stop_df = pickle.load(open("dataset/patient_start_stop.pkl", 'rb'))
patient_times = start_stop_df[start_stop_df['patient_id'] == patient_id].values
start = patient_times[0,1]
stop = patient_times[-1,2]
print(start)
print(stop)

pred_file = open(pred_filename % patient_id, 'rb')
preds = np.load(pred_file)
if length == 60:
    preds = np.nanmax(preds, 1)
else:
    preds = preds.flatten()
preds = EEGEvaluator.postprocess_outputs(preds, length)
preds = np.nan_to_num(preds)
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

print("Results for predictions from %s" % patient_id)
# metrics = EEGEvaluator.evaluate_metrics(labels, preds)
# EEGEvaluator.test_results(metrics)
stats_sz_sens = EEGEvaluator.sz_sens(patient_id, preds, pred_length=length)
stats_data_reduc = EEGEvaluator.data_reduc(patient_id, preds, pred_length=length)
stats_false_alerts = EEGEvaluator.false_alert_rate(patient_id, preds, pred_length=length)
EEGEvaluator.compare_outputs_plot(patient_id, preds, length=(stop-start)/60, pred_length=length)

# %% Looped Evalauations

pt_list_nsz = np.array([
                        # 'CNT684', 
                        'CNT685', 'CNT687', 'CNT688', 'CNT689', 'CNT690', 'CNT691', 
                        'CNT692', 'CNT694', 'CNT695', 'CNT698', 'CNT700', 'CNT701', 'CNT702', 
                        'CNT705', 'CNT706', 'CNT708', 'CNT710', 'CNT711', 'CNT713', 
                        # 'CNT715', 
                        'CNT720', 
                        # 'CNT723', 
                        'CNT724', 'CNT725', 
                        # 'CNT726', 
                        'CNT729', 'CNT730', 
                        'CNT731', 'CNT732', 'CNT733', 'CNT734', 'CNT737', 'CNT740', 'CNT741', 
                        'CNT742', 'CNT743', 
                        # 'CNT748', 
                        'CNT750', 'CNT757', 'CNT758', 'CNT765', 
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
# pt_list = np.array(["CNT685", "ICUDataRedux_0060"])

# print(len(pt_list_nsz))
# print(len(pt_list_sz))
print(len(pt_list))
# pt_list = np.array(['CNT690', 'CNT691'])
# pt_list = np.array(['ICUDataRedux_0040'])

start_stop_df = pickle.load(open('dataset/patient_start_stop.pkl', 'rb'))
pred_map = {}
label_map = {}

# iterate over different post-processing thresholds and save association btw sz sensitivity and data reduction
mean_sz_sens_to_data_reduc = np.zeros((len(smooth_range), 2))
median_sz_sens_to_data_reduc = np.zeros((len(smooth_range), 2))
for j in range(len(smooth_range)):
    smooth_thresh = smooth_range[j]

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
        preds = np.load(pred_file, allow_pickle=False)
        if length == 60:
            preds = np.nanmax(preds, 1)
        else:
            preds = preds.flatten()
        
        # Set artifacts as inter-ictal
        preds = np.nan_to_num(preds)

        # Smooth preds based on given threshold
        preds = EEGEvaluator.postprocess_outputs(preds, length, threshold=smooth_thresh)

        # get number of prediction samples from deployment
        num_samp = int((batch_length / length) * int((end - start) / batch_length))

        print('Results for %s' % pt)

        # print("Predictions:")
        # print(preds)
        # print(f"Shape of predictions: {preds.shape}")
        pred_file.close()

        # get labels
        filename_pick = 'dataset/from_json/%s_from_json.pkl' % pt
        labels = EEGEvaluator.annots_pkl_to_1D(filename_pick, start, end, pred_length=length)
        labels = labels[:num_samp]
        labels = labels[-preds.shape[0]:]

        # labels = labels[:preds.shape[0]]
        # print("Labels:")
        # print(labels)
        # print(f"Shape of labels: {labels.shape}")
        # f_pick = open(filename_pick, 'rb')
        # annots = pickle.load(f_pick)
        # annots = annots.sort_values(by=['start'], ignore_index=True)
        # # print(annots)
        # annots['event'] = annots['event'].apply(lambda x: 1 if x == 'seizure' else 0)
        # # print(annots)
        # f_pick.close()

        pred_map[pt] = preds 
        label_map[pt] = labels

        # perform evaluations
        stats_sz_sens = EEGEvaluator.sz_sens(pt, preds, batch_length=batch_length, pred_length=length)
        if not stats_sz_sens[2] is None:
            sz_sens_arr[i] = stats_sz_sens[2]
        stats_data_reduc = EEGEvaluator.data_reduc(pt, preds, batch_length=batch_length, pred_length=length)
        data_reduc_arr[i] = stats_data_reduc[2]
        # stats_false_alerts = EEGEvaluator.false_alert_rate(pt, preds, pred_length=length)
        # false_alert_arr[i] = stats_false_alerts[1]
        # if sz_sens_arr[i] > 0.9 or data_reduc_arr[i] > 0.98:
        #     EEGEvaluator.compare_outputs_plot(pt, preds, plot_length=(end-start)/60, pred_length=length, save=save, filename=fig_filename, show=True)
        # else:
        #     EEGEvaluator.compare_outputs_plot(pt, preds, plot_length=(end-start)/60, pred_length=length, save=save, filename=fig_filename, show=False)
        EEGEvaluator.compare_outputs_plot(pt, preds, pred_length=length, plot_length=(end-start)/60, save=save, filename=fig_filename, show=False)

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

    mean_sz_sens_to_data_reduc[j, 0] = np.mean(sz_sens_arr)
    mean_sz_sens_to_data_reduc[j, 1] = np.mean(data_reduc_arr)
    median_sz_sens_to_data_reduc[j, 0] = np.median(sz_sens_arr)
    median_sz_sens_to_data_reduc[j, 1] = np.median(data_reduc_arr)

    print('Summary Stats Visualization:')
    plot_stats_by_patient(sz_sens_arr.flatten(), data_reduc_arr.flatten(), save=save, fig_name='summary_stats_referential_%.2f_predict_threshold_%.2f_smoothing' % (predict_proba, smooth_thresh))

plot_smoothing_thresh_curve(mean_sz_sens_to_data_reduc, median_sz_sens_to_data_reduc, save=save, fig_name='smoothing_curve_referential_%.2f_predict_threshold' % predict_proba)
get_best_smoothing_val(smooth_range, mean_sz_sens_to_data_reduc, median_sz_sens_to_data_reduc)


# %%
