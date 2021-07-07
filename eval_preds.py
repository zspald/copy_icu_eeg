# %% Imports and annotation conversion function

from evaluate import EEGEvaluator
from load_dataset import IEEGDataLoader
import numpy as np
import pickle
from evaluate import EEGEvaluator

def annots_pkl_to_1D(pkl_filename, pred_length, start, end, method='any_sz', sz_thresh=0.45):
    #load in labels from pickle file
    f_pick = open(pkl_filename, 'rb')
    data = pickle.load(f_pick)
    f_pick.close()

    # order events by start time
    data = data.sort_values(by=['start'], ignore_index=True)

    # create array to store labels
    num_pred = int((end - start) / pred_length)
    labels_1d = np.zeros(num_pred)

    start_time = start
    # predict seizure if any event in the prediction window is a seizure
    if method == 'any_sz':
        ind_pkl = 0
        ind_1d = 0
        while ind_pkl < data.shape[0] and ind_1d < labels_1d.shape[0]:
            # mark end time of prediction window
            end_time = start_time + pred_length

            # get event data from pkl file
            event = (data.event)[ind_pkl]
            event_start = (data.start)[ind_pkl]
            event_end = (data.stop)[ind_pkl]

            # if prediciton window includes the end of the event
            if end_time > event_end:
                # if event switches from ii -> sz or sz -> ii
                if end_time < (data.stop)[ind_pkl + 1]:
                    # mark as seizure
                    labels_1d[ind_1d] = 1
                # switch occurrs from ii -> unannotated or sz -> unannotated
                else:
                    # get label as the current event
                    if event == 'seizure':
                        labels_1d[ind_1d] = 1
                    else:
                        labels_1d[ind_1d] = 0
                # move to next event
                ind_pkl += 1
                # slide prediction window
                ind_1d += 1
                start_time = end_time

            # if prediction window is past current event
            elif start_time > event_end:
                # move to next event
                ind_pkl += 1

            # if end of prediction window is before the start of the event
            elif end_time < event_start:
                # mark label as nan (event switches are handled above, 
                # this accounts for when the prediction window starts before
                # any annotations)
                labels_1d[ind_1d] = np.nan

                # progress prediction window
                ind_1d += 1
                start_time = end_time

            # prediction window is completely inside the event
            else:
                # get label as the current event
                if event == 'seizure':
                    labels_1d[ind_1d] = 1
                else:
                    labels_1d[ind_1d] = 0
                ind_1d += 1
                start_time = end_time

    elif method == 'threshold':
        raise RuntimeError("Threshold method not implemented yet.")
    else:
        raise ValueError("Method must be 'any_sz' or 'threshold'.")


    return labels_1d
# %% Load in files and get predictions and labels
patient_id = "ICUDataRedux_0062"
# patient_id = "ICUDataRedux_0085"
length = 60
start = 79 #500
end = 15164 #24000

if patient_id == "ICUDataRedux_0062":
    start = 500
    end = 24000
elif patient_id == "ICUDataRedux_0085":
    start = 79
    end = 15164

pred_filename = "deployment/%s_predictions_ICU-EEG-conv-50.npy" % patient_id
pred_file = open(pred_filename, 'rb')
preds = np.load(pred_file)
# print(preds)
preds = np.nanmax(preds, 1)
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

labels = annots_pkl_to_1D(filename_pick, length, start, end)
print("Labels:")
print(labels)
print(f"Shape of labels: {labels.shape}")

# %% Evaluate predictions

print("Results for predictions from %s" % patient_id)
metrics = EEGEvaluator.evaluate_metrics(labels, preds)
EEGEvaluator.test_results(metrics)
# stats = EEGEvaluator.evaluate_stats(annots.values, preds, length=length)
# %%
