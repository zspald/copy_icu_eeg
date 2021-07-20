# %% Imports and annotation conversion function

from evaluate import EEGEvaluator
from load_dataset import IEEGDataLoader
import numpy as np
import pickle
from evaluate import EEGEvaluator
import pandas as pd

# patient_id = "ICUDataRedux_0062"
patient_id = "ICUDataRedux_0085"
# patient_id = "CNT685"
length = 1
start = 79 #500
end = 15164 #24000

if patient_id == "ICUDataRedux_0062":
    start = 500
    end = 24000
elif patient_id == "ICUDataRedux_0085":
    start = 79
    end = 15164
elif "CNT" in patient_id:
    start = 100
    end = 10100

pred_filename = "deployment_rf/%s_predictions_rf_1s.npy" % patient_id
pred_file = open(pred_filename, 'rb')
preds = np.load(pred_file)
if length == 60:
    preds = np.nanmax(preds, 1)
elif length == 1:
    preds = preds.flatten()
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


labels = EEGEvaluator.annots_pkl_to_1D(filename_pick, start, end, pred_length=length)
labels = labels[:preds.shape[0]]
print("Labels:")
print(labels)
print(f"Shape of labels: {labels.shape}")

# %%
# Evaluate predictions

print("Results for predictions from %s" % patient_id)
# metrics = EEGEvaluator.evaluate_metrics(labels, preds)
# EEGEvaluator.test_results(metrics)
stats_sz = EEGEvaluator.sz_sens(patient_id, preds, pred_length=length)
stats_non_sz = EEGEvaluator.data_reduc(patient_id, preds, pred_length=length)
EEGEvaluator.compare_outputs_plot(patient_id, preds, length=(end-start)/60, pred_length=length)

# %%
