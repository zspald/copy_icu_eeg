# %% Imports and annotation conversion function

from evaluate import EEGEvaluator
from load_dataset import IEEGDataLoader
import numpy as np
import pickle
from evaluate import EEGEvaluator
import pandas as pd

# patient_id = "ICUDataRedux_0062"
patient_id = "ICUDataRedux_0085"
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
print(annots)
annots['event'] = annots['event'].apply(lambda x: 1 if x == 'seizure' else 0)
# print(annots)
f_pick.close()


labels = EEGEvaluator.annots_pkl_to_1D(filename_pick, length, start, end)
print("Labels:")
print(labels)
print(f"Shape of labels: {labels.shape}")

# %%
# Evaluate predictions

print("Results for predictions from %s" % patient_id)
metrics = EEGEvaluator.evaluate_metrics(labels, preds)
EEGEvaluator.test_results(metrics)
stats_sz = EEGEvaluator.sz_sens(patient_id, preds, pred_length=60)
stats_non_sz = EEGEvaluator.data_reduc(patient_id, preds, pred_length=60)
EEGEvaluator.compare_outputs_plot(patient_id, preds, length=(end-start)/60, pred_length=60)

# %%
