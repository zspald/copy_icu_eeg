# %%

# Import the IEEGDataProcessor and EEGLearner to test their functionalities
from preprocess_dataset import IEEGDataProcessor
from train import EEGLearner
from features_2d import EEGMap
import h5py

EEG_FEATS = ['Line Length', 'Delta Power', 'Theta Power', 'Alpha Power', 'Beta Power',
             'Skewness', 'Kurtosis', 'Envelope']

# sample_idx=3
# feat_ind=1
dataset_id='CNT684'
for feat_ind in range(8):
    for sample_idx in range(5,10):
        f = h5py.File('data/%s_data.h5' % dataset_id, 'r')
        print("Generating Referential Map for patient %s, %s Feature, Segment %d" % (dataset_id, EEG_FEATS[feat_ind], sample_idx + 1))
        map_outputs = f['maps']
        EEGMap.visualize_map(map_outputs, feat_ind, sample_idx=sample_idx)
        f.close()

        f_bip = h5py.File('data/%s_data_bipolar.h5' % dataset_id, 'r')
        print("Generating Bipolar Map for patient %s, %s Feature, Segment %d" % (dataset_id, EEG_FEATS[feat_ind], sample_idx + 1))
        map_outputs_bip = f_bip['maps']
        EEGMap.visualize_map(map_outputs_bip, feat_ind, sample_idx=sample_idx, bipolar=True)
        f_bip.close()

# %%
 