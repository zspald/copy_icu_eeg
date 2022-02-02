#########################################################################################
# Allows the user to simulate real-time EEG seizure detection interactively from IEEG.org
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
#########################################################################################

# Import Libraries
from getpass import getpass
from process_data import IEEGDataProcessor
import argparse
import json
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm
import sys

# Disable tensorflow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model

# Pre-define EEG sample length as trained by the model
sample_len = 3 # 1

# define probability threshold to assign seizure label
predict_thresh = 0.40

# User inputs and corresponding prompts
inputs = {'username': '', 'password': '', 'patient_id': '', 'model': '', 'start': 0, 'end': 0, 'length': 0,
          'threshold': 0.45}
prompts = {'username': 'Enter the IEEG username: ', 'password': 'Enter the IEEG password: ', 'patient_id':
           'Enter the patient ID: ', 'model': 'Enter the model type (conv, conv-gru, convlstm): ',
           'start': 'Enter the starting time in seconds: ', 'end': 'Enter the ending time in seconds: ',
           'threshold': 'Enter the detection threshold (0.45 recommended): '}


# Initializes variables with external user arguments
# Inputs
#   None
# Outputs
#   parser - the argparser object containing user inputs for automated seizure detection
def __init__():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--username', required=False, help='username')
    parser.add_argument('-p', '--password', required=False, help='password')
    parser.add_argument('-id', '--patient_id', required=False, help='patient_id')
    # parser.add_argument('-m', '--model', required=False, help='model')
    parser.add_argument('-t', '--threshold', required=False, help='threshold')
    # parser.add_argument('-s', '--start', required=False, help='start')
    # parser.add_argument('-e', '--end', required=False, help='end')
    # parser.add_argument('-d', '--duration', required=False, help='segment_duration')
    parser.add_argument('-l', '--length', required=False, help='length')
    parser.add_argument('-pos', '--position', required=False, help='tqdm position')
    return parser


# Main method for the real-time detector
def __main__():
    # print('====================================================================')
    # print('===================== ICU-EEG Seizure Detector =====================')
    # print('====================================================================')
    # print('=== Developed by Penn Center for Neuroengineering & Therapeutics ===')
    # print('====================================================================')
    # print('==================== University of Pennsylvania ====================')
    # print('====================================================================')
    parser_main = __init__()
    args = parser_main.parse_args()
    # Iterate over the given arguments and fill in any missing ones
    for key, value in args.__dict__.items():
        if value is None and key != 'length' and key != 'position':
            if key == 'password':
                inputs[key] = getpass(prompts[key])
            else:
                inputs[key] = input(prompts[key])
        else:
            inputs[key] = value
    # inputs['start'], inputs['end'] = int(inputs['start']), int(inputs['end'])

    print('====================================================================')
    print("Starting processing for %s." % inputs['patient_id'])
    print('====================================================================')

    # get start and stop times for the current patient
    start_stop_df = pickle.load(open("patient_start_stop.pkl", 'rb'))
    patient_times = start_stop_df[start_stop_df['patient_id'] == inputs['patient_id']].values
    start = patient_times[0,1]
    stop = patient_times[-1,2]

    inputs['threshold'] = float(inputs['threshold'])
    # Check whether the processing is done in real-time or batch
    if args.length is None:
        length_input = None
    else:
        length_input = int(args.length)
    require_input = length_input is None or length_input < 60 or length_input > stop - start
    while require_input:
        length_input = int(input('Enter the duration of each detection (min 60 seconds): '))
        require_input = length_input is None or length_input < 60 or length_input > stop - start
    inputs['length'] = length_input

    # Iteratively load the EEG map renderings from IEEG
    processor = IEEGDataProcessor(inputs['patient_id'], inputs['username'], inputs['password'])
    channels_to_use = processor.filter_channels(eeg_only=True)
    recording_start = processor.crawl_data(0, interval_length=600, threshold=1e4, channels_to_use=channels_to_use)
    timepoint = max(recording_start, start)

    # Load in test patients dictionary
    test_pts_filename='cnn_models\model-conv\cnn_test_pts_by_fold.pkl'
    test_pts = pickle.load(open(test_pts_filename, 'rb'))

    # Determine which model to use based on the list of test patients for each model and the current patient
    model_num = -1
    for fold, pts in test_pts.items():
        if inputs['patient_id'] in pts:
            model_num = fold

    # handle inability to find proper model
    if model_num == -1:
        print("Current patient not found in list of test patients associated to models. Using model at index 0.")
        model_num = 0

    # Load in the proper rf model for the patient (to avoid predictions on a pt the model was trained on)
    model_dir ='cnn_models\model-conv\conv-fold-%d.h5' % model_num
    model = load_model(model_dir)

    # # Define the model path based on user input
    # model_name = "ICU-EEG-conv-50"
    # model_dir = 'cnn_models\\' + model_name + ".h5"  # default is set to convolutional neural network trained over 50 epochs
    # if inputs['model'] == 'conv':
    #     model_dir = 'ICU-EEG-conv-50.h5'
    # model = load_model(model_dir)

    # Initialize output
    sz_events = list()
    # Use the first 30 minutes to extract patient-specific EEG statistics
    processor.initialize_stats(1800, timepoint, sample_len)
    pred_list = []
    num_batch = int(inputs['length'] / sample_len)
    with tqdm(total=(inputs['length']*int((stop - timepoint) / inputs['length'])), desc='%s' % inputs['patient_id'], file=sys.stdout, position=int(inputs['position'])) as pbar:
        while timepoint + inputs['length'] <= stop:
            # print('--- Predictions starting from %d seconds ---' % timepoint)
            eeg_maps, eeg_indices = processor.generate_map(num_batch, timepoint, sample_len)
            # Check whether the given EEG segment is artifact
            if eeg_maps is None:
                # print("The given segment has been classified as an artifact.")
                sz_events.append(['artifact', timepoint, timepoint + inputs['length']])
                # fill preds with artifact predictions to maintain proper prediction length
                pred_list.append(np.empty((num_batch,)) * np.nan)
            else:
                predict = model.predict(eeg_maps, batch_size=np.size(eeg_maps, axis=0), verbose=0)
                # print(predict)
                # Post-process the model outputs
                # print(np.argmax(predict, 1))
                # print(predict.shape)
                predict = (predict[:,1] >= predict_thresh).astype('int')
                # print(predict)
                predict = processor.postprocess_outputs(predict, sample_len, threshold=inputs['threshold'])
                # predict = processor.postprocess_outputs(np.argmax(predict, 1), sample_len, threshold=inputs['threshold'])
                # print(predict)
                predict = processor.fill_predictions(predict, eeg_indices)
                # print(predict)
                pred_list.append(predict)
                sz_events.extend(processor.write_events(predict, timepoint, sample_len, include_artifact=True))
            timepoint += inputs['length']
            pbar.update(inputs['length'])

    # # Save the predictions into a JSON file
    # sz_events = pd.DataFrame(sz_events, columns=['event', 'start', 'stop'])
    # sz_events_json = sz_events.to_json()
    # file_path = '%s-%s-%d-%d-%ds-%d' % (inputs['patient_id'], inputs['model'], stop,
    #                                 stop, sample_len, inputs['length'])
    # print('Saving outputs to ' + file_path + '.json' + ' and ' + file_path + '.pkl')
    # with open(file_path + '.json', 'w') as file:
    #     json.dump(sz_events_json, file)
    # sz_events.to_pickle(file_path + '.pkl')

    pred_filename = "pred_data/%s_predictions_%ds_0.%s" % (inputs['patient_id'], sample_len, str(inputs['threshold'])[-2:])
    predict_thresh_str = '%.2f' % predict_thresh
    proba_suffix = '_proba0.%s' % predict_thresh_str[-2:]
    pred_filename += proba_suffix
    np.save(pred_filename + '.npy', pred_list)

    # print('====================================================================')
    # print("Real-time processing complete for %s." % inputs['patient_id'])
    # print('====================================================================')


# Run the main method for the detector
if __name__ == "__main__":
    __main__()
