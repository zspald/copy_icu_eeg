#########################################################################################
# Allows the user to simulate real-time EEG seizure detection interactively from IEEG.org
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
#########################################################################################

# Import Libraries
from getpass import getpass
from process_data_rf import IEEGDataProcessor
import argparse
import json
import numpy as np
import os
import pandas as pd
import pickle

# Pre-define EEG sample length as trained by the model
sample_len = 3 # 1

# define probability threshold to assign seizure label
predict_thresh = 0.07

# User inputs and corresponding prompts
# inputs = {'username': '', 'password': '', 'patient_id': '', 'model': '', 'start': 0, 'end': 0, 'length': 0,
#           'threshold': 0.45}
# prompts = {'username': 'Enter the IEEG username: ', 'password': 'Enter the IEEG password: ', 'patient_id':
#            'Enter the patient ID: ', 'model': 'Enter the model type (conv, conv-gru, convlstm): ',
#            'start': 'Enter the starting time in seconds: ', 'end': 'Enter the ending time in seconds: ',
#            'threshold': 'Enter the detection threshold (0.45 recommended): '}
inputs = {'username': '', 'password': '', 'patient_id': '', 'bipolar': 0, 'pool': 0, 'ref_and_bip': 0, 'length': 0, 'threshold': 0.45}
prompts = {'username': 'Enter the IEEG username: ', 'password': 'Enter the IEEG password: ', 'patient_id':
           'Enter the patient ID: ', 'bipolar': 'Use bipolar montage? (y=1, n=0)', 'pool': 'Pool features by region? (y=1, n=0)',
           'ref_and_bip': 'Combine referential and bipolar montage? (y=1, n=0)', 'threshold': 'Enter the detection threshold (0.45 recommended):'}


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
    parser.add_argument('-b', '--bipolar', required=False, help='bipolar montage')
    parser.add_argument('-po', '--pool', required=False, help='regional pooling')
    parser.add_argument('-rb', '--ref_and_bip', required=False, help='referential and bipolar montage')
    parser.add_argument('-t', '--threshold', required=False, help='threshold')
    # parser.add_argument('-s', '--start', required=False, help='start')
    # parser.add_argument('-e', '--end', required=False, help='end')
    # parser.add_argument('-d', '--duration', required=False, help='segment_duration')
    parser.add_argument('-l', '--length', required=False, help='length')
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
        if value is None and key != 'length':
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

    # print(f"Bipolar option: {inputs['bipolar']}")
    # determine montage type
    if inputs['bipolar'] == '1':
        bipolar = True
    else:
        bipolar = False

    # set pooling settings
    if inputs['pool'] == '1':
        pool = True
    else:
        pool = False

     # set pooling settings
    if inputs['ref_and_bip'] == '1':
        print('Combining montages')
        ref_and_bip = True
    else:
        ref_and_bip = False

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
    test_pts_filename='model_test_pts_wt'
    test_pts_filename += '_%ds' % sample_len

    if ref_and_bip:
        test_pts_filename += '_refbip'
    elif bipolar:
        test_pts_filename += '_bipolar'

    if pool:
        test_pts_filename += '_pool'
    
    test_pts_filename += '.pkl'

    # Determine which model to use based on the list of test patients for each model and the current patient
    model_num = -1
    test_pts = pickle.load(open(test_pts_filename, 'rb'))
    for fold, pts in test_pts.items():
        if inputs['patient_id'] in pts:
            model_num = fold

    # handle inability to find proper model
    if model_num == -1:
        print("Current patient not found in list of test patients associated to models. Using model at index 0.")
        model_num = 0

    # Load in the proper rf model for the patient (to avoid predictions on a pt the model was trained on)
    model_filename='rf_models_wt'
    model_filename += '_%ds' % sample_len

    if ref_and_bip:
        model_filename += '_refbip'
    elif bipolar:
        model_filename += '_bipolar'

    if pool:
        model_filename += '_pool'
    
    model_filename += '.npy'
    model_arr = np.load(model_filename, allow_pickle=True)
    model = model_arr[model_num]
    model.set_params(rf_classifier__verbose = 0)

    # Initialize output
    sz_events = list()
    # Use the first 30 minutes to extract patient-specific EEG statistics
    processor.initialize_stats(1800, timepoint, sample_len, bipolar=bipolar, pool_region=pool, ref_and_bip=ref_and_bip)
    pred_list = []
    while timepoint + inputs['length'] <= stop:
        # print('--- Predictions starting from %d seconds ---' % timepoint)
        # eeg_maps, eeg_indices = processor.generate_map(inputs['length'], timepoint, sample_len)
        eeg_feats, eeg_indices = processor.process_feats(inputs['length'], timepoint, sample_len, bipolar=bipolar, pool_region=pool, ref_and_bip=ref_and_bip)
        # Check whether the given EEG segment is artifact
        if eeg_feats is None:
            print("The given segment has been classified as an artifact.")
            sz_events.append(['artifact', timepoint, timepoint + inputs['length']])
        else:
            eeg_feats = eeg_feats.reshape(eeg_feats.shape[0], -1)
            eeg_feats = np.nan_to_num(eeg_feats)
            # predict = model.predict(eeg_feats)
            predict = (model.predict_proba(eeg_feats)[:,1] >= predict_thresh).astype('int')
            # print(predict)
            # # Post-process the model outputs
            # predict = processor.postprocess_outputs(predict, sample_len, threshold=inputs['threshold'])
            # print(predict)
            predict = processor.fill_predictions(predict, eeg_indices)
            # print(predict)
            pred_list.append(predict)
            sz_events.extend(processor.write_events(predict, timepoint, sample_len, include_artifact=True))
        timepoint += inputs['length']
    # Save the predictions into a JSON file
    sz_events = pd.DataFrame(sz_events, columns=['event', 'start', 'stop'])
    sz_events_json = sz_events.to_json()
    file_path = 'pred_data/%s-rf-%d-%d-%ds-%d' % (inputs['patient_id'], start,
                                    stop, sample_len, inputs['length'])
    if ref_and_bip:
        file_path += '-refbip'
    elif bipolar:
        file_path += '-bipolar'
    if pool:
        file_path += '-pool'
    # print('Saving outputs to ' + file_path + '.json' + ' and ' + file_path + '.pkl')
    # with open(file_path + '.json', 'w') as file:
    #     json.dump(sz_events_json, file)
    # sz_events.to_pickle(file_path + '.pkl')
    pred_filename = "pred_data/%s_predictions_rf_%ds_0.%s" % (inputs['patient_id'], sample_len, str(inputs['threshold'])[-2:])
    if ref_and_bip:
        pred_filename += '_refbip'
    elif bipolar:
        pred_filename += '_bipolar'
    if pool:
        pred_filename += '_pool'
    predict_thresh_str = '%.2f' % predict_thresh
    proba_suffix = '_proba0.%s' % predict_thresh_str[-2:]
    pred_filename += proba_suffix
    np.save(pred_filename + '.npy', pred_list)
    print('====================================================================')
    print("Real-time processing complete for %s." % inputs['patient_id'])
    print('====================================================================')


# Run the main method for the detector
if __name__ == "__main__":
    __main__()
