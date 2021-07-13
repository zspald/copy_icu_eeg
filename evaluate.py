############################################################################
# Evaluates various deep learning models that detect seizure from EEG data
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.patches import Rectangle
import pandas as pd

# List of all test metrics
test_metrics = ['accuracy', 'recall (sz)', 'precision (sz)', 'recall (n)', 'precision (n)']


# A class that contains methods for evaluating/post-processing EEG data
class EEGEvaluator:

    # Computes evaluation metrics for given predictions and labels
    # Inputs
    #   labels: a list of seizure annotations given as a 1D numpy array
    #   predictions: a list of seizure predictions given as a 1D numpy array
    # Outputs
    #   accuracy: accuracy of the outputs
    #   recall_pos: recall of the positive outputs (seizures)
    #   precision_pos: precision of the positive outputs
    #   recall_neg: recall of the negative outputs (non-seizures)
    #   precision_neg: precision of the negative outputs
    @staticmethod
    def evaluate_metrics(labels, predictions):
        assert (np.size(labels, axis=0) == np.size(predictions, axis=0))
        # Compute the accuracy of the outputs
        seizure_correct = np.sum(np.multiply(labels, predictions))
        normal_correct = np.sum(np.multiply(1 - labels, 1 - predictions))
        accuracy = (seizure_correct + normal_correct) / len(labels)
        # Compute the sensitivity of the positive outputs
        if np.sum(labels) > 0:
            recall_pos = seizure_correct / np.sum(labels)
        else:
            recall_pos = np.nan
        # Compute the precision of the positive outputs
        if np.sum(predictions) > 0:
            precision_pos = seizure_correct / np.sum(predictions)
        else:
            precision_pos = np.nan
        # Compute the sensitivity of the negative outputs
        if np.sum(1 - labels) > 0:
            recall_neg = normal_correct / np.sum(1 - labels)
        else:
            recall_neg = np.nan
        # Compute the precision of the negative outputs
        if np.sum(1 - predictions) > 0:
            precision_neg = normal_correct / np.sum(1 - predictions)
        else:
            precision_neg = np.nan
        return accuracy, recall_pos, precision_pos, recall_neg, precision_neg

    # Computes seizure detection statistics from the given set of labels and predictions
    # Inputs
    #   labels: annotations from the dataset files, which contains the label (0/1),
    #           starting point and endpoint (in seconds).
    #   predictions: a list of seizure predictions given as a 1D numpy array
    #   length: minimum time interval to consider pairs of events to be separate
    #   display: whether to display statistics
    # Outputs
    #   true_positive: number of seizures correctly predicted
    #   num_seizures: total number of seizures in the annotation
    #   false_positive: number of false alerts given by the model
    #   num_alerts: total number of alerts given by the model
    @staticmethod
    def evaluate_stats(labels, predictions, length=60, display=True):
        # Initialize bookkeeping variables for seizure detection
        prev_timepoint = -1 * (length + 1)
        seizure_detected = False
        true_positive, num_seizures = 0, 0
        # Find number of seizures correctly predicted
        for idx, label in enumerate(labels):
            # Check whether the labeled seizure is a new occurrence
            if label[0] == 1 and label[1] > prev_timepoint + length:
                num_seizures += 1
                if predictions[idx] == 1:
                    seizure_detected = True
                    true_positive += 1
                else:
                    seizure_detected = False
            # Check whether the current seizure is newly detected
            elif label[0] == 1 and predictions[idx] == 1 and not seizure_detected:
                seizure_detected = True
                true_positive += 1
            # Update previous timepoint if current observation is a seizure
            if label[0] == 1:
                prev_timepoint = label[1]
        sz_spec = true_positive / num_seizures
        if display:
            print('Total number of seizures: ', num_seizures)
            print('Number of seizures detected: ', true_positive)
            print('Seizure Specificity: ', sz_spec)
        # # Initialize bookkeeping variables for seizure alerts
        # prev_timepoint = -1 * (length + 1)
        # seizure_alerted = False
        # true_negative, num_alerts = 0, 0
        # # Find number of seizures incorrectly predicted (false alerts)
        # for idx, pred in enumerate(predictions):
        #     # Check whether the seizure alert is a new occurrence
        #     if pred == 1 and labels[idx, 1] > prev_timepoint + length:
        #         num_alerts += 1
        #         if labels[idx, 0] == 1:
        #             seizure_alerted = True
        #             true_negative += 1
        #         else:
        #             seizure_alerted = False
        #     # Check whether the seizure alert overlaps with a labeled seizure
        #     elif pred == 1 and labels[idx, 0] == 1 and not seizure_alerted:
        #         seizure_alerted = True
        #         true_negative += 1
        #     # Update previous timepoint if current observation is an alert
        #     if pred == 1:
        #         prev_timepoint = labels[idx, 1]
        # false_positive = num_alerts - true_negative
        # if display:
        #     print('Total number of alerts: ', num_alerts)
        #     print('Number of false alerts: ', false_positive)
        # return true_positive, num_seizures, false_positive, num_alerts
        return true_positive, num_seizures, sz_spec

    # Method header: TODO
    @staticmethod
    def sz_sens(id, predictions, pred_length=60, display=True):
        # access seizure labels
        with open('dataset/' + id + '.pkl', 'rb') as file:
                label_df = pickle.load(file)
        label_df = label_df.sort_values(by=['start'], ignore_index=True)
        start_time = label_df.start[0]
        label_df = label_df[label_df.event == 'seizure'].reset_index(drop=True)
        num_sz = label_df.shape[0]
        
        # create seizure interval dataframe for current predictions
        pred_start_stop = EEGEvaluator.pred_to_df(predictions, start_time, 
                                    pred_length=pred_length)

        true_positive = 0
        for _, label in enumerate(label_df.values):
            start = label[1]
            stop = label[2]
            if EEGEvaluator.overlap_interval(start, stop, pred_start_stop):
                true_positive += 1

        sz_spec = true_positive / num_sz
        if display:
            print('Total number of seizures: ', num_sz)
            print('Number of seizures detected: ', true_positive)
            print('Seizure Specificity: ', sz_spec)

        return true_positive, num_sz, sz_spec

    # Method header: TODO
    @staticmethod
    def data_reduc(id, predictions, pred_length=60, display=True):
        with open('dataset/' + id + '.pkl', 'rb') as file:
                label_df = pickle.load(file)
        label_df = label_df.sort_values(by=['start'], ignore_index=True)
        start_time = label_df.start[0]
        label_df = label_df[label_df.event == 'interictal'].reset_index(drop=True)
        num_non_sz = label_df.shape[0]

        # create seizure interval dataframe for current predictions
        pred_start_stop = EEGEvaluator.pred_to_df(predictions, start_time, 
                                    pred_length=pred_length, mode='no_sz')

        true_negative = 0
        for _, label in enumerate(label_df.values):
            start = label[1]
            stop = label[2]
            if EEGEvaluator.overlap_interval(start, stop, pred_start_stop):
                true_negative += 1

        data_reduc = true_negative / num_non_sz
        if display:
            print('Total number of inter-ictal: ', num_non_sz)
            print('Number of inter-ictal detected: ', true_negative)
            print('Data Reduction: ', data_reduc)

        return true_negative, num_non_sz, data_reduc

    # function to convert prediction array to seizure interval dataframe
    @staticmethod
    def pred_to_df(predictions, start_time, pred_length=60, mode='sz'):
        # switch predictions for creation of inter-ictal df
        if mode == 'no_sz':
            predictions = 1 - predictions

        pred_df = pd.DataFrame(columns=['start', 'stop'])
        start = -1
        stop = -1

        # determine if predictions start with seizure
        if predictions[0] == 1:
            start = start_time
        prev = predictions[0]
        # loop through inner predictions and determine seizure intervals
        for i in range(1, predictions.shape[0] - 1):
            # check for seizure onset
            if predictions[i] == 1 and prev == 0:
                # record starting time
                start = i * pred_length + start_time
                prev = 1
            # check for end of seizure
            if predictions[i] == 0 and prev == 1:
                stop = i * pred_length + start_time
                pred_df = pred_df.append({'start': start, 'stop': stop}, ignore_index=True)
                prev = 0

        # determine if final predictions continues a seizure
        if predictions[-1] == 1 and predictions[-2] == 1:
            stop = (predictions.shape[0]) * pred_length + start_time
            pred_df = pred_df.append({'start': start, 'stop': stop}, ignore_index=True)
        return pred_df

    # function to check for overlap between an individual seizure interval and
    # a dataframe of seizure intervals
    @staticmethod
    def overlap_interval(start, stop, sz_df):
        # create set for range of individual interval
        check_set = set(range(int(start), int(stop+1)))

        # iterate through interval dataframe
        for _, interval in enumerate(sz_df.values):
            # create set for current interval in df
            curr_start = int(interval[0])
            curr_stop = int(interval[1]) + 1
            curr_set = set(range(curr_start, curr_stop))

            # get the intersection of the individual set and current set
            overlap = check_set & curr_set

            # if elements are in the intersection, then there is an interval overlap
            if len(overlap) > 0:
                return True

        # return false if no overlap is found
        return False


    # Visualizes the predictions of the model with regards to the labels
    # Inputs
    #   labels: a list of seizure annotations given as a 1D numpy array
    #   predictions: an array of predictions returned by the model
    #   length: length of each EEG segment
    #   patient_id: ID of the patient dataset
    #   num_points: maximum number of points to label on the time axis
    #   postprocess: whether to visualize post-processed outputs of the model
    #   threshold: the threshold probability for seizure detection
    # Outputs
    #   plots of seizure detections of the model and the clinicians
    @staticmethod
    def visualize_outputs(labels, predictions, length, patient_id, num_points=15, postprocess=True, threshold=0.8):
        raw_outputs = np.array([pred[1] for pred in predictions])
        samples_to_min = int(60 / length)
        minutes = [x for x in range(len(labels)) if x % samples_to_min == 0]
        minutes = [x for i, x in enumerate(minutes) if i % (max(1, int(len(minutes) / num_points))) == 0]
        plt.plot(labels, 'r')
        plt.plot(raw_outputs + 1e-2, 'b')
        plt.plot(np.array([threshold for _ in range(len(labels))]), 'm:')
        plt.ylim(0, 1.05)
        plt.xticks(minutes, [str(int(x / samples_to_min)) for x in minutes])
        plt.xlabel('Minutes elapsed')
        plt.ylabel('Output probability')
        plt.title('Raw Detection - %s' % patient_id)
        plt.legend(['labels', 'outputs'], loc='upper left')
        plt.show()
        if postprocess:
            # Postprocess the model's outputs
            predicts = np.argmax(predictions, axis=1)
            processed_outputs = EEGEvaluator.postprocess_outputs(predicts, length=length, threshold=threshold)
            # Plot the processed outputs of the model
            plt.plot(labels, 'r')
            plt.plot(processed_outputs + 1e-2, 'b')
            plt.ylim(0, 1.05)
            plt.xticks(minutes, [str(int(x / samples_to_min)) for x in minutes])
            plt.xlabel('Minutes elapsed')
            plt.ylabel('Output probability')
            plt.title('Processed Detection - %s' % patient_id)
            plt.legend(['labels', 'outputs'], loc='upper left')
        plt.show()

    # Post-processes seizure predictions obtained from the model using a sliding window
    # Inputs
    #   predictions: a list of seizure predictions given as a 1D numpy array
    #   length: the length of every EEG sample
    #   sz_length: the minimum length of a typical seizure
    #   threshold: the minimum value for accepting a prediction as seizure
    # Outputs
    #   predictions_outputs: a list of post-processed seizure predictions
    @staticmethod
    def postprocess_outputs(predictions, length, sz_length=30, threshold=0.8):
        # Initialize outputs and parameters for the sliding window
        predictions_outputs = np.array([pred for pred in predictions])
        window_size = int(sz_length / length)
        window_disp = int(window_size / 3)
        window_pos = 0
        # Slide the window and fill in regions frequently predicted as seizure
        while window_pos + window_size <= np.size(predictions_outputs, axis=0):
            if np.sum(predictions[window_pos:window_pos + window_size]) >= window_size * threshold:
                predictions_outputs[window_pos:window_pos + window_size] = 1
            window_pos += window_disp
        # Initialize a smaller window to be run over the predictions
        window_size = int(window_size / 6)
        window_pos = 0
        # Slide another window and remove outlying predictions
        while window_pos < np.size(predictions_outputs, axis=0):
            if window_pos < window_size and np.sum(predictions[:window_pos]) <= window_pos * threshold:
                predictions_outputs[:window_pos] = 0
            elif window_pos > np.size(predictions_outputs, axis=0) - window_size and \
                    np.sum(predictions[window_pos:]) <= len(predictions[window_pos:]) * threshold:
                predictions_outputs[window_pos:] = 0
            elif np.sum(predictions[window_pos - window_size:window_pos + window_size]) <= 2 * window_size * threshold:
                predictions_outputs[window_pos - window_size:window_pos + window_size] = 0
            window_pos += window_size
        return predictions_outputs

    # Plots training loss and accuracy for a single process
    # Inputs
    #   history: the output of the model's fit method that contains training history
    # Output
    #   plots displaying training/validation loss and accuracy of the model over epochs
    @staticmethod
    def training_curve(history):
        print('Training Loss History: ', history.history['loss'])
        print('Training Accuracy History: ', history.history['acc'])
        print('Validation Loss History: ', history.history['val_loss'])
        print('Validation Accuracy History: ', history.history['val_acc'])
        plt.plot(history.history['loss'], 'b')
        plt.plot(history.history['val_loss'], 'r')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Model loss')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
        plt.plot(history.history['acc'], 'b')
        plt.plot(history.history['val_acc'], 'r')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Model accuracy')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
        return

    # Plots training loss and accuracy for cross-validation processes
    # Inputs
    #   history_list: list of history objects returned by the model's fit method
    # Outputs
    #   plots displaying training/validation loss and accuracy of the model over epochs
    @staticmethod
    def training_curve_cv(history_list):
        num_patients = len(history_list)
        num_epochs = len(history_list[0].history['loss'])
        # Initialize loss and accuracy matrices
        train_loss = np.zeros((num_patients, num_epochs))
        train_acc = np.zeros((num_patients, num_epochs))
        # Populate loss and accuracy matrices
        for idx, history in enumerate(history_list):
            train_loss[idx, :] = history_list[idx].history['loss']
            train_acc[idx, :] = history_list[idx].history['acc']
        # Compute mean and standard deviations of the curves
        loss_mean = np.mean(train_loss, axis=0)
        loss_std = np.std(train_loss, axis=0)
        acc_mean = np.mean(train_acc, axis=0)
        acc_std = np.std(train_acc, axis=0)
        # Plot the training loss
        plt.plot(loss_mean, 'r')
        plt.fill_between(np.arange(num_epochs), loss_mean - loss_std, loss_mean + loss_std, alpha=0.3,
                         edgecolor='r', facecolor='r')
        plt.xlabel('epochs')
        plt.xticks(np.arange(num_epochs), [str(epoch) for epoch in np.arange(num_epochs) + 1])
        plt.ylabel('loss')
        plt.title('Training Loss')
        plt.show()
        # Plot the training accuracy
        plt.plot(acc_mean, 'b')
        plt.fill_between(np.arange(num_epochs), acc_mean - acc_std, acc_mean + acc_std, alpha=0.3,
                         edgecolor='b', facecolor='b')
        plt.xlabel('epochs')
        plt.xticks(np.arange(num_epochs), [str(epoch) for epoch in np.arange(num_epochs) + 1])
        plt.ylabel('accuracy')
        plt.title('Training Accuracy')
        plt.show()
        return

    # Prints and visualizes test results for a single process
    # Inputs
    #   metrics: a quintuple that contains accuracy, recall and precision for
    #            both positive and negative samples
    # Outputs
    #   command line displays and visualizations of test metrics
    @staticmethod
    def test_results(metrics):
        # Display results on the command line
        print("Test Accuracy: ", metrics[0])
        print("Recall (seizure): ", metrics[1])
        print("Precision (seizure): ", metrics[2])
        print("Recall (normal): ", metrics[3])
        print("Precision (normal): ", metrics[4])
        # Create a bar chart of the test metrics
        plt.bar(np.arange(len(metrics)), list(metrics), width=0.4, color='g')
        plt.xticks(0.2 + np.arange(len(metrics)), ['%s' % x for x in test_metrics])
        plt.yticks(0.2 * np.arange(6), np.round(0.2 * np.arange(6), decimals=1))
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.title('Test Metrics for Seizure Detection')
        plt.show()
        return

    # Prints and visualizes test results for cross-validation processes
    # Inputs
    #   metric_list: a list of quintuples that contain test metrics described above
    # Outputs
    #   command line displays and visualizations of test metrics
    @staticmethod
    def test_results_cv(metric_list):
        metric_array = np.asarray(metric_list)
        # Display results on the command line
        metric_average = np.mean(metric_array, axis=0)
        print("Test Accuracy: ", metric_average[0])
        print("Recall (seizure): ", metric_average[1])
        print("Precision (seizure): ", metric_average[2])
        print("Recall (normal): ", metric_average[3])
        print("Precision (normal): ", metric_average[4])
        # Iterate over all metrics and display the corresponding bar chart
        num_patients, num_metrics = np.shape(metric_array)
        for idx in range(num_metrics):
            title = test_metrics[idx]
            plt.bar(np.arange(num_metrics), metric_array[idx, :], width=0.4)
            plt.xticks(0.2 + np.arange(num_patients), ['%d' % x for x in np.arange(num_patients) + 1])
            plt.yticks(0.2 * np.arange(6), np.round(0.2 * np.arange(6), decimals=1))
            plt.xlabel('Patient')
            plt.ylabel(title)
            plt.title('Comparison of model %s via cross-validation' % title)
            plt.show()
        return

    # Method header: TODO
    @staticmethod
    def annots_pkl_to_1D(pkl_filename, start, end, pred_length = 1, method='any_sz', sz_thresh=0.45):
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

    # Method header: TODO
    @staticmethod
    def compare_outputs_plot(id, pred_list, length=120, pred_length=5):
        #create figure with parameters for rectangles
        fig, ax = plt.subplots()
        label_y = 2.25
        pred_y = 0
        width = pred_length / 60
        height = 2

        #access seizure labels
        with open('dataset/' + id + '.pkl', 'rb') as file:
                label_df = pickle.load(file)
        label_df = label_df.sort_values(by=['start'], ignore_index=True)
        start_time = label_df.start[0]
        bound = start_time + length*60
        label_df = label_df[label_df.event == 'seizure'].reset_index(drop=True)

        if not label_df.empty:
            label_df = label_df[label_df.start < bound]
            sz_start = np.asarray(label_df)[:, 1]
            sz_stop = np.asarray(label_df)[:, 2]

            #create label visuals from annotations
            for i in range(sz_start.shape[0]):
                start = sz_start[i] / 60
                stop = sz_stop[i] / 60
                if stop > bound:
                    stop = length
                label_width = stop - start
                ax.add_patch(Rectangle((start - (start_time/60), label_y), label_width, 
                height, edgecolor = 'r', facecolor='r', fill=True))
            
        #create prediction visuals from model output
        for idx, val in enumerate(pred_list):
            if val == 1:
                ax.add_patch(Rectangle((idx / (60 / pred_length), pred_y), width, height,
                edgecolor = 'g', facecolor='g', fill=True))

        #format plot
        plt.yticks([pred_y + height / 2, label_y + height / 2], ['Outputs', 'Labels'])
        ax.set_xlabel('Time in Recording (min)')
        ax.set_ylim(bottom=-1, top=label_y + height + 1)
        ax.set_xlim(left = 0, right=length)
        plt.show()
        return 
