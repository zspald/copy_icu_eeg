############################################################################
# Evaluates various deep learning models that detect seizure from EEG data
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
import matplotlib.pyplot as plt
import numpy as np


# A class that contains methods for evaluating/post-processing EEG data
class EEGEvaluator:

    # Computes evaluation metrics for given predictions and labels
    # Inputs
    #   labels: a list of seizure annotations given as a 1D numpy array
    #   predictions: a list of seizure predictions given as a 1D numpy array
    # Outputs
    #   accuracy: accuracy of the outputs
    #   sensitivity_pos: sensitivity of the positive outputs (seizures)
    #   precision_pos: precision of the positive outputs
    #   sensitivity_neg: sensitivity of the negative outputs (non-seizures)
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
            sensitivity_pos = seizure_correct / np.sum(labels)
        else:
            print("Sensitivity is not defined for positive outputs")
            sensitivity_pos = np.nan
        # Compute the precision of the positive outputs
        if np.sum(predictions) > 0:
            precision_pos = seizure_correct / np.sum(predictions)
        else:
            print("Precision is not defined for positive outputs")
            precision_pos = np.nan
        # Compute the sensitivity of the negative outputs
        if np.sum(1 - labels) > 0:
            sensitivity_neg = normal_correct / np.sum(1 - labels)
        else:
            print("Sensitivity is not defined for negative outputs")
            sensitivity_neg = np.nan
        # Compute the precision of the negative outputs
        if np.sum(1 - predictions) > 0:
            precision_neg = normal_correct / np.sum(1 - predictions)
        else:
            print("Precision is not defined for negative outputs")
            precision_neg = np.nan
        return accuracy, sensitivity_pos, precision_pos, sensitivity_neg, precision_neg

    # Post-processes seizure predictions obtained from the model using a sliding window
    # Inputs
    #   predictions: a list of seizure predictions given as a 1D numpy array
    #   length: the length of every EEG sample
    #   sz_length: the minimum length of a typical seizure
    #   threshold: the minimum value for accepting a prediction as seizure
    # Outputs
    #   predictions_outputs: a list of post-processed seizure predictions
    @staticmethod
    def postprocess_outputs(predictions, length, sz_length=120, threshold=0.8):
        # Initialize outputs and parameters for the sliding window
        predictions_outputs = np.array([pred for pred in predictions])
        window_size = int(sz_length / length)
        window_stride = int(window_size / 3)
        window_pos = 0
        # Slide the window and fill in regions frequently predicted as seizure
        while window_pos + window_size <= np.size(predictions_outputs, axis=0):
            if np.sum(predictions[window_pos:window_pos + window_size]) >= int(0.5 * window_size * threshold):
                predictions_outputs[window_pos:window_pos + window_size] = 1
            window_pos += window_stride
        # Initialize a smaller window to be run over the predictions
        window_size = int(window_size / 6)
        window_pos = 0
        # Slide another window and remove outlying predictions
        while window_pos + window_size <= np.size(predictions_outputs, axis=0):
            if window_pos < window_size and np.sum(predictions[:window_pos]) <= threshold:
                predictions_outputs[:window_pos] = 0
            elif window_pos > np.size(predictions_outputs, axis=0) - window_size and \
                    np.sum(predictions[window_pos:]) <= threshold:
                predictions_outputs[window_pos:] = 0
            elif np.sum(predictions[window_pos - window_size:window_pos + window_size]) <= 2 * threshold:
                predictions_outputs[window_pos - window_size:window_pos + window_size] = 0
            window_pos += window_size
        return predictions_outputs
