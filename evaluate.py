############################################################################
# Evaluates various deep learning models that detect seizure from EEG data
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
import matplotlib.pyplot as plt
import numpy as np

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
            print("Recall is not defined for positive outputs")
            recall_pos = np.nan
        # Compute the precision of the positive outputs
        if np.sum(predictions) > 0:
            precision_pos = seizure_correct / np.sum(predictions)
        else:
            print("Precision is not defined for positive outputs")
            precision_pos = np.nan
        # Compute the sensitivity of the negative outputs
        if np.sum(1 - labels) > 0:
            recall_neg = normal_correct / np.sum(1 - labels)
        else:
            print("Recall is not defined for negative outputs")
            recall_neg = np.nan
        # Compute the precision of the negative outputs
        if np.sum(1 - predictions) > 0:
            precision_neg = normal_correct / np.sum(1 - predictions)
        else:
            print("Precision is not defined for negative outputs")
            precision_neg = np.nan
        return accuracy, recall_pos, precision_pos, recall_neg, precision_neg

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
            if np.sum(predictions[window_pos:window_pos + window_size]) >= int(0.5 * window_size * threshold):
                predictions_outputs[window_pos:window_pos + window_size] = 1
            window_pos += window_disp
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

    # Plots training loss and accuracy for a single process
    # Inputs
    #   history: the output of the model's fit method that contains training history
    # Output
    #   plots displaying training/validation loss and accuracy of the model over epochs
    @staticmethod
    def training_curve(history):
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
