############################################################################
# Generates EEG maps for multiple patients with user-friendly interface
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
############################################################################

# Import libraries
from preprocess_dataset import IEEGDataProcessor
import argparse


# Initializes the argparser object based on user input
# Inputs
#   None
# Outputs
#   parser - the argparser object containing relevant inputs for preprocessing the data
def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--username', required=False, help='username')
    parser.add_argument('-p', '--password', required=False, help='password')
    parser.add_argument('-id', '--patient_id', required=False, help='patient_id')
    parser.add_argument('-n', '--num_iter', required=False, help='num_iter')
    parser.add_argument('-b', '--num_batch', required=False, help='num_batch')
    parser.add_argument('-s', '--start', required=False, help='start')
    parser.add_argument('-l', '--length', required=False, help='length')
    parser.add_argument('-f', '--filter', required=False, help='filter')
    parser.add_argument('-eo', '--eeg_only', required=False, help='eeg_only')
    parser.add_argument('-no', '--normalize', required=False, help='normalize')
    return parser


# Main method of the file - runs map generation for user-designated EEG segments
if __name__ == "__main__":
    parser_main = init()
    args = parser_main.parse_args()
    # Check whether the user had pre-specified the parameters
    if args.username is None:
        username = input('Enter the IEEG username: ')
        password = input('Enter the IEEG password: ')
        patient_id = input('Enter the patient ID: ')
        num_iter = input('Enter the number of iterations: ')
        num_batch = input('Enter the number of batches: ')
        start = input('Enter the starting point, in seconds: ')
        length = input('Enter the length of each EEG segment, in seconds: ')
        use_filter = input('Enter 1 to apply a bandpass filter (0.5-20 Hz) and 0 otherwise: ') > '0'
        eeg_only = input('Enter 1 to only use EEG channels and 0 otherwise: ') > '0'
        normalize = input('Enter 1 to apply normalization and 0 otherwise: ') > '0'
        seizure = input('Enter 1 to indicate that the patient has seizure and 0 otherwise') > '0'
    else:
        username = args.username
        password = args.password
        patient_id = args.patient_id
        num_iter = args.num_iter
        num_batch = args.num_batch
        start = args.start
        length = args.length
        use_filter = args.filter > '0'
        eeg_only = args.eeg_only > '0'
        normalize = args.normalize > '0'
        seizure = 'RID' in patient_id
    # Create the IEEGDataProcessor object and generate the map
    dataset = IEEGDataProcessor(patient_id, username, password)
    dataset.generate_map(int(num_iter), int(num_batch), int(start), int(length), use_filter, eeg_only, normalize,
                         seizure)
