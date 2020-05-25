import argparse
import getpass
from ieeg.auth import Session
import numpy as np
import os
import pickle


def main():
    """
    Prints requested data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-pt', '--patient', required=True, help='patient')
    parser.add_argument('-u', '--user', required=True, help='username')
    parser.add_argument('-p', '--password',
                        help='password (will be prompted if omitted)')

    args = parser.parse_args()

    if not args.password:
        args.password = getpass.getpass()

    with Session(args.user, args.password) as session:

        # Channel of interest
        channel_idx = 11

        # Obtain data from first patient RID0061
        dataset_name = args.patient
        dataset1 = session.open_dataset(dataset_name)
        channels = list(range(len(dataset1.ch_labels)))

        # 1 minute sample from 2000 seconds for RID0061
        first_patient_data = dataset1.get_data(2000 * 1e6, 60 * 1e6, channels)
        print(first_patient_data[:10, channel_idx])

        # Close dataset for RID0061
        session.close_dataset(dataset_name)


if __name__ == "__main__":
    print(os.cpu_count())
    # patients = ['RID0060', 'RID0061', 'RID0062', 'RID0063', 'RID0064', 'RID0065', 'RID0066'
    #             , 'RID0067', 'RID0068', 'RID0069', 'RID0072', 'RID0073', 'RID0074',
    #             'RID235_1d807e48', 'RID244_2aa72934', 'RID249_9f3b5d22']
    # x = 0
    # for patient in patients:
    #     with open('dataset/%s.pkl' % patient, 'rb') as file:
    #         dataframe = pickle.load(file)
    #     dataframe = dataframe[dataframe.event == 'seizure']
    #     print(dataframe)
    #     sz_start = np.asarray(dataframe)[:, 1]
    #     sz_stop = np.asarray(dataframe)[:, 2]
    #     x += int(np.sum(sz_stop - sz_start))
    # # print(x / len(patients))
    # x = np.array([[1, 2], [3, 2], [5, 1]])
    # y = np.array([1, 2, 3])
    # y[x[:, 1] == 2] = 0
    # print(y)
    with open('dataset/%s.pkl' % 'CNT685', 'rb') as file:
        dataframe = pickle.load(file)
    # dataframe = dataframe[dataframe.event == 'seizure']
    print(dataframe)
