import argparse
import getpass
import numpy as np
from ieeg.auth import Session


def main():
    """
    Prints requested data
    """
    parser = argparse.ArgumentParser()
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
        dataset_name = 'RID0061'
        dataset1 = session.open_dataset(dataset_name)
        channels = list(range(len(dataset1.ch_labels)))

        # 1 minute sample from 2000 seconds for RID0061
        first_patient_data = dataset1.get_data(2000 * 1e6, 60 * 1e6, channels)
        print(dataset1.name, first_patient_data[:10, channel_idx])

        # Close dataset for RID0061
        session.close_dataset(dataset_name)

        # Obtain data from second patient RID0062
        dataset_name = 'RID0062'
        dataset2 = session.open_dataset(dataset_name)
        channels = list(range(len(dataset2.ch_labels)))

        # 1 minute sample from 2000 seconds for RID0062
        second_patient_data = dataset2.get_data(2000 * 1e6, 60 * 1e6, channels)
        print(dataset2.name, second_patient_data[:10, channel_idx])

        # Should return false since the dataset is from different patients, but returns True
        print(np.array_equal(first_patient_data[:, channel_idx], second_patient_data[:, channel_idx]))


if __name__ == "__main__":
    main()
