import argparse
import getpass
from ieeg.auth import Session
import numpy as np
import os


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
