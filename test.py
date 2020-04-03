import argparse
import getpass
from ieeg.auth import Session
import numpy as np


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
    # main()
    # a = np.array([np.nan, np.nan, np.nan, np.nan])
    # print(np.size(a))
    # b = np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]]])
    # print(a[:, None, None] * b)
    a = np.array([[1, np.nan], [1, 1], [np.nan, 1], [1, np.nan]])
    b = np.array([[[1, 2, 1], [1, 2, 2]], [[1, 2, 3], [1, 2, 4]], [[1, 2, 5], [1, 2, 6]], [[1, 2, 7], [1, 2, 8]]])
    print(np.shape(b))
    c = np.expand_dims(a, axis=-1) * b
    print(c)
    print(np.shape(c))
    # a = np.array([[1, 0], [1, 1], [0, 1], [1, 0]])
    # a = (1-a).astype('float')
    # a[a == 0] = np.nan
    # print(a)
    # b = np.array([[[1, 2], [0, 0]], [[1, 0], [1, 2]]])
    # a = ['a', 'b', 'c', 'd', 'e']
    # b = ['a', 'd']
    # x = np.nonzero(np.in1d(a, b))[0]
    # y = np.ones((10, 5, 10))
    # z = y[:, x, :]
    # print(np.shape(z))
    # a = np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]]])
    # b = np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]], [[1, 2], [1, 2]]]) * -1
    # c = np.array([1, 0, 1, 0]).astype(bool)
    # a = a[c]
    # print(a)
