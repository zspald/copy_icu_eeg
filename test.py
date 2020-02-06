# This is a test file to check a user's connection to Github from local PyCharm.
from load_dataset import IEEGDataLoader
from preprocess_dataset import IEEGDataProcessor
import numpy as np
from scipy import stats

username = 'danieljkim0118'
password = 'kjm39173917#'

# Test IEEGDataLoader
# loader = IEEGDataLoader('RID0068', username, password)
# dataset = loader.dataset
# a = dataset.get_annotation_layers()
# all_keys = list(a.keys())
# b = dataset.get_annotations(all_keys[0])
# for ii in range(20):
#     print(ii)
#     print(b[ii].description)
#     print(b[ii].start_time_offset_usec)
# print()

# data = loader.load_data_batch(10, 600, 5)
# print(np.shape(data))
#
# processor = IEEGDataProcessor('CNT694', username, password)
# processor.process_data(10, 600, 5)

x = [10, 20, 30, 40, 50]
y = [1, 0, 1, 1, 0]
z = np.multiply(x, y)
print(z)
