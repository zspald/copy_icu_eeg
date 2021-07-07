import h5py
import os

# # filename = 'data/ICUDataRedux_0085_data.h5'
# filename = 'data/ICUDataRedux_0091_data.h5'

# f = h5py.File(filename, 'r')
# print(filename)
# print(f['maps'].shape[0])
# f.close()


for filename in os.listdir("data"):
    h5_filename = "data/" + filename
    f = h5py.File(h5_filename, 'r')
    try:
        print(f"{filename}: {f['maps'].shape[0]}")
        f.close()
    except KeyError:
        print(f"{filename}: Maps empty")
        f.close()

