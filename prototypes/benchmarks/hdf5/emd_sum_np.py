import sys
import time
import h5py
import numpy as np

emd_filename_list = sys.argv[1:]
emd_filename_list.sort()

for emd_filename in emd_filename_list:
    t0 = time.time()
    f = h5py.File(emd_filename, 'r')
    # eagerly load data
    data = f['experimental/science_data']['data'].value
    t1 = time.time()
    result = np.ndarray((128, 128))

    for x in range(128):
        for y in range(128):
            result[x, y] = data[x, y].sum()

    print("\n{}".format(emd_filename))
    t2 = time.time()
    delta = t2 - t0
    print("init", t1 - t0)
    print(delta)
    print("{} MB/s (overall)".format(data.nbytes / delta / 1024 / 1024))
    print("{} MB/s (without init)".format(data.nbytes / (t2 - t1) / 1024 / 1024))
