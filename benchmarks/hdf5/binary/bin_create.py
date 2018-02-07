import numpy as np


def create_raw_file(filename):
    data = np.random.random((128, 128, 128, 128))
    data.tofile(filename)


create_raw_file("test.raw")
