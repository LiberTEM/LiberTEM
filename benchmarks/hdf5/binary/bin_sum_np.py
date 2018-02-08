import sys
import time
import numpy as np


for filename in sys.argv[1:]:
    t0 = time.time()
    data = np.fromfile(filename).reshape((128, 128, 128, 128))
    result = np.ndarray((128, 128))

    for x in range(128):
        for y in range(128):
            result[x, y] = data[x, y].sum()
    delta = time.time() - t0
    print(delta)
    print("{} MB/s".format(data.nbytes / delta / 1024 / 1024))
