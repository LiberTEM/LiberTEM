import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt

from libertem import api
from libertem.contrib.dask import make_dask_array


if __name__ == '__main__':
    with api.Context() as ctx:
        try:
            path = sys.argv[1]
        except IndexError:
            path = ('C:/Users/weber/Nextcloud/Projects/'
                    'Open Pixelated STEM framework/Data/EMPAD/'
                    'acquisition_12.xml')

        ds = ctx.load(
            'EMPAD',
            path=path
        )

        # Construct a Dask array from the dataset
        # The second return value contains information
        # on workers that hold parts of a dataset in local
        # storage to ensure optimal data locality
        dask_array, workers = make_dask_array(ds)

        # Perform calculations using the worker map.
        result = dask_array.sum(axis=(-1, -2)).compute(workers=workers)

        plt.figure()
        plt.imshow(result)
        plt.show()
