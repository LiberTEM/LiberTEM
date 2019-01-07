import os
import sys
import logging
# Disable threading, we already use multiprocessing
# to saturate the CPUs
# The variables have to be set before any numerics
# libraries are loaded.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt

from libertem import api

logging.basicConfig(level=logging.WARNING)


# Protect the entry point.
# LiberTEM uses dask, which uses multiprocessing to
# start worker processes.
# https://docs.python.org/3/library/multiprocessing.html
if __name__ == '__main__':

    # api.Context() starts a new local cluster.
    # The "with" clause makes sure we shut it down in the end.
    with api.Context() as ctx:
        try:
            path = sys.argv[1]
        except IndexError:
            path = ('C:/Users/weber/Nextcloud/Projects/'
                    'Open Pixelated STEM framework/Data/EMPAD/'
                    'scan_11_x256_y256.emd')
        ds = ctx.load(
            'hdf5',
            path=path,
            ds_path='experimental/science_data/data',
            tileshape=(1, 8, 128, 128)
        )

        (scan_y, scan_x, detector_y, detector_x) = ds.shape
        mask_shape = (detector_y, detector_x)

        # LiberTEM sends functions that create the masks
        # rather than mask data to the workers in order
        # to reduce transfers in the cluster.
        def mask(): return np.ones(shape=mask_shape)

        job = ctx.create_mask_analysis(dataset=ds, factories=[mask])

        result = ctx.run(job)

        # Do something useful with the result:
        print(result)
        print(result.mask_0.raw_data)

        # For each mask, one channel is present in the result.
        # This may be different for other analyses.
        # You can access the result channels by their key on
        # the result object:
        plt.figure()
        plt.imshow(result.mask_0.raw_data)
        plt.show()

        # Otherwise, results handle like lists.
        # For example, you can iterate over the result channels:
        raw_result_list = [channel.raw_data for channel in result]
