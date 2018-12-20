import pprint
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from libertem import api
from libertem.executor.inline import InlineJobExecutor
from raw_parts import RawFilesDataSet


# @profile
def main():
    ctx = api.Context(executor=InlineJobExecutor())

    ds = RawFilesDataSet(
        path="/home/clausen/Data/many_small_files/frame00016293.bin",
        # path="/home/clausen/Data/many_medsize_files/frame00000001.bin",
        nav_shape=(256, 256),
        sig_shape=(128, 128),
        file_shape=(16, 128, 128),
        tileshape=(1, 8, 128, 128),
        dtype="float32"
    )
    ds.initialize()
    pprint.pprint(list(ds.get_partitions()))

    job = ctx.create_mask_analysis(dataset=ds, factories=[lambda: np.ones(ds.shape.sig)])

    result = ctx.run(job)


if __name__ == "__main__":
    main()
