import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from libertem import api
from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.raw import RawFileDataSet


# @profile
def main():
    ctx = api.Context(executor=InlineJobExecutor())

    ds = RawFileDataSet(
        path="/home/clausen/Data/EMPAD/scan_11_x256_y256.raw",
        scan_size=(256, 256),
        detector_size_raw=(130, 128),
        crop_detector_to=(128, 128),
        tileshape=(1, 8, 128, 128),
        dtype="float32"
    )
    ds.initialize()

    job = ctx.create_mask_analysis(dataset=ds, factories=[lambda: np.ones(ds.shape.sig)])

    result = ctx.run(job)


if __name__ == "__main__":
    main()
