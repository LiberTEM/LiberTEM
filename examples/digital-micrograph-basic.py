import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import multiprocessing

from libertem import api
import numpy as np

multiprocessing.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))

if __name__ == "__main__":
    # We use the special subprocess method to create a local executor.
    # The normal method doesn't work from within an embedded interpreter
    # The context manager makes sure that the executor is closed in the end
    with api.Context() as ctx:

        ds = ctx.load(
            "raw",
            path=("C:/Users/weber/Nextcloud/Projects/Open Pixelated STEM framework/"
            "Data/EMPAD/scan_11_x256_y256.raw"),
            dtype="float32",
            scan_size=(256, 256),
            detector_size_raw=(130, 128),
            crop_detector_to=(128, 128),
        )

        DM.DoEvents()
        sum_analysis = ctx.create_sum_analysis(dataset=ds)
        sum_result = ctx.run(sum_analysis)

        DM.DoEvents()

        sum_image = DM.CreateImage(sum_result.intensity.raw_data)
        sum_image.ShowImage()
        DM.DoEvents()

        haadf_analysis = ctx.create_ring_analysis(dataset=ds)
        haadf_result = ctx.run(haadf_analysis)

        DM.DoEvents()
        haadf_image = DM.CreateImage(haadf_result.intensity.raw_data)
        haadf_image.ShowImage()
        DM.DoEvents()
