import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import multiprocessing

from libertem import api
import numpy as np

# Since the interpreter is embedded, we have to set the Python executable.
# Otherwise we'd spawn new instances of Digital Micrograph instead of workers.
multiprocessing.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))

if __name__ == "__main__":

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

        sum_analysis = ctx.create_sum_analysis(dataset=ds)
        sum_result = ctx.run(sum_analysis)

        sum_image = DM.CreateImage(sum_result.intensity.raw_data)
        sum_image.ShowImage()

        haadf_analysis = ctx.create_ring_analysis(dataset=ds)
        haadf_result = ctx.run(haadf_analysis)

        haadf_image = DM.CreateImage(haadf_result.intensity.raw_data)
        haadf_image.ShowImage()
