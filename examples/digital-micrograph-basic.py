import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import multiprocessing

from libertem import api


# Since the interpreter is embedded, we have to set the Python executable.
# Otherwise we'd spawn new instances of Digital Micrograph instead of workers.
multiprocessing.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))

if __name__ == "__main__":

    with api.Context() as ctx:

        ds = ctx.load(
            "EMPAD",
            path=("C:/Users/weber/Nextcloud/Projects/Open Pixelated STEM framework/"
            "Data/EMPAD/acquisition_12.xml")
        )

        sum_analysis = ctx.create_sum_analysis(dataset=ds)
        sum_result = ctx.run(sum_analysis)

        sum_image = DM.CreateImage(sum_result.intensity.raw_data.copy())
        sum_image.ShowImage()

        haadf_analysis = ctx.create_ring_analysis(dataset=ds)
        haadf_result = ctx.run(haadf_analysis)

        haadf_image = DM.CreateImage(haadf_result.intensity.raw_data.copy())
        haadf_image.ShowImage()
