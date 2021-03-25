import sys
import os
import multiprocessing
import threading

import numpy as np

from libertem import api

# This example starts a local cluster each time the script is run. This can be
# time-consuming. Using an external cluster (see digital-micrograph-cluster.py)
# or using the InlineJobExecutor (see digital-micrograph-inline.py) can speed up
# script execution.

# Change to a writable folder since Dask.Distributed uses the working directory
# as a default for scratch space, and GMS runs in C:\Windows\system32 by default.
os.chdir(os.environ['USERPROFILE'])

# Since the interpreter is embedded, we have to set the Python executable.
# Otherwise we'd spawn new instances of Digital Micrograph instead of workers.
multiprocessing.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


# The workload is wrapped into a `main()` function
# to run it in a separate background thread since using Numba
# can hang when used directly in a GMS Python background thread
def main():
    with api.Context() as ctx:
        ds = ctx.load(
            "RAW",
            path=r"C:\Users\Dieter\testfile-32-32-32-32-float32.raw",
            nav_shape=(32, 32),
            sig_shape=(32, 32),
            dtype=np.float32
        )

        sum_analysis = ctx.create_sum_analysis(dataset=ds)
        sum_result = ctx.run(sum_analysis)

        sum_image = DM.CreateImage(sum_result.intensity.raw_data.copy())
        sum_image.ShowImage()

        haadf_analysis = ctx.create_ring_analysis(dataset=ds)
        haadf_result = ctx.run(haadf_analysis)

        haadf_image = DM.CreateImage(haadf_result.intensity.raw_data.copy())
        haadf_image.ShowImage()


if __name__ == "__main__":
    # Start the workload and wait for it to finish
    th = threading.Thread(target=main)
    th.start()
    th.join()
