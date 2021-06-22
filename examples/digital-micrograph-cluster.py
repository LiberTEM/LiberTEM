import threading

import numpy as np
import DigitalMicrograph as DM

from libertem import api
from libertem.executor.dask import DaskJobExecutor

# This example connects to an external Dask.Distributed cluster. See
# https://libertem.github.io/LiberTEM/usage.html#cluster on how to start such a
# cluster. This method gives the fastest script execution in GMS for most use
# cases. Alternatively, you can use an InlineJobExecutor (see
# digital-micrograph-inline.py) or start a local cluster each time the script is
# run (see digital-micrograph-local.py).


# The workload is wrapped into a `main()` function
# to run it in a separate background thread since using Numba
# can hang when used directly in a GMS Python background thread
def main():
    with DaskJobExecutor.connect('tcp://localhost:8786') as executor:
        ctx = api.Context(executor=executor)
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
