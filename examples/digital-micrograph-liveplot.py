import threading
import functools

import numpy as np

from libertem import api
from libertem.executor.inline import InlineJobExecutor
from libertem.viz.gms import GMSLive2DPlot
from libertem.udf.masks import ApplyMasksUDF
from libertem.udf.sum import SumUDF
from libertem.masks import ring


# This example shows how to use live plotting in GMS.
# It is based on digital-micrograph-inline.py.
# See this and other examples for general info on running LiberTEM in GMS.

def main():
    with api.Context(executor=InlineJobExecutor()) as ctx:
        ds = ctx.load(
            "RAW",
            path=r"C:\Users\Dieter\testfile-32-32-32-32-float32.raw",
            nav_shape=(32, 32),
            sig_shape=(32, 32),
            dtype=np.float32
        )

        sum_udf = SumUDF()

        ring_udf = ApplyMasksUDF(
            mask_factories=[functools.partial(
                ring,
                centerX=16,
                centerY=16,
                imageSizeX=32,
                imageSizeY=32,
                radius=15,
                radius_inner=11,
            )]
        )

        live_sum = GMSLive2DPlot(DM=DM, dataset=ds, udf=sum_udf)
        live_ring = GMSLive2DPlot(DM=DM, dataset=ds, udf=ring_udf)

        live_sum.display()
        live_ring.display()

        ctx.run_udf(dataset=ds, udf=[sum_udf, ring_udf], plots=[live_sum, live_ring])


if __name__ == "__main__":
    # Start the workload and wait for it to finish
    th = threading.Thread(target=main)
    th.start()
    th.join()
