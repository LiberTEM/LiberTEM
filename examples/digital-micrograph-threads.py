import threading

import numpy as np
import DigitalMicrograph as DM

from libertem.api import Context
from libertem.viz.gms import GMSLive2DPlot
from libertem.udf.sum import SumUDF

# The dataset used in this example is available at
# https://doi.org/10.5281/zenodo.5113448
path = r"C:\Users\Dieter Weber\Downloads\20200518 165148\20200518 165148\default.hdr"

# This example uses a threaded executor which starts up very quickly
# and is perfectly adequate on desktop systems.
# It doesn't scale well to many cores due to Python's
# Global Interpreter Lock, however.

# For best performance on powerful workstations with many cores you
# can connect to an external cluster (see digital-micrograph-cluster.py).
# Starting a process-based executor such as the default Dask executor
# each time the script is run is possible, but not recommended
# because of their significant startup time.

def main():
    with Context.make_with('threads', plot_class=GMSLive2DPlot) as ctx:
        ds = ctx.load(
            "auto",
            path=path,
        )

        udf = SumUDF()
        sum_res = ctx.run_udf(dataset=ds, udf=udf, plots=True)

        haadf_analysis = ctx.create_ring_analysis(dataset=ds)
        haadf_result = ctx.run(haadf_analysis)

        haadf_image = DM.CreateImage(haadf_result.intensity.raw_data.copy())
        haadf_image.ShowImage()


# This pattern avoids issues in case the script spawns processes.
if __name__ == "__main__":
    main()
