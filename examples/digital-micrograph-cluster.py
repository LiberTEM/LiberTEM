import distributed  # noqa:F401
import DigitalMicrograph as DM

from libertem.api import Context
from libertem.executor.dask import DaskJobExecutor
from libertem.viz.gms import GMSLive2DPlot
from libertem.udf.sum import SumUDF

# The dataset used in this example is available at
# https://doi.org/10.5281/zenodo.5113448
path = r"C:\Users\Dieter Weber\Downloads\20200518 165148\20200518 165148\default.hdr"

# This example connects to an external Dask cluster.
# This achieves the best performance on powerful workstations with many cores
# and avoids restarting the cluster each time the script is run.
# See https://libertem.github.io/LiberTEM/deployment/clustercontainer.html#starting-a-custom-cluster
# on how to start such a cluster.

# Alternatively, you can use the threaded executor,
# see example digital-micrograph-threads.py
# This is easier to set up and perfectly adequate on most desktop systems.

# Starting a process-based executor such as the default Dask executor
# each time the script is run is possible, but not recommended
# because of their significant startup time.


def main():
    with DaskJobExecutor.connect('tcp://127.0.0.1:8786') as executor:
        ctx = Context(executor=executor, plot_class=GMSLive2DPlot)
    # If you also want to use the Dask cluster for other Dask-based computations,
    # uncomment the next two lines and replace the previous two lines with this code:
    # client = distributed.Client('tcp://127.0.0.1:8786')
    # with Context.make_with('dask-integration', plot_class=GMSLive2DPlot) as ctx:
        ds = ctx.load(
            "auto",
            path=path,
        )

        udf = SumUDF()
        sum_res = ctx.run_udf(dataset=ds, udf=udf, plots=True)  # noqa:F841

        haadf_analysis = ctx.create_ring_analysis(dataset=ds)
        haadf_result = ctx.run(haadf_analysis)

        haadf_image = DM.CreateImage(haadf_result.intensity.raw_data.copy())
        haadf_image.ShowImage()


# This pattern avoids issues in case the script spawns processes.
if __name__ == "__main__":
    main()
