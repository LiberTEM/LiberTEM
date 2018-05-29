from nexusformat import nexus
import dask.array as da
import hyperspy.api as hs

nexus_data = nexus.nxload('test_nexus_file.nxs')
dask_array = da.from_array(nexus_data.entry.data.data, chunks=nexus_data.entry.data.data.chunks)
s = hs.signals.Signal2D(dask_array).as_lazy()
