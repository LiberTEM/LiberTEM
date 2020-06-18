import sys
import time
import dask
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler, visualize
from multiprocessing.pool import ThreadPool
import hyperspy.api as hs

emd_filename_list = sys.argv[1:]
emd_filename_list.sort()

with dask.set_options(pool=ThreadPool(8)), Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
    for emd_filename in emd_filename_list:
        s = hs.load(emd_filename, lazy=True).transpose(signal_axes=(2, 3))
        t0 = time.time()
        result = s.sum()
        print(emd_filename)
        delta = time.time() - t0
        print(delta)
        print("{} MB/s".format(s.data.nbytes / delta / 1024 / 1024))

visualize([prof, rprof, cprof])
