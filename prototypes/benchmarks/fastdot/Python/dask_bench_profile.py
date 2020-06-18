import bench_dask
from dask.dot import dot_graph
import dask.array as da
from dask.diagnostics import (
    Profiler, ResourceProfiler, CacheProfiler, visualize
)

maskcount = 16
framesize = 512**2
tilesize = 256**2
stackheight = 8

frames = bench_dask.bufsize // framesize // bench_dask.dtype_data(1).itemsize

stacks = frames // stackheight

masks = da.ones((maskcount, framesize), dtype=bench_dask.dtype_mask,
                chunks=(maskcount, tilesize))
data = da.ones((stacks*stackheight, framesize), dtype=bench_dask.dtype_data,
               chunks=(stackheight, tilesize))

with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
    result = bench_dask.iter_dot(data, masks, 1)

dot_graph(result.dask)
visualize([prof, rprof, cprof])
