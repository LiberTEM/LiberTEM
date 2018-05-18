Why Python?
===========

TODO: existing community, leverage implementations and fast prototyping, ...

Isn't Python slow?
------------------

Yes, but we only use Python for setting up the computation, creating buffers,
setting parameters, etc. We use it as a glue language for native parts
(libhdfs3, numpy/OpenBLAS, ...).

See for example this profile, visualized as a flamegraph:

.. image:: ./images/read_from_hdfs_profile.png

Most of the time is spent reading the file (block on the left: `sys_read`) or
actually performing the matrix multiplication (center block: anything containing `dgemm`).
The Python parts are mostly in the small (= little time) but high (= deep call stacks)
pillar on the right. The dask scheduler is also visible in the profile, but takes up
less than 2% of the total samples.

Note the `swapper` part on the right: this was a full-system profile, so unrelated
things like `swapper` or `intel_idle` are also included. 

But what about (multicore) scaling?
-----------------------------------

``numpy`` releases the GIL, so multiple threads can work at the same time. Even if
this were not the case, we could  still use the multiprocessing workers of ``dask.distributed``
and scale to multiple cores.
