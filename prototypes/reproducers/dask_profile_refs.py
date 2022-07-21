import tempfile
import mmap

import dask
# Set the profiling interval so short that the bug is very likely to be triggered.
# It also happens with the default interval, just more rarely.
dask.config.set({"distributed.worker.profile.interval": "1 ms"})

# The bug doesn't happen when the profiler is disabled
# dask.config.set({"distributed.worker.profile.enabled": False})


import numpy as np
import distributed


def recurse(arr, index):
    '''
    Create many references into the array on the call stack.
    '''
    if index >= len(arr):
        return
    # This slice references memory from the memory map
    data = arr[index:index+1]
    return recurse(arr, index + 1)


def work_on_mmap(mm):
    '''
    Create a NumPy array backed by the memory map
    and do some work on it.
    '''
    aa = np.frombuffer(mm, dtype=np.uint8)
    recurse(aa, 0)


def do_map():
    '''
    Entry point, function to run on distributed cluster
    '''
    with tempfile.NamedTemporaryFile() as f:
        f.write(b"abc"*100)
        f.seek(0)
        mm = mmap.mmap(f.fileno(), 0)
        work_on_mmap(mm)
        mm.close()


if __name__ == '__main__':
    for i in range(1000):
        do_map()  # works

    with distributed.Client() as client:
        for i in range(1000):
            # breaks
            future = client.submit(do_map, priority=1, pure=False)
            future.result()
