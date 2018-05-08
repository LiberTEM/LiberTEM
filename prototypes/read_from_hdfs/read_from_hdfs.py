import os
import sys
import json
import time
import functools
import numpy as np
from dask import distributed as dd


def get_fs():
    import hdfs3

    return hdfs3.HDFileSystem('localhost', port=8020, pars={
        'input.localread.default.buffersize': str(1),
        'dfs.client.read.shortcircuit': '1',
        'input.read.default.verify': '0'
    })


def get_index(path):
    fs3 = get_fs()
    with fs3.open(path) as f:
        idx = json.load(f)
    return idx


stackheight = 8
maskcount = 8


def process_partition(part, idx, masks):
    fs3 = get_fs()
    path = os.path.join("test", part['filename'])
    frames_per_partition = part['end'] - part['start']
    num_stacks = frames_per_partition // stackheight
    orig_shape = idx['orig_shape']
    framesize = orig_shape[-1] * orig_shape[-2]

    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

    data = np.ndarray((stackheight, framesize), dtype=idx['dtype'])
    bytes_processed = 0
    with fs3.open(path, 'rb', buff=data.nbytes) as f:
        for stack in range(num_stacks):
            f.read(length=data.nbytes, out_buffer=data)
            data.dot(masks)
            bytes_processed += data.nbytes
    return bytes_processed


def make_dask_processor(idx):
    client = dd.Client("tcp://localhost:8786", processes=False)
    orig_shape = idx['orig_shape']
    framesize = orig_shape[-1] * orig_shape[-2]
    masks = np.ones((framesize, maskcount))

    def _process_with_dask(partitions):
        t1 = time.time()
        futures = client.map(
            functools.partial(process_partition, masks=masks, idx=idx),
            partitions
        )
        bytes_processed = sum(client.gather(futures))
        delta = time.time() - t1
        print("%d MB processed in %0.5fs (%.3f MB/s)" % (
            (bytes_processed // 1024 // 1024),
            delta,
            (bytes_processed // 1024 // 1024) / delta,
        ))
        return delta
    return _process_with_dask


if __name__ == "__main__":
    idx = get_index(sys.argv[1])
    processor = make_dask_processor(idx)
    delta = processor(idx['partitions'])
