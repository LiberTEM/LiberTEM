import dask
import dask.array

from libertem.io.partitioner import (
    Partitioner, PartitionGenerator,
)


def make_dask_array(dataset, dtype='float32', roi=None):
    '''
    Create a Dask array using the DataSet's partitions as blocks.
    '''
    chunks = []
    workers = {}
    # FIXME: we need static partition sizing here
    # static partition sizing should be fine here,
    # as we don't know anything about downstream usage (task timing etc.)
    # and dask also doesn't care about a feedback rate
    partitioner = Partitioner(
        dataset_shape=dataset.shape,
        roi=roi,
        target_feedback_rate_hz=10,
    )
    partition_gen = PartitionGenerator(
        dataset=dataset,
        partitioner=partitioner,
    )
    for p in partition_gen:
        d = dask.delayed(p.get_macrotile)(
            dest_dtype=dtype, roi=roi
        )
        workers[d] = p.get_locations()
        chunks.append(
            dask.array.from_delayed(
                d,
                dtype=dtype,
                shape=p.slice.adjust_for_roi(roi).shape,
            )
        )
    arr = dask.array.concatenate(chunks, axis=0)
    if roi is None:
        arr = arr.reshape(dataset.shape)
    return (arr, workers)
