from typing import Optional, TYPE_CHECKING

import numpy as np
import dask
import dask.array
from libertem.io.dataset.base.dataset import DataSet

if TYPE_CHECKING:
    import numpy.typing as nt


def make_dask_array(
    dataset: DataSet,
    dtype: "nt.DTypeLike" = 'float32',
    roi: Optional[np.ndarray] = None
):
    '''
    Create a Dask array using the DataSet's partitions as blocks.
    '''
    chunks = []
    workers = {}
    # static partition sizing here, as we don't know anything about downstream
    # usage (task timing etc.) and dask also doesn't care about a feedback rate
    size = 128  # FIXME: get default partition size here!
    partition_gen = dataset.get_const_partitions(partition_size=size)
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
