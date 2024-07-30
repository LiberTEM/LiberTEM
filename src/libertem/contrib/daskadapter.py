from typing import Optional

from sparseconverter import ArrayBackend, NUMPY, for_backend
import dask
import dask.array

from libertem.io.dataset.base import DataSet, DataTile


def _get_tile_straight(p, dest_dtype, roi, array_backend):
    return p.get_macrotile(dest_dtype=dest_dtype, roi=roi, array_backend=array_backend)


def _get_tile_converted(p, dest_dtype, roi, array_backend):
    arr_tile = p.get_macrotile(dest_dtype=dest_dtype, roi=roi, array_backend=None)
    converted_chunk = for_backend(arr_tile.data, array_backend)
    return DataTile(
        data=converted_chunk.reshape(arr_tile.shape),
        tile_slice=arr_tile.tile_slice,
        scheme_idx=arr_tile.scheme_idx
    )


def make_dask_array(dataset: DataSet, dtype='float32', roi=None,
        array_backend: Optional[ArrayBackend] = None):
    '''
    Create a Dask array using the DataSet's partitions as blocks.

    Parameters
    ----------

    dataset
        The LiberTEM dataset to load from

    dtype
        The numpy dtype into which the data should be converted

    roi
        Restrict the dataset to this region of interest (nav-shaped mask)

    array_backend
        The array type the data should be converted to
    '''
    chunks = []
    workers = {}
    if array_backend is None:
        array_backend = NUMPY
    if array_backend in dataset.array_backends:
        get_tile = _get_tile_straight
    else:
        get_tile = _get_tile_converted

    for p in dataset.get_partitions():
        d = dask.delayed(get_tile)(
            p=p, dest_dtype=dtype, roi=roi, array_backend=array_backend
        ).data
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
