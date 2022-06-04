import dask
import dask.array

import numpy as np
import contextlib
from functools import partial
from typing import TYPE_CHECKING, Tuple, Optional, List, Dict

from libertem.io.dataset.base.partition import Partition
from libertem.common.shape import Shape
from libertem.common.slice import Slice

from libertem.io.dataset.dask import array_mult

if TYPE_CHECKING:
    from libertem.io.dataset.base.dataset import DataSet


def make_dask_array(dataset: 'DataSet',
                    dtype='float32',
                    roi: Optional[np.ndarray] = None,
                    min_blocks: int = 1) -> Tuple[dask.array.Array, Dict]:
    '''
    Create a Dask array using the DataSet's partitions as blocks.

    Forces a partition structure which splits the nav dims evenly
    to avoid rechunking when the flat dataset is reshaped into its
    multi-dimensional form.

    Use of an ROI will still cause rechunking issues and would be
    problematic to solve in a general way.

    Currently the partition structure is enforced by monkeypatching
    Partition.make_slices and therefore will do nothing for any
    dataset which defines its own Partition with a custom make_slices
    or does not use the default pathway:
        i.e. ds.get_slices() -> partition.make_slices()
    when creating partitions. In this case the chunking of the
    dask array may be sub-optimal.
    '''
    slices = _get_aligned_slices(dataset, min_blocks=min_blocks)
    make_slices = partial(_force_make_slices, slices)
    with _patch_make_slices(make_slices):
        return _make_dask_array(dataset, dtype=dtype, roi=roi)


def _make_dask_array(dataset: 'DataSet',
                     dtype='float32',
                     roi: Optional[np.ndarray] = None) -> Tuple[dask.array.Array, Dict]:
    """
    Create a Dask array using the DataSet's partitions as blocks.

    See make_dask_array for docstring

    Should be called from make_dask_array() to ensure aligned chunking
    but could be used directly to bypass that functionality
    """
    chunks = []
    workers = {}
    for p in dataset.get_partitions():
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


def _get_aligned_slices(dataset: 'DataSet', min_blocks: int = 1) -> List[Tuple[int, int]]:
    """
    For a given dataset, try to find a chunk structure which
    splits the navigation dimensions evenly in a way compatible
    with Partitions built from the flat navigation dimension

    Ensures at least max(min_blocks, n_cores) are created to
    to match the behaviour of Dataset.get_num_partitions()

    In contrast to ds.get_num_partitions(), this function will
    ensure every partition contains less than dataset.MAX_PARTITION_SIZE
    bytes (if read as float32), whereas that function will
    allow the next partition size beyond dataset.MAX_PARTITION_SIZE
    (essentially due to the use of floored division).

    Parameters
    ----------
    dataset : DataSet
        The dataset to chunk
    min_blocks : int, optional
        The minimum number of blocks to create, by default 1,
        but will be overriden if n_cores > min_blocks

    Returns
    -------
    List[Tuple[int, int]]
        The flat nav-dimension frame slices [(start, stop), ...]
    """
    # following .get_num_partitions(), this can cause over-partitioning
    # when the dataset raw dtype is smaller than float32
    dtype_size_bytes = 4
    sigsize = dataset.shape.sig.size * dtype_size_bytes
    ideal_frames_per_part = dataset.MAX_PARTITION_SIZE / sigsize
    min_blocks = max(dataset._cores, min_blocks)
    return _flat_slices_for_chunking(dataset.shape.nav,
                                     ideal_frames_per_part,
                                     min_blocks=min_blocks)


def _flat_slices_for_chunking(shape, max_chunksize, min_blocks=1):
    """
    Get the (start, stop) slices into flattened shape which obey
    max_chunksize, min_blocks and align onto the dimensions of shape

    Parameters
    ----------
    shape : Tuple[int, ...]
        The shape to split
    max_chunksize : int
        The maximum number of elements allowed per chunk
    min_blocks : int, optional
        The minimum number of blocks, by default 1

    Returns
    -------
    List[Tuple[int, int]]
        The flat nav-dimension frame slices [(start, stop), ...]
    """
    chunks = _chunks_for_target_size(shape,
                                     max_chunksize,
                                     min_blocks=min_blocks)
    chunksizes = array_mult(*_apply_chunking(shape, chunks))
    frame_numbers = [0] + np.cumsum(chunksizes).astype(int).tolist()
    return [(start, stop) for start, stop in zip(frame_numbers[:-1], frame_numbers[1:])]


def _apply_chunking(shape: Tuple[int, ...], chunking: Tuple[int, ...]) -> Tuple[Tuple[int]]:
    """
    Split shape into a number of chunks along each dimension
    When a split does not even divide a dimension, the remainder
    is distributed along the chunks of that dimension

    Parameters
    ----------
    shape : Tuple[int, ...]
        The shape to split
    chunking : Tuple[int, ...]
        The number of chunks to split each dimension of shape into
        Must have the same length as shape

    Returns
    -------
    Tuple[Tuple[int]]
        The chunksizes along each dimension, same length as shape
    """
    sizes = []
    for dim, nchunks in zip(shape, chunking):
        _sizes = [dim // nchunks] * nchunks
        spare = dim % nchunks
        # Distribute the remainder
        for _idx, _ in enumerate(range(spare)):
            _sizes[_idx % nchunks] += 1
        sizes.append(tuple(_sizes))
    return tuple(sizes)


def _chunks_for_target_size(shape: Tuple[int, ...],
                            max_chunksize: int,
                            min_blocks: int = 1) -> Tuple[int]:
    """
    Find the chunking to split shape such that no individual chunk
    contains more than max_chunksize elements

    This is applied strictly, so a block with max_chunksize + 1
    elements will cause the number of chunks on the active dimension
    to be incremented. This could be modified to return the previous
    solution if a smaller number of partitions is preferred.

    Equally, ensure that at least min_blocks chunks are
    created in total.

    Parameters
    ----------
    shape : Tuple[int, ...]
        The shape to split
    max_chunksize : int
        The maximum number of elements allowed per chunk
    min_blocks : int, optional
        The minimum number of blocks, by default 1

    Returns
    -------
    Tuple[int]
        The number of chunks per dimension, same length as shape
    """
    _chunking = [1] * len(shape)
    _increment = 0
    _chunksizes = array_mult(*_apply_chunking(shape, _chunking))
    while _chunksizes.max() > max_chunksize or _chunksizes.size < min_blocks:
        if _increment >= len(shape):
            break
        if _chunking[_increment] < shape[_increment]:
            _chunking[_increment] += 1
        else:
            _increment += 1
            continue
        _chunksizes = array_mult(*_apply_chunking(shape, _chunking))
    return tuple(_chunking)


def _force_make_slices(slices: List[Tuple[int, int]], shape, num_partitions, sync_offset=0):
    """
    A version of Partition.make_slices with manual slicing

    Uses pre-defined slice tuples to create partition slices
    """
    for (start, stop) in slices:
        part_slice = Slice(
            origin=(start,) + tuple([0] * shape.sig.dims),
            shape=Shape(((stop - start),) + tuple(shape.sig),
                        sig_dims=shape.sig.dims)
        )
        yield part_slice, start + sync_offset, stop + sync_offset


@contextlib.contextmanager
def _patch_make_slices(fn):
    """
    Temporarily over-ride Partition.make_slices with fn
    """
    _original_fn = Partition.make_slices
    Partition.make_slices = fn
    yield
    Partition.make_slices = _original_fn
