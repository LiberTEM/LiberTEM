import numpy as np
import pathlib
import dask
import itertools
import dask.array as da


def _create_chunk(*, chunk_shape, dtype, value=1, **kwargs):
    return np.ones(chunk_shape, dtype=dtype) * value


def _mmap_load_chunk(*, dataset_shape, dtype, filename, sl, offset=0, **kwargs):
    data = np.memmap(filename, mode='r', shape=dataset_shape, dtype=dtype, offset=offset)
    return data[sl]


def _take_n(iterable, n):
    """
    Split a interable into chunks of length n with the final element
    being the remainder len < n if n does not divide evenly
    """
    len_iter = len(iterable)
    return [iterable[i: min(i + n, len_iter)] for i in range(0, len_iter, n)]


def _reshape_list(to_reshape, new_shape):
    """
    Recursively reshape a list into new_shape following C-ordering
    Unlike numpy.reshape this will happily create a ragged array when
    the dimensions in new_shape do not divide perfectly into to_reshape!
    """
    left = new_shape[:-1]
    n = new_shape[-1]
    reshaped = _take_n(to_reshape, n)
    if len(left) == 0:
        return reshaped[0]
    else:
        return _reshape_list(reshaped, left)


def _blocksize_to_dim_slices(blocksize, dim):
    """
    Generate the slices along a dimension of length dim
    to split dim into chunks of length blocksize, with the final chunk
    being the remainder if dim % blocksize != 0

    A blocksize of -1 is converted to a None slice
    """
    slices = []
    if blocksize == -1:
        slices.append(slice(None))
    else:
        for index in range(0, dim, blocksize):
            chunk_size = min(blocksize, dim - index)
            slices.append(slice(index, index + chunk_size))
    return slices


def _slices_to_chunk_shape(slices, full_shape):
    """
    Generate the shape tuple of the array resulting from applying
    the sequence of slices to an array with shape full_shape
    """
    chunk_shape = []
    for s, dim in zip(slices, full_shape):
        if s.start is not None and s.stop is not None:
            chunk_shape.append(s.stop - s.start)
        elif s.start is None and s.stop is None:
            chunk_shape.append(dim)
        else:
            raise NotImplementedError(f'Open-ended slices are currently unhandled {s}')
    return tuple(chunk_shape)


def _get_block_slices(blocksizes, shape):
    """
    Converts blocksizes into a list of list of slices along each dimension in shape
    An integer blocksizes is treated as a blocksize for the 0th dimension only
    A blocksize of -1 is treated as slice(None) for that dimension
    """
    if isinstance(blocksizes, int):
        # assume chunking only in first dimension
        blocksizes = (blocksizes,) + ((-1,) * len(shape[1:]))
    assert len(blocksizes) == len(shape), ('Must supply a blocksize for every ',
                                           'dimension, (-1 == no chunking), or int')
    return [_blocksize_to_dim_slices(bsize, dim) for bsize, dim in zip(blocksizes, shape)]


def _mk_dask_from_delayed(shape, chunking, dtype='float32', filename=None, value=None):
    """
    Create a dask array by combining individually created blocks

    If filename is not None will load from file using np.memmap
    otherwise will generate numbered partitions using np.ones * chunk_idx
    or partitions of uniform value if value is not None
    """
    if filename is not None:
        create = dask.delayed(_mmap_load_chunk, name='create_chunk', pure=True, traverse=False)
        filename = pathlib.Path(filename)
    else:
        create = dask.delayed(_create_chunk, name='create_chunk', pure=True, traverse=False)

    slices_per_dim = _get_block_slices(chunking, shape)
    blocks = []
    # rightmost advances fastest with itertools.product
    for chunk_idx, chunk_slices in enumerate(itertools.product(*slices_per_dim)):
        chunk_value = chunk_idx if value is None else value
        chunk_shape = _slices_to_chunk_shape(chunk_slices, shape)
        chunk = dask.array.from_delayed(
            create(
                dataset_shape=shape,
                chunk_shape=chunk_shape,
                dtype=dtype,
                value=chunk_value,
                filename=filename,
                sl=chunk_slices
            ),
            shape=chunk_shape,
            dtype=dtype
        )
        blocks.append(chunk)

    nblocks_per_dim = tuple(len(ss) for ss in slices_per_dim)
    blocks = _reshape_list(blocks, nblocks_per_dim)
    return da.block(blocks)
