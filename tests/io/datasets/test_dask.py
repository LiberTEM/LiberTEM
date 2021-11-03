import numpy as np
import pytest
import dask
import dask.array as da
import itertools

from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.sum import SumUDF
from libertem.io.dataset.base import DataSetException
from libertem.io.dataset.dask import DaskDataSet

from utils import dataset_correction_verification


def _create_chunk(shape, dtype, value=1):
    return np.ones(shape, dtype=dtype) * value


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


def _mk_dask_from_delayed(shape, chunking, dtype='float32', indexed_values=False):
    """
    Create a dask array by combining individually created blocks
    """
    create = dask.delayed(_create_chunk, name='create_chunk', pure=True, traverse=False)

    slices_per_dim = _get_block_slices(chunking, shape)
    blocks = []
    # rightmost advances fastest with itertools.product
    for chunk_idx, chunk_slices in enumerate(itertools.product(*slices_per_dim)):
        chunk_value = chunk_idx if indexed_values else 1
        chunk_shape = _slices_to_chunk_shape(chunk_slices, shape)
        chunk = dask.array.from_delayed(
            create(
                shape=chunk_shape,
                dtype=dtype,
                value=chunk_value
            ),
            shape=chunk_shape,
            dtype=dtype
        )
        blocks.append(chunk)

    nblocks_per_dim = tuple(len(ss) for ss in slices_per_dim)
    blocks = _reshape_list(blocks, nblocks_per_dim)
    return da.block(blocks)


@pytest.mark.parametrize(
    "dest_dtype", ('float32', 'int32')
)
def test_get_macrotile(lt_ctx, dest_dtype):
    data = _mk_dask_from_delayed(shape=(16, 16, 16, 16), chunking=(1, -1, -1, -1))
    ds = lt_ctx.load('dask', data, sig_dims=2, preserve_dimensions=True)
    p = next(ds.get_partitions())
    mt = p.get_macrotile(dest_dtype=dest_dtype)
    assert tuple(mt.shape) == (256, 16, 16)
    assert mt.dtype == np.dtype(dest_dtype)


def test_merge_sig(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(1, -1, 8, 8))
    ds = lt_ctx.load('dask', data, sig_dims=2, preserve_dimensions=True, min_size=0.)
    assert tuple(ds.array.chunks) == ((1,) * data.shape[0], (25,), (16,), (16,))


def test_contig_nav(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(2, 5, -1, -1))
    ds = lt_ctx.load('dask', data, sig_dims=2, preserve_dimensions=True, min_size=0.)
    assert tuple(ds.array.chunks) == ((2, 2, 1), (25,), (16,), (16,))


def test_reorient_nav(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(5, 1, -1, -1))
    sig_dims = 2
    ds = lt_ctx.load('dask', data, sig_dims=sig_dims, min_size=0.)
    assert tuple(ds.array.chunks) == ((1,) * data.shape[1], (5,), (16,), (16,))
    assert ds.array.shape == (25, 5, 16, 16)
    assert tuple(reversed(sorted(len(c) for c in ds.array.chunks[:-sig_dims])))\
           == tuple(len(c) for c in ds.array.chunks[:-sig_dims])


def test_size_based_merging(lt_ctx):
    dtype = np.float32
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(1, -1, -1, -1), dtype=dtype)
    chunksize = np.prod(data.shape[1:]) * np.dtype(dtype).itemsize
    min_size = chunksize * 2 - 1
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=min_size)
    assert tuple(ds.array.chunks) == ((3, 2), (25,), (16,), (16,))


def test_size_based_merging2(lt_ctx):
    dtype = np.float32
    data = _mk_dask_from_delayed(shape=(12, 25, 16, 16), chunking=(1, 2, -1, -1), dtype=dtype)
    frame_size = np.prod(data.shape[2:]) * np.dtype(dtype).itemsize
    min_size = frame_size * 4
    ds = lt_ctx.load('dask', data, sig_dims=2, preserve_dimensions=True, min_size=min_size)
    assert tuple(ds.array.chunks) == ((1,) * data.shape[0], (6, 4, 4, 4, 7), (16,), (16,))


def test_no_chunking(lt_ctx):
    dtype = np.float32
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(-1, -1, -1, -1), dtype=dtype)
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=0.)
    assert tuple(ds.array.chunks) == ((5,), (25,), (16,), (16,))


def test_check_if_dask(lt_ctx):
    with pytest.raises(DataSetException):
        lt_ctx.load('dask', False, sig_dims=2)


def test_chunking_defined(lt_ctx):
    data = da.random.random(size=(5, 25, 16, 16), chunks=((-1,) * 4))
    data += 0.1
    mask = data.sum(axis=(1, 2, 3)) > 0.
    data = data[mask, ...]

    with pytest.raises(DataSetException):
        lt_ctx.load('dask', data, sig_dims=2)


def test_check_sig_dims(lt_ctx):
    data = da.random.random(size=(5, 25, 16, 16), chunks=((-1,) * 4))

    with pytest.raises(DataSetException):
        lt_ctx.load('dask', data, sig_dims=3.)


def test_check_sig_dims2(lt_ctx):
    data = da.random.random(size=(5, 25, 16, 16), chunks=((-1,) * 4))

    with pytest.raises(DataSetException):
        lt_ctx.load('dask', data, sig_dims=7)


def test_io_backend():
    data = da.random.random(size=(5, 25, 16, 16), chunks=((-1,) * 4))
    with pytest.raises(DataSetException):
        DaskDataSet(data, sig_dims=2, io_backend=True)


def test_get_num_part(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(2, 5, -1, -1))
    ds = lt_ctx.load('dask', data, sig_dims=2, preserve_dimensions=True, min_size=0.)
    assert ds.get_num_partitions() == 3


def test_3d_array(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 16, 16), chunking=(1, 8, 8))
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=0.)
    assert tuple(ds.array.chunks) == ((1,) * data.shape[0], (16,), (16,))


def test_4d_1sig_array(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 36, 50, 16), chunking=(1, 6, 10, 8))
    ds = lt_ctx.load('dask', data, sig_dims=1, preserve_dimensions=True, min_size=0.)
    assert tuple(ds.array.chunks) == ((1,) * 5, (6,) * 6, (50,), (16,))


def test_4d_3sig_array(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 36, 50, 16), chunking=(1, 6, 10, 8))
    ds = lt_ctx.load('dask', data, sig_dims=3, preserve_dimensions=True, min_size=0.)
    assert tuple(ds.array.chunks) == ((1,) * 5, (36,), (50,), (16,))


def test_5d_array(lt_ctx):
    data = _mk_dask_from_delayed(shape=(8, 4, 5, 16, 16), chunking=(4, 2, 1, 8, 8))
    ds = lt_ctx.load('dask', data, sig_dims=2, preserve_dimensions=True, min_size=0.)
    assert tuple(ds.array.chunks) == ((4, 4), (4,), (5,), (16,), (16,))


def test_5d_size_based_merging(lt_ctx):
    dtype = np.float32
    data = _mk_dask_from_delayed(shape=(8, 12, 25, 16, 16), chunking=(1, 1, 2, -1, -1), dtype=dtype)
    frame_size = np.prod(data.shape[2:]) * np.dtype(dtype).itemsize
    min_size = frame_size * 4
    ds = lt_ctx.load('dask', data, sig_dims=2, preserve_dimensions=True, min_size=min_size)
    assert tuple(ds.array.chunks) == ((1,) * data.shape[0], (4, 4, 4), (25,), (16,), (16,))


def test_6d_array(lt_ctx):
    data = _mk_dask_from_delayed(shape=(4, 8, 4, 5, 16, 16), chunking=(2, 4, 2, 1, 8, 8))
    ds = lt_ctx.load('dask', data, sig_dims=2, preserve_dimensions=True, min_size=0.)
    assert tuple(ds.array.chunks) == ((2, 2), (8,), (4,), (5,), (16,), (16,))


def test_sum_udf(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(1, -1, 8, 8))
    ds = lt_ctx.load('dask', data, sig_dims=2, preserve_dimensions=True, min_size=0.)
    res = lt_ctx.run_udf(ds, udf=SumUDF())
    assert np.allclose(res['intensity'].data, np.prod(data.shape[:2]) * np.ones(data.shape[2:]))


def test_sumsig_udf(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(2, -1, 8, 8))
    ds = lt_ctx.load('dask', data, sig_dims=2, preserve_dimensions=True, min_size=0.)
    res = lt_ctx.run_udf(ds, udf=SumSigUDF())
    assert np.allclose(res['intensity'].data, np.prod(data.shape[2:]) * np.ones(data.shape[:2]))


def test_part_file_mapping(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16),
                                 chunking=(1, -1, -1, -1),
                                 indexed_values=True)
    ds = lt_ctx.load('dask', data, sig_dims=2, preserve_dimensions=True, min_size=0.)

    for part_idx, part in enumerate(ds.get_partitions()):
        macrotile = part.get_macrotile(dest_dtype="float32")
        assert np.unique(np.asarray(macrotile)) == np.asarray([part_idx]).astype(np.float32)


def test_part_file_mapping2(lt_ctx):
    data = _mk_dask_from_delayed(shape=(4, 24, 16, 16),
                                 chunking=(2, 12, -1, -1),
                                 indexed_values=True)
    ds = lt_ctx.load('dask', data, sig_dims=2, preserve_dimensions=True, min_size=0.)

    sequence = np.asarray([0, 1]).astype(np.float32)
    for part in ds.get_partitions():
        macrotile = part.get_macrotile(dest_dtype="float32")
        assert np.allclose(np.unique(np.asarray(macrotile)), sequence)
        sequence += sequence.size


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
def test_correction(lt_ctx, with_roi):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(2, -1, -1, -1))
    ds = lt_ctx.load('dask', data, sig_dims=2, preserve_dimensions=True, min_size=0.)

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)
