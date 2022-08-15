import pathlib
import importlib
import numpy as np
import pytest
import dask.array as da

from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.sum import SumUDF
from libertem.io.dataset.base import DataSetException
from libertem.io.dataset.dask import DaskDataSet

from utils import dataset_correction_verification
# Load the dask dataset utils from this same folder
# This is really ugly but necessary to ensure that
# dask-worker processes can find utils_dask.py
location = pathlib.Path(__file__).parent / "utils_dask.py"
spec = importlib.util.spec_from_file_location("utils_dask", location)
utils_dask = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_dask)
_mk_dask_from_delayed = utils_dask._mk_dask_from_delayed


@pytest.mark.parametrize(
    "dest_dtype", ('float32', 'int32')
)
def test_get_macrotile(lt_ctx, dest_dtype):
    data = _mk_dask_from_delayed(shape=(16, 16, 16, 16), chunking=(1, -1, -1, -1))
    ds = lt_ctx.load('dask', data, sig_dims=2)
    p = next(ds.get_partitions())
    mt = p.get_macrotile(dest_dtype=dest_dtype)
    assert tuple(mt.shape) == (256, 16, 16)
    assert mt.dtype == np.dtype(dest_dtype)


def test_merge_sig(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(1, -1, 8, 8))
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=0.)
    assert tuple(ds.array.chunks) == ((1,) * data.shape[0], (25,), (16,), (16,))


def test_contig_nav(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(2, 5, -1, -1))
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=0.)
    assert tuple(ds.array.chunks) == ((2, 2, 1), (25,), (16,), (16,))


def test_reorient_nav(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(5, 1, -1, -1))
    sig_dims = 2
    ds = lt_ctx.load('dask', data, sig_dims=sig_dims, preserve_dimensions=False, min_size=0.)
    assert tuple(ds.array.chunks) == ((1,) * data.shape[1], (5,), (16,), (16,))
    assert ds.array.shape == (25, 5, 16, 16)
    assert tuple(reversed(sorted(len(c) for c in ds.array.chunks[:-sig_dims])))\
           == tuple(len(c) for c in ds.array.chunks[:-sig_dims])


def test_reorient_nav2(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(1, -1, -1, -1))
    sig_dims = 2
    ds = lt_ctx.load('dask', data, sig_dims=sig_dims, preserve_dimensions=False, min_size=0.)
    assert tuple(ds.array.chunks) == ((1,) * data.shape[0], (25,), (16,), (16,))
    assert ds.array.shape == data.shape


def test_reorient_nav3(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(2, 1, -1, -1))
    sig_dims = 2
    ds = lt_ctx.load('dask', data, sig_dims=sig_dims, preserve_dimensions=True, min_size=0.)
    assert ds.array.shape == data.shape


@pytest.mark.dist
def test_size_based_merging(dist_ctx):
    dtype = np.float32
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(1, -1, -1, -1), dtype=dtype)
    chunksize = np.prod(data.shape[1:]) * np.dtype(dtype).itemsize
    min_size = chunksize * 2 - 1
    ds = dist_ctx.load('dask', data, sig_dims=2, min_size=min_size)
    assert tuple(ds.array.chunks) == ((3, 2), (25,), (16,), (16,))


def test_size_based_merging2(lt_ctx):
    dtype = np.float32
    data = _mk_dask_from_delayed(shape=(12, 25, 16, 16), chunking=(1, 2, -1, -1), dtype=dtype)
    frame_size = np.prod(data.shape[2:]) * np.dtype(dtype).itemsize
    min_size = frame_size * 4
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=min_size)
    assert tuple(ds.array.chunks) == ((1,) * data.shape[0], (6, 4, 4, 4, 7), (16,), (16,))


def test_size_based_merging3(lt_ctx):
    dtype = np.float32
    data = _mk_dask_from_delayed(shape=(12, 25, 16, 16), chunking=(1, 2, -1, -1), dtype=dtype)
    min_size = 0  # a zero minimum size should have no size-based merging
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=min_size)
    assert tuple(ds.array.chunks) == data.chunks


def test_size_based_merging4(lt_ctx):
    dtype = np.float32
    data = _mk_dask_from_delayed(shape=(6, 10, 16, 16), chunking=(1, -1, -1, -1), dtype=dtype)
    min_size = np.inf  # should result in one large partition
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=min_size)
    assert tuple(ds.array.chunks) == tuple((el,) for el in data.shape)


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
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=0.)
    assert ds.get_num_partitions() == 3


def test_3d_array(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 16, 16), chunking=(1, 8, 8))
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=0.)
    assert tuple(ds.array.chunks) == ((1,) * data.shape[0], (16,), (16,))


def test_4d_1sig_array(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 36, 50, 16), chunking=(1, 6, 10, 8))
    ds = lt_ctx.load('dask', data, sig_dims=1, min_size=0.)
    assert tuple(ds.array.chunks) == ((1,) * 5, (6,) * 6, (50,), (16,))


def test_4d_3sig_array(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 36, 50, 16), chunking=(1, 6, 10, 8))
    ds = lt_ctx.load('dask', data, sig_dims=3, min_size=0.)
    assert tuple(ds.array.chunks) == ((1,) * 5, (36,), (50,), (16,))


def test_5d_array(lt_ctx):
    data = _mk_dask_from_delayed(shape=(8, 4, 5, 16, 16), chunking=(4, 2, 1, 8, 8))
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=0.)
    assert tuple(ds.array.chunks) == ((4, 4), (4,), (5,), (16,), (16,))


def test_5d_size_based_merging(lt_ctx):
    dtype = np.float32
    data = _mk_dask_from_delayed(shape=(8, 12, 25, 16, 16), chunking=(1, 1, 2, -1, -1), dtype=dtype)
    frame_size = np.prod(data.shape[2:]) * np.dtype(dtype).itemsize
    min_size = frame_size * 4
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=min_size)
    assert tuple(ds.array.chunks) == ((1,) * data.shape[0], (4, 4, 4), (25,), (16,), (16,))


def test_6d_array(lt_ctx):
    data = _mk_dask_from_delayed(shape=(4, 8, 4, 5, 16, 16), chunking=(2, 4, 2, 1, 8, 8))
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=0.)
    assert tuple(ds.array.chunks) == ((2, 2), (8,), (4,), (5,), (16,), (16,))


def test_sum_udf(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(1, -1, 8, 8))
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=0.)
    res = lt_ctx.run_udf(ds, udf=SumUDF())
    assert np.allclose(res['intensity'].data, data.sum(axis=(0, 1)).compute())


def test_sumsig_udf(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(2, -1, 8, 8))
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=0.)
    res = lt_ctx.run_udf(ds, udf=SumSigUDF())
    assert np.allclose(res['intensity'].data, data.sum(axis=(2, 3)).compute())


def test_part_file_mapping(lt_ctx):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16),
                                 chunking=(1, -1, -1, -1))
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=0.)

    for part_idx, part in enumerate(ds.get_partitions()):
        macrotile = part.get_macrotile(dest_dtype="float32")
        assert np.unique(np.asarray(macrotile.data)) == np.asarray([part_idx]).astype(np.float32)


def test_part_file_mapping2(lt_ctx):
    data = _mk_dask_from_delayed(shape=(4, 24, 16, 16),
                                 chunking=(2, 12, -1, -1))
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=0.)

    sequence = np.asarray([0, 1]).astype(np.float32)
    for part in ds.get_partitions():
        macrotile = part.get_macrotile(dest_dtype="float32")
        assert np.allclose(np.unique(np.asarray(macrotile.data)), sequence)
        sequence += sequence.size


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
def test_correction(lt_ctx, with_roi):
    data = _mk_dask_from_delayed(shape=(5, 25, 16, 16), chunking=(2, -1, -1, -1))
    ds = lt_ctx.load('dask', data, sig_dims=2, min_size=0.)

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


@pytest.mark.dist
def test_dist_process(dist_ctx):
    shape = (3, 5, 7, 11)
    data = da.random.random(shape, chunks=(2, 2, 2, 2))
    roi = np.random.choice([True, False], data.shape[:2])
    # Dask doesn't do multi-dimensional fancy indexing with booleans,
    # unlike NumPy :rolleyes:
    ref = data.reshape((np.prod(shape[:2]), *shape[2:]))[roi.flatten()].sum(axis=0).compute()
    ds = dist_ctx.load("dask", data, sig_dims=2)
    res = dist_ctx.run_udf(dataset=ds, udf=SumUDF(), roi=roi)
    assert np.allclose(res['intensity'].raw_data, ref)
