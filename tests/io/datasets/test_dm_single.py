import os
import uuid

import numpy as np
import pytest

from libertem.common.shape import Shape
from libertem.common.math import prod
from libertem.io.dataset.base.exceptions import DataSetException
from libertem.io.dataset.base.tiling_scheme import TilingScheme
from libertem.udf.base import UDF
from libertem.udf.sumsigudf import SumSigUDF
from libertem.io.dataset.dm_single import SingleDMDataSet
from libertem.contrib.convert_transposed import convert_dm4_transposed

try:
    import hyperspy.api as hs
except ModuleNotFoundError:
    hs = None

from utils import ValidationUDF, _mk_random, dataset_correction_verification, get_testdata_path


DM_TESTDATA_PATH = os.path.join(get_testdata_path(), 'dm')
HAVE_DM_TESTDATA = os.path.exists(DM_TESTDATA_PATH)


@pytest.fixture(scope='module')
def dm4_c_order_4d_path():
    return os.path.join(DM_TESTDATA_PATH, '4d_data_c_order_zen4307783.dm4')


@pytest.fixture(scope='module')
def dm4_f_order_3d_path():
    return os.path.join(DM_TESTDATA_PATH, '3d_data_f_order_zen2580185.dm4')


@pytest.fixture(scope='module')
def dm3_3dstack_path():
    return os.path.join(DM_TESTDATA_PATH, '3D', 'alpha-50_obj.dm3')


@pytest.fixture(scope='module')
def dm4_2dimage_path():
    return os.path.join(DM_TESTDATA_PATH, '2018-7-17 15_29_0000.dm4')


@pytest.fixture(scope='module')
def dm4_c_order_4d_raw(dm4_c_order_4d_path):
    return hs.load(dm4_c_order_4d_path).data


@pytest.fixture(scope='module')
def dm4_f_order_3d_raw(dm4_f_order_3d_path):
    return hs.load(dm4_f_order_3d_path).data


@pytest.fixture(scope='module')
def dm3_3dstack_raw(dm3_3dstack_path):
    return hs.load(dm3_3dstack_path).data


@pytest.fixture(scope='module')
def dm4_2dimage_raw(dm4_2dimage_path):
    return hs.load(dm4_2dimage_path).data


@pytest.mark.skipif(not HAVE_DM_TESTDATA, reason="No DM4 test data")
def test_3d_f_order(lt_ctx_fast, dm4_f_order_3d_path):
    with pytest.raises(DataSetException):
        lt_ctx_fast.load('dm', dm4_f_order_3d_path)


@pytest.mark.skipif(not HAVE_DM_TESTDATA, reason="No DM4 test data")
@pytest.mark.skipif(hs is None, reason="No HyperSpy found")
@pytest.mark.slow
def test_3d_f_order_force(lt_ctx_fast, dm4_f_order_3d_path, dm4_f_order_3d_raw):
    ds = lt_ctx_fast.load('dm', dm4_f_order_3d_path, force_c_order=True)
    assert ds.shape == Shape((3710, 146, 228), sig_dims=1)
    res = lt_ctx_fast.run_udf(ds, SumSigUDF())
    # best we can do!
    assert np.allclose(res['intensity'].data.sum(), dm4_f_order_3d_raw.sum())


@pytest.mark.skipif(not HAVE_DM_TESTDATA, reason="No DM4 test data")
@pytest.mark.skipif(hs is None, reason="No HyperSpy found")
@pytest.mark.slow
def test_3d_c_order(lt_ctx_fast, dm3_3dstack_path, dm3_3dstack_raw):
    ds = lt_ctx_fast.load('dm', dm3_3dstack_path)
    assert ds.shape == Shape((20, 3838, 3710), sig_dims=2)
    res = lt_ctx_fast.run_udf(ds, SumSigUDF())
    assert np.allclose(res['intensity'].data, dm3_3dstack_raw.sum(axis=(-2, -1)))


@pytest.mark.skipif(not HAVE_DM_TESTDATA, reason="No DM4 test data")
@pytest.mark.skipif(hs is None, reason="No HyperSpy found")
@pytest.mark.slow
def test_2d_c_order(lt_ctx_fast, dm4_2dimage_path, dm4_2dimage_raw):
    ds = lt_ctx_fast.load('dm', dm4_2dimage_path)
    assert ds.shape == Shape((1, 3838, 3710), sig_dims=2)
    res = lt_ctx_fast.run_udf(ds, SumSigUDF())
    assert np.allclose(res['intensity'].data.squeeze(), dm4_2dimage_raw.sum(axis=(-2, -1)))


@pytest.mark.skipif(not HAVE_DM_TESTDATA, reason="No DM4 test data")
@pytest.mark.skipif(hs is None, reason="No HyperSpy found")
@pytest.mark.slow
def test_4d_c_order(lt_ctx_fast, dm4_c_order_4d_path, dm4_c_order_4d_raw):
    ds = lt_ctx_fast.load('dm', dm4_c_order_4d_path)
    assert isinstance(ds, SingleDMDataSet)
    assert ds.shape.to_tuple() == (213, 342, 128, 128)
    assert ds.dtype == np.uint8
    res = lt_ctx_fast.run_udf(ds, SumSigUDF())
    assert np.allclose(res['intensity'].data, dm4_c_order_4d_raw.sum(axis=(-2, -1)))


@pytest.mark.skipif(not HAVE_DM_TESTDATA, reason="No DM4 test data")
@pytest.mark.skipif(hs is None, reason="No HyperSpy found")
@pytest.mark.slow
def test_comparison_raw(lt_ctx_fast, dm4_c_order_4d_path, dm4_c_order_4d_raw):
    ds = lt_ctx_fast.load('dm', dm4_c_order_4d_path)
    assert ds.meta.shape.to_tuple() == dm4_c_order_4d_raw.shape
    flat_data = dm4_c_order_4d_raw.reshape(-1, *dm4_c_order_4d_raw.shape[-2:])
    udf = ValidationUDF(reference=flat_data)
    lt_ctx_fast.run_udf(udf=udf, dataset=ds)


# Remaining tests are on mockfile objects rather than the large, real files


class MockFileDM:
    def __init__(self, shapes, offsets, dtypes, orders):
        self.dataShape = []
        self.xSize = []
        self.ySize = []
        self.zSize = []
        self.zSize2 = []
        self.dataOffset = []
        self.dataType = []
        self.allTags = {}
        self.thumbnail = False
        self.numObjects = len(shapes)

        c_shape_order = (self.zSize2,
                         self.zSize,
                         self.ySize,
                         self.xSize)
        f_shape_order = (self.ySize,
                         self.xSize,
                         self.zSize2,
                         self.zSize)

        for ds_idx, (shape, offset, dtype, order) in enumerate(zip(shapes,
                                                                   offsets,
                                                                   dtypes,
                                                                   orders),
                                                               start=1):
            self.dataShape.append(len(shape))
            self.dataOffset.append(offset)
            self.dataType.append(dtype)
            if order == 'C':
                key = f'ImageList.{ds_idx}.ImageTags.Meta Data.Data Order Swapped'
                self.allTags[key] = 1
                shape_unpack = zip(shape, c_shape_order)
            else:
                shape_unpack = zip(shape, f_shape_order)
            key = f'ImageList.{ds_idx}.ImageTags.Meta Data.Format'
            self.allTags[key] = 'Diffraction image'
            for s, dim in shape_unpack:
                dim.append(s)

    def __call__(self, path, on_memory=True):
        return self

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        ...

    def _DM2NPDataType(self, dtype):
        if dtype is None:
            raise OSError('no dtype')
        return dtype


def generate_dm4_mockfile(filename, shape, dtype,
                          c_order=False, header=1000):
    array = _mk_random(shape.to_tuple(), dtype=dtype)
    sig_dims = shape.sig.dims

    order = 'C' if c_order else 'F'

    with open(filename, 'ab') as fp:
        fp.write(b'0' * header)
        offset = fp.tell()
        if c_order:
            array.tofile(fp)
        else:
            dims = tuple(range(shape.dims))
            f_axis_order = dims[-sig_dims:] + dims[:-sig_dims]
            np.transpose(array, f_axis_order).tofile(fp)

    return array, order, offset


def _make_mock_array(datadir, shape, dtype, c_order):
    suffix = str(uuid.uuid1())[:4]
    filename = os.path.join(datadir, f'dm4_mock_f_{suffix}.dm4')
    array, order, offset = generate_dm4_mockfile(filename,
                                                 shape,
                                                 dtype,
                                                 c_order=c_order)
    mock_fileDM = MockFileDM((shape,), (offset,), (dtype,), (order,))
    return (array, filename), mock_fileDM


@pytest.fixture(scope='module')
def dm4_mockfile_f(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    shape = Shape((4, 8, 16, 24), 2)
    dtype = np.float32
    return _make_mock_array(datadir, shape, dtype, False)


@pytest.fixture(scope='module')
def dm4_mockfile_c(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    shape = Shape((4, 8, 16, 24), 2)
    dtype = np.float32
    return _make_mock_array(datadir, shape, dtype, True)


def _patch_filedm(monkeypatch, dm_obj):
    monkeypatch.setattr('ncempy.io.dm.fileDM', dm_obj)


def test_f_order_raises(monkeypatch, dm4_mockfile_f, lt_ctx_fast):
    (array, filename), mock_fileDM = dm4_mockfile_f
    _patch_filedm(monkeypatch, mock_fileDM)

    with pytest.raises(DataSetException):
        lt_ctx_fast.load('dm', filename)


def test_comparison(monkeypatch, dm4_mockfile_c, lt_ctx_fast):
    (array, filename), mock_fileDM = dm4_mockfile_c
    _patch_filedm(monkeypatch, mock_fileDM)

    ds = lt_ctx_fast.load('dm', filename)
    assert ds.meta.shape.to_tuple() == array.shape
    flat_data = array.reshape(-1, *array.shape[-2:])
    udf = ValidationUDF(reference=flat_data)
    lt_ctx_fast.run_udf(udf=udf, dataset=ds)


def test_comparison_roi(monkeypatch, dm4_mockfile_c, lt_ctx_fast):
    (array, filename), mock_fileDM = dm4_mockfile_c
    _patch_filedm(monkeypatch, mock_fileDM)

    ds = lt_ctx_fast.load('dm', filename)
    assert ds.meta.shape.to_tuple() == array.shape

    roi = np.random.choice(
        [True, False],
        size=tuple(ds.shape.nav),
        p=[0.5, 0.5]
    )

    flat_data = array.reshape(-1, *array.shape[-2:])
    udf = ValidationUDF(reference=flat_data[roi.ravel()])
    lt_ctx_fast.run_udf(udf=udf, dataset=ds, roi=roi)


def test_many_tiles(monkeypatch, dm4_mockfile_c, lt_ctx_fast):
    (array, filename), mock_fileDM = dm4_mockfile_c
    _patch_filedm(monkeypatch, mock_fileDM)

    ds = lt_ctx_fast.load('dm', filename, num_partitions=8)
    _, _, sy, sx = array.shape
    flat_data = array.reshape(-1, sy, sx)
    # depth 5, height 3, divides neither flat_nav or sy evenly
    tileshape = Shape((5, 3, sx), sig_dims=2)
    tiling_scheme = TilingScheme.make_for_shape(tileshape, ds.meta.shape, intent='tile')
    parts = [*ds.get_partitions()]
    assert len(parts) > 1
    ds_visited = np.zeros(flat_data.shape, dtype=int)
    for part in parts:
        for tile in part.get_tiles(tiling_scheme):
            tile_slice = tile.tile_slice
            assert np.allclose(tile_slice.get(flat_data), tile.data)
            ds_visited[tile_slice.get()] += 1
    assert (ds_visited == 1).all()


class SumFrameUDF(UDF):
    def get_result_buffers(self):
        return {'sum': self.buffer('nav')}

    def process_frame(self, frame):
        self.results.sum[:] += frame.sum()


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
def test_process_frame(monkeypatch, dm4_mockfile_c, lt_ctx_fast, with_roi):
    (array, filename), mock_fileDM = dm4_mockfile_c
    _patch_filedm(monkeypatch, mock_fileDM)

    ds = lt_ctx_fast.load('dm', filename)
    result = array.sum(axis=(2, 3))

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
        result[np.logical_not(roi)] = np.nan
    else:
        roi = None

    res = lt_ctx_fast.run_udf(dataset=ds, udf=SumFrameUDF(), roi=roi)
    if roi is not None:
        assert (np.isnan(res['sum'].data) == np.logical_not(roi)).all()
    assert np.allclose(res['sum'].data, result, equal_nan=True)


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
def test_corrections_default(monkeypatch, dm4_mockfile_c, lt_ctx_fast, with_roi):
    (array, filename), mock_fileDM = dm4_mockfile_c
    _patch_filedm(monkeypatch, mock_fileDM)

    ds = lt_ctx_fast.load('dm', filename)

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx_fast)


def test_macrotile_normal(monkeypatch, dm4_mockfile_c, lt_ctx_fast):
    (array, filename), mock_fileDM = dm4_mockfile_c
    _patch_filedm(monkeypatch, mock_fileDM)

    ds = lt_ctx_fast.load('dm', filename, num_partitions=4)

    ps = ds.get_partitions()
    _ = next(ps)
    p2 = next(ps)
    macrotile = p2.get_macrotile()
    assert macrotile.tile_slice.shape == p2.shape
    assert macrotile.tile_slice.origin[0] == p2._start_frame


def test_macrotile_roi(monkeypatch, dm4_mockfile_c, lt_ctx_fast):
    (array, filename), mock_fileDM = dm4_mockfile_c
    _patch_filedm(monkeypatch, mock_fileDM)

    ds = lt_ctx_fast.load('dm', filename)

    roi = np.zeros(ds.shape.nav, dtype=bool)
    roi[0, 5] = 1
    roi[1, 1] = 1
    p = next(ds.get_partitions())
    macrotile = p.get_macrotile(roi=roi)
    assert tuple(macrotile.tile_slice.shape) == (2, 16, 24)


def test_positive_sync_offset(monkeypatch, dm4_mockfile_c, lt_ctx):
    (array, filename), mock_fileDM = dm4_mockfile_c
    _patch_filedm(monkeypatch, mock_fileDM)

    udf = SumSigUDF()
    sync_offset = 2

    ds_no_offset = lt_ctx.load('dm', filename)
    ds_with_offset = SingleDMDataSet(
        path=filename,
        sync_offset=sync_offset,
        num_partitions=4,
    )
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == 2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (4,) + tuple(ds_with_offset.shape.sig),
        sig_dims=ds_with_offset.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds_with_offset.shape,
    )

    # t0 = next(p0.get_tiles(tiling_scheme))
    # Not guaranteed as FortranReader can emit tiles out-of-order
    # assert tuple(t0.tile_slice.origin) == (0, 0, 0)

    for p in ds_with_offset.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p.slice.origin == (24, 0, 0)
    assert p.slice.shape[0] == 8

    result = lt_ctx.run_udf(dataset=ds_no_offset, udf=udf)
    result = result['intensity'].raw_data[sync_offset:]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[
        :ds_with_offset._meta.image_count - sync_offset
    ]

    assert np.allclose(result, result_with_offset)


def test_negative_sync_offset(monkeypatch, dm4_mockfile_c, lt_ctx):
    (array, filename), mock_fileDM = dm4_mockfile_c
    _patch_filedm(monkeypatch, mock_fileDM)

    udf = SumSigUDF()
    sync_offset = -2

    ds_no_offset = lt_ctx.load('dm', filename)
    ds_with_offset = SingleDMDataSet(
        path=filename,
        sync_offset=sync_offset,
        num_partitions=4,
    )
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == -2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (4,) + tuple(ds_with_offset.shape.sig),
        sig_dims=ds_with_offset.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds_with_offset.shape,
    )

    # t0 = next(p0.get_tiles(tiling_scheme))
    # Not guaranteed as FortranReader can emit tiles out-of-order
    # assert tuple(t0.tile_slice.origin) == (0, 0, 0)

    for p in ds_with_offset.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p.slice.origin == (24, 0, 0)
    assert p.slice.shape[0] == 8

    result = lt_ctx.run_udf(dataset=ds_no_offset, udf=udf)
    result = result['intensity'].raw_data[:ds_no_offset._meta.image_count - abs(sync_offset)]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[abs(sync_offset):]

    assert np.allclose(result, result_with_offset)


def test_nav_reshaping_c_error(monkeypatch, dm4_mockfile_c, lt_ctx_fast):
    (array, filename), mock_fileDM = dm4_mockfile_c
    _patch_filedm(monkeypatch, mock_fileDM)

    nav_shape = tuple(d + 1 for d in array.shape[:2])
    with pytest.raises(DataSetException):
        lt_ctx_fast.load('dm', filename, nav_shape=nav_shape)


def test_nav_reshaping_c(monkeypatch, dm4_mockfile_c, lt_ctx_fast):
    (array, filename), mock_fileDM = dm4_mockfile_c
    _patch_filedm(monkeypatch, mock_fileDM)

    # read only part of the nav space
    manual_nav_shape = tuple(d - 1 for d in array.shape[:2])
    ds = lt_ctx_fast.load('dm', filename, nav_shape=manual_nav_shape)
    res = lt_ctx_fast.run_udf(dataset=ds, udf=SumSigUDF())
    result_array = res['intensity'].data

    _, _, sy, sx = array.shape
    nframes = prod(manual_nav_shape)
    partial_sum = array.reshape(-1, sy, sx).sum(axis=(1, 2))[:nframes]
    assert np.allclose(result_array, partial_sum.reshape(manual_nav_shape))


def test_sig_reshaping_c_error(monkeypatch, dm4_mockfile_c, lt_ctx_fast):
    (array, filename), mock_fileDM = dm4_mockfile_c
    _patch_filedm(monkeypatch, mock_fileDM)

    sig_shape = tuple(d + 1 for d in array.shape[2:])
    with pytest.raises(DataSetException):
        lt_ctx_fast.load('dm', filename, sig_shape=sig_shape)


def test_sig_reshaping_c(monkeypatch, dm4_mockfile_c, lt_ctx_fast):
    (array, filename), mock_fileDM = dm4_mockfile_c
    _patch_filedm(monkeypatch, mock_fileDM)

    # read only part of each sig space
    manual_sig_shape = tuple(d - 1 for d in array.shape[:2])
    ds = lt_ctx_fast.load('dm', filename, sig_shape=manual_sig_shape)
    res = lt_ctx_fast.run_udf(dataset=ds, udf=SumSigUDF())
    result_array = res['intensity'].data

    ny, nx = array.shape[:2]
    flat_array = array.ravel()
    nel = ny * nx * prod(manual_sig_shape)
    partial_array = flat_array[:nel].reshape(array.shape[:2] + manual_sig_shape)
    partial_sum = partial_array.sum(axis=(2, 3))
    assert np.allclose(result_array, partial_sum)


@pytest.mark.skipif(not HAVE_DM_TESTDATA, reason="No DM4 test data")
def test_convert_not_transposed_raises(lt_ctx, dm4_c_order_4d_path):
    with pytest.raises(DataSetException):
        convert_dm4_transposed(
            dm4_c_order_4d_path,
            'out.npy',
            ctx=lt_ctx,
        )


def test_convert_f_ordered(monkeypatch, dm4_mockfile_f, lt_ctx, tmpdir_factory):
    tdir = tmpdir_factory.mktemp('convert_transposed')

    # the mockfile array in memory is already c-ordered
    (array, filename), mock_fileDM = dm4_mockfile_f
    _patch_filedm(monkeypatch, mock_fileDM)

    converted_path = os.path.join(tdir, 'out_dm4_f.npy')
    convert_dm4_transposed(
        filename,
        converted_path,
        ctx=lt_ctx,
    )

    ds_npy = lt_ctx.load('npy', path=converted_path)
    pick_a = lt_ctx.create_pick_analysis(ds_npy, 4, 3)
    pick_frame = lt_ctx.run(pick_a).intensity.raw_data
    # flipped coords because pick_a takes x, y arguments
    assert np.allclose(pick_frame, array[3, 4, :, :])


@pytest.mark.skipif(not HAVE_DM_TESTDATA, reason="No DM4 test data")
def test_num_partitions(lt_ctx, dm3_3dstack_path):
    ds = lt_ctx.load(
        "dm",
        path=dm3_3dstack_path,
        num_partitions=2,
    )
    assert len(list(ds.get_partitions())) == 2
