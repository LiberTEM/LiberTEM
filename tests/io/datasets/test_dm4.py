import os
import uuid

import numpy as np
import pytest

from libertem.common.shape import Shape
from libertem.io.dataset.base.tiling_scheme import TilingScheme
from libertem.udf.base import UDF

from utils import ValidationUDF, _mk_random


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
                key = f'ImageList.{ds_idx}.Null'
                self.allTags[key] = 0
                shape_unpack = zip(shape, f_shape_order)
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


@pytest.fixture(scope='session')
def dm4_mockfile_f(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    shape = Shape((4, 8, 16, 24), 2)
    dtype = np.float32
    return _make_mock_array(datadir, shape, dtype, False)


@pytest.fixture(scope='session')
def dm4_mockfile_c(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    shape = Shape((4, 8, 16, 24), 2)
    dtype = np.float32
    return _make_mock_array(datadir, shape, dtype, True)


def _patch_filedm(monkeypatch, dm_obj):
    monkeypatch.setattr('ncempy.io.dm.fileDM', dm_obj)


@pytest.mark.parametrize(
    "dm4_mockfile", [("dm4_mockfile_c"), ("dm4_mockfile_f")]
)
def test_comparison_f(monkeypatch, dm4_mockfile, lt_ctx_fast, request):
    (array, filename), mock_fileDM = request.getfixturevalue(dm4_mockfile)
    _patch_filedm(monkeypatch, mock_fileDM)

    ds = lt_ctx_fast.load('dm4', filename)
    assert ds.meta.shape.to_tuple() == array.shape
    flat_data = array.reshape(-1, *array.shape[-2:])
    udf = ValidationUDF(reference=flat_data)
    lt_ctx_fast.run_udf(udf=udf, dataset=ds)


@pytest.mark.parametrize(
    "dm4_mockfile", [("dm4_mockfile_c"), ("dm4_mockfile_f")]
)
def test_comparison_roi(monkeypatch, dm4_mockfile, lt_ctx_fast, request):
    (array, filename), mock_fileDM = request.getfixturevalue(dm4_mockfile)
    _patch_filedm(monkeypatch, mock_fileDM)

    ds = lt_ctx_fast.load('dm4', filename)
    assert ds.meta.shape.to_tuple() == array.shape

    roi = np.random.choice(
        [True, False],
        size=tuple(ds.shape.nav),
        p=[0.5, 0.5]
    )

    flat_data = array.reshape(-1, *array.shape[-2:])
    udf = ValidationUDF(reference=flat_data[roi.ravel()])
    lt_ctx_fast.run_udf(udf=udf, dataset=ds, roi=roi)


@pytest.mark.parametrize(
    "dm4_mockfile", [("dm4_mockfile_c"), ("dm4_mockfile_f")]
)
def test_many_tiles(monkeypatch, dm4_mockfile, lt_ctx_fast, request):
    (array, filename), mock_fileDM = request.getfixturevalue(dm4_mockfile)
    _patch_filedm(monkeypatch, mock_fileDM)

    ds = lt_ctx_fast.load('dm4', filename)
    ds.set_num_cores(8)
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
            assert np.allclose(tile_slice.get(flat_data), tile)
            ds_visited[tile_slice.get()] += 1
    assert (ds_visited == 1).all()


class SumFrameUDF(UDF):
    def get_result_buffers(self):
        return {'sum': self.buffer('nav')}

    def process_frame(self, frame):
        self.results.sum[:] += frame.sum()


@pytest.mark.parametrize(
    "dm4_mockfile", [("dm4_mockfile_c"), ("dm4_mockfile_f")]
)
def test_process_frame(monkeypatch, dm4_mockfile, lt_ctx_fast, request):
    (array, filename), mock_fileDM = request.getfixturevalue(dm4_mockfile)
    _patch_filedm(monkeypatch, mock_fileDM)

    ds = lt_ctx_fast.load('dm4', filename)
    res = lt_ctx_fast.run_udf(dataset=ds, udf=SumFrameUDF())
    assert np.allclose(res['sum'].data, array.sum(axis=(2, 3)))
