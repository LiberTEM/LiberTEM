import os
import uuid

import numpy as np
import pytest

from libertem.common.shape import Shape

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


def _test_validation(ctx, array, filename):
    ds = ctx.load('dm4', filename)
    assert ds.meta.shape.to_tuple() == array.shape
    flat_data = array.reshape(-1, *array.shape[-2:])
    udf = ValidationUDF(reference=flat_data)
    ctx.run_udf(udf=udf, dataset=ds)


def test_comparison_f(monkeypatch, dm4_mockfile_f, lt_ctx_fast):
    (array, filename), mock_fileDM = dm4_mockfile_f
    monkeypatch.setattr('ncempy.io.dm.fileDM', mock_fileDM)
    _test_validation(lt_ctx_fast, array, filename)


def test_comparison_c(monkeypatch, dm4_mockfile_c, lt_ctx_fast):
    (array, filename), mock_fileDM = dm4_mockfile_c
    monkeypatch.setattr('ncempy.io.dm.fileDM', mock_fileDM)
    _test_validation(lt_ctx_fast, array, filename)


def _test_roi(ctx, array, filename):
    ds = ctx.load('dm4', filename)
    assert ds.meta.shape.to_tuple() == array.shape

    roi = np.random.choice(
        [True, False],
        size=tuple(ds.shape.nav),
        p=[0.5, 0.5]
    )

    flat_data = array.reshape(-1, *array.shape[-2:])
    udf = ValidationUDF(reference=flat_data[roi.ravel()])
    ctx.run_udf(udf=udf, dataset=ds, roi=roi)


def test_comparison_f_roi(monkeypatch, dm4_mockfile_f, lt_ctx_fast):
    (array, filename), mock_fileDM = dm4_mockfile_f
    monkeypatch.setattr('ncempy.io.dm.fileDM', mock_fileDM)
    _test_roi(lt_ctx_fast, array, filename)


def test_comparison_c_roi(monkeypatch, dm4_mockfile_c, lt_ctx_fast):
    (array, filename), mock_fileDM = dm4_mockfile_c
    monkeypatch.setattr('ncempy.io.dm.fileDM', mock_fileDM)
    _test_roi(lt_ctx_fast, array, filename)
