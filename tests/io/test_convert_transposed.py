import pathlib
import numpy as np

from libertem.udf.raw import PickUDF
from libertem.udf.sum import SumUDF
from libertem.contrib.convert_transposed import _convert_transposed_ds

from utils import _mk_random


def test_functional(lt_ctx, tmpdir_factory):
    dir = pathlib.Path(tmpdir_factory.mktemp('convert_transposed'))
    data = _mk_random(size=(7, 5, 3, 4), dtype='float32')
    converted_data = np.moveaxis(data, (0, 1, 2, 3), (2, 3, 0, 1))
    ds = lt_ctx.load('memory', data=data, num_partitions=2)

    convert_path = dir / 'out.npy'
    _convert_transposed_ds(lt_ctx, ds, convert_path)

    ds_npy = lt_ctx.load('npy', path=convert_path)
    assert ds_npy.shape.to_tuple() == converted_data.shape

    for check_slice in (
        np.s_[1, 3],
        np.s_[2, 0],
        np.s_[0, 1],
    ):
        roi = np.zeros(ds_npy.shape.nav, dtype=bool)
        roi[check_slice] = True
        frame = lt_ctx.run_udf(ds_npy, PickUDF(), roi=roi)['intensity'].data.squeeze()
        assert np.allclose(frame, converted_data[check_slice])

    sum_res = lt_ctx.run_udf(ds_npy, SumUDF())['intensity'].data
    assert np.allclose(sum_res, converted_data.sum(axis=(0, 1)))
