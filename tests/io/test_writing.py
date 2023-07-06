import pathlib
import numpy as np
import pytest

from libertem.udf.sumsigudf import SumSigUDF


@pytest.mark.parametrize(
    "sync_offset", [-2, 0, 2]
)
def test_write_npy(tmpdir_factory, raw_data_8x8x8x8_path, lt_ctx_fast, sync_offset):
    write_dir = tmpdir_factory.mktemp('data')
    write_path = pathlib.Path(write_dir) / 'write_raw_data.npy'
    ds = lt_ctx_fast.load(
        'raw',
        raw_data_8x8x8x8_path,
        np.float32,
        nav_shape=(8, 8),
        sig_shape=(8, 8),
        sync_offset=sync_offset,
    )
    lt_ctx_fast.export_dataset(ds, path=write_path)

    res = lt_ctx_fast.run_udf(udf=SumSigUDF(), dataset=ds)
    sum_ds = res['intensity'].data

    sum_file = np.load(write_path).sum(axis=(2, 3))
    assert np.allclose(sum_ds, sum_file)


def test_bad_extension(tmpdir_factory, raw_data_8x8x8x8_path, lt_ctx_fast):
    write_dir = tmpdir_factory.mktemp('data')
    write_path = pathlib.Path(write_dir) / 'bad.foo'
    ds = lt_ctx_fast.load(
        'raw',
        raw_data_8x8x8x8_path,
        np.float32,
        nav_shape=(8, 8),
        sig_shape=(8, 8),
    )
    with pytest.raises(ValueError):
        lt_ctx_fast.export_dataset(ds, path=write_path)


def test_no_overwrite(tmpdir_factory, raw_data_8x8x8x8_path, lt_ctx_fast):
    write_dir = tmpdir_factory.mktemp('data')
    write_path = pathlib.Path(write_dir) / 'exists.npy'

    some_data = np.ones((5, 5))
    np.save(write_path, some_data)

    ds = lt_ctx_fast.load(
        'raw',
        raw_data_8x8x8x8_path,
        np.float32,
        nav_shape=(8, 8),
        sig_shape=(8, 8),
    )
    with pytest.raises(FileExistsError):
        lt_ctx_fast.export_dataset(ds, path=write_path)
