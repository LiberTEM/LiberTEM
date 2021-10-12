import platform

import pytest

from libertem.udf.sum import SumUDF
from libertem.io.dataset.base import IOBackend

from utils import FakeBackend


def test_backend_selection(lt_ctx, default_raw):
    ds = lt_ctx.load(
        "raw",
        path=default_raw._path,
        dtype="float32",
        nav_shape=(16, 16),
        sig_shape=(128, 128),
    )
    p = next(ds.get_partitions())

    expected_backend = 'MMapBackend'
    if platform.system() == 'Windows':
        expected_backend = 'BufferedBackend'

    assert p.get_io_backend().__class__.__name__ == expected_backend


def test_supported_backends():
    backends = IOBackend.get_supported()
    if platform.system() == "Darwin":
        assert backends == ["mmap", "buffered", "fake"]
    else:
        assert backends == ["mmap", "buffered", "direct", "fake"]


def test_load_uses_correct_backend(lt_ctx, default_raw):
    with pytest.raises(RuntimeError):
        ds = lt_ctx.load(
            "raw",
            path=default_raw._path,
            dtype="float32",
            nav_shape=(16, 16),
            sig_shape=(128, 128),
            io_backend=FakeBackend(),
        )
        lt_ctx.run_udf(
            dataset=ds,
            udf=SumUDF(),
        )


def test_auto_uses_correct_backend_hdf5(lt_ctx, hdf5):
    # H5DataSet currently doesn't support alternative I/O backends
    with pytest.raises(ValueError):
        ds = lt_ctx.load(
            "auto",
            path=hdf5.filename,
            ds_path="data",
            io_backend=FakeBackend(),
        )
        lt_ctx.run_udf(
            dataset=ds,
            udf=SumUDF(),
        )
