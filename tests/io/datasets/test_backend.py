import platform

import pytest

from libertem.udf.sum import SumUDF
from libertem.io.dataset.base.backend import IOBackend, IOBackendImpl


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


class FakeBackend(IOBackend, id_="fake"):
    def get_impl(self):
        return FakeBackendImpl()


class FakeBackendImpl(IOBackendImpl):
    def get_tiles(
        self, tiling_scheme, fileset, read_ranges, roi, native_dtype, read_dtype, decoder,
        sync_offset, corrections,
    ):
        raise RuntimeError("nothing to see here")
        # to make this a generator, there needs to be a yield statement in
        # the body of the function, even if it is never executed:
        yield  
    
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


@pytest.mark.xfail
def test_auto_uses_correct_backend(lt_ctx, hdf5):
    with pytest.raises(RuntimeError):
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
