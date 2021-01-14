import pytest
import numpy as np

from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.base import MMapBackend
from libertem.udf.sum import SumUDF
from libertem.api import Context


def test_ctx_load(lt_ctx, default_raw):
    lt_ctx.load(
        "raw",
        path=default_raw._path,
        nav_shape=(16, 16),
        dtype="float32",
        sig_shape=(128, 128),
    )


def test_context_arguments():
    with pytest.raises(ValueError):
        # refs https://github.com/LiberTEM/LiberTEM/issues/918
        Context(executor=InlineJobExecutor)


def test_run_udf_with_io_backend(lt_ctx, default_raw):
    io_backend = MMapBackend(enable_readahead_hints=True)
    lt_ctx.load(
        "raw",
        path=default_raw._path,
        io_backend=io_backend,
        scan_size=(16, 16),
        dtype="float32",
        detector_size=(128, 128),
    )
    res = lt_ctx.run_udf(dataset=default_raw, udf=SumUDF())
    assert np.array(res['intensity']).shape == (128, 128)
