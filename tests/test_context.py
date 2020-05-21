import pytest

from libertem.executor.inline import InlineJobExecutor
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
