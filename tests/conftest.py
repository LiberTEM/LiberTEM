import os
import tempfile

import pytest
import h5py
import numpy as np

from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.hdf5 import H5DataSet
from libertem import api as lt


@pytest.fixture
def inline_executor():
    return InlineJobExecutor()


@pytest.fixture
def lt_ctx(inline_executor):
    return lt.Context(executor=inline_executor)


@pytest.fixture
def hdf5():
    f, tmpfn = tempfile.mkstemp(suffix=".h5")
    os.close(f)
    with h5py.File(tmpfn, "w") as f:
        yield f
    os.unlink(tmpfn)


@pytest.fixture
def hdf5_ds_1(hdf5):
    hdf5.create_dataset("data", data=np.ones((5, 5, 16, 16)))
    return H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 5, 16, 16), target_size=512*1024*1024
    )
