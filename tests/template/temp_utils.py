import h5py
import numpy as np
from libertem.io.dataset.hdf5 import H5DataSet
from libertem.executor.inline import InlineJobExecutor


def _get_hdf5_params(path):
    return {
        "type": "HDF5",
        "params": {
            "path": path,
            "ds_path": "/data"
            },
    }


def create_random_hdf5(path):
    with h5py.File(path, 'w') as f:
        sample_data = np.random.randn(16, 16, 16, 16).astype("float32")
        f.create_dataset("data", (16, 16, 16, 16), data=sample_data)
        # read and provide the ds
    ds = H5DataSet(path=path, ds_path='data')
    ds = ds.initialize(InlineJobExecutor())
    return ds
