import h5py
import io
import numpy as np
import pathlib
import matplotlib.pyplot as plt

import libertem.api as lt
from libertem.udf.sumsigudf import SumSigUDF


if __name__ == '__main__':
    data_root = pathlib.Path('./data')

    true_shape = (12, 15)
    nav_prod = np.prod(true_shape)
    sig_shape = (6, 9)
    dtype = np.float32
    
    # row = np.arange(true_shape[1], dtype=dtype)[:, np.newaxis, np.newaxis]
    # row = np.repeat(row, sig_shape[-1], axis=-1)
    # row = np.repeat(row, sig_shape[-2], axis=-2)
    # data = np.repeat(row[np.newaxis, ...], true_shape[0], axis=0)
    data = np.random.uniform(low=0., high=10., size=true_shape + sig_shape).astype(np.float32)
    flat_data = data.reshape(2, 6, 15, *sig_shape)
    chunks = (2, 3, 5) + sig_shape
    filepath = data_root / 'data.hdf5'
    ds_path = 'data'
    roi = np.random.choice([True, False], size=true_shape).astype(bool)
    
    sum_result = data.sum(axis=(-2, -1))
    if roi is not None:
        sum_result[np.logical_not(roi)] = np.nan

    with h5py.File(filepath,'w') as fp:
        dset = fp.create_dataset(ds_path, data=flat_data, chunks=chunks)

    ctx = lt.Context.make_with('inline')
    udf = SumSigUDF()
    ds = ctx.load(
            'hdf5',
            path=filepath,
            ds_path=ds_path,
            nav_shape=true_shape,
            target_size=np.dtype(dtype).itemsize * true_shape[1] * 2 * np.prod(sig_shape),
        )
    res = ctx.run_udf(ds, udf=udf, roi=roi)
    result = res['intensity'].data.copy()

    assert np.allclose(result, sum_result, equal_nan=roi is not None)
