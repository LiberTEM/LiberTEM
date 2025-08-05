import os

import numpy as np
import pytest
import h5py

from libertem.api import Context
from libertem.udf.masks import ApplyMasksUDF
from libertem.executor.inline import InlineJobExecutor

from utils import drop_cache, warmup_cache


@pytest.fixture(scope='module')
def chunked_emd(tmpdir_factory):
    lt_ctx = Context(executor=InlineJobExecutor())
    datadir = tmpdir_factory.mktemp('hdf5_chunked_data')
    filename = os.path.join(datadir, 'chunked.emd')

    chunks = (32, 32, 128, 128)

    with h5py.File(filename, mode="w") as f:
        f.attrs.create('version_major', 0)
        f.attrs.create('version_minor', 2)

        f.create_group('experimental/science_data')
        group = f['experimental/science_data']
        group.attrs.create('emd_group_type', 1)

        data = np.ones((256, 256, 128, 128), dtype=np.float32)

        group.create_dataset(name='data', data=data, chunks=chunks)
        group.create_dataset(name='dim1', data=range(256))
        group['dim1'].attrs.create('name', b'dim1')
        group['dim1'].attrs.create('units', b'units1')
        group.create_dataset(name='dim2', data=range(256))
        group['dim2'].attrs.create('name', b'dim2')
        group['dim2'].attrs.create('units', b'units2')
        group.create_dataset(name='dim3', data=range(128))
        group['dim3'].attrs.create('name', b'dim3')
        group['dim3'].attrs.create('units', b'units3')
        group.create_dataset(name='dim4', data=range(128))
        group['dim4'].attrs.create('name', b'dim4')
        group['dim4'].attrs.create('units', b'units4')
        f.close()

    yield lt_ctx.load("auto", path=filename, ds_path="/experimental/science_data/data")


class TestUseSharedExecutor:
    @pytest.mark.benchmark(
        group="io"
    )
    @pytest.mark.parametrize(
        "drop", ("cold_cache", "warm_cache")
    )
    @pytest.mark.parametrize(
        "context", ("dist", "inline")
    )
    def test_mask(self, benchmark, drop, shared_dist_ctx, lt_ctx_fast, context, chunked_emd):
        if context == 'dist':
            ctx = shared_dist_ctx
        elif context == 'inline':
            ctx = lt_ctx_fast
        else:
            raise ValueError

        ds = chunked_emd

        def mask():
            return np.ones(ds.shape.sig, dtype=bool)

        udf = ApplyMasksUDF(mask_factories=[mask], backends=('numpy', ))

        # warmup executor
        ctx.run_udf(udf=udf, dataset=ds)

        if drop == "cold_cache":
            drop_cache([ds.path])
        elif drop == "warm_cache":
            warmup_cache([ds.path])
        else:
            raise ValueError("bad param")

        benchmark.pedantic(
            ctx.run_udf, kwargs=dict(udf=udf, dataset=ds),
            warmup_rounds=0,
            rounds=1,
            iterations=1
        )
