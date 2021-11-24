import os
from glob import glob

import pytest
import distributed
import numpy as np

from libertem.executor.delayed import DelayedJobExecutor
from libertem.executor.dask import DaskJobExecutor
from libertem.executor.concurrent import ConcurrentJobExecutor
from libertem.executor.inline import InlineJobExecutor
from libertem.api import Context
from libertem.udf.stddev import StdDevUDF
from libertem.udf.masks import ApplyMasksUDF

from utils import get_testdata_path


@pytest.fixture(
    params=[
        "inline_executor_fast", "dask_executor", "delayed_default",
        "delayed_dist", "threaded_dask_executor", "concurrent"
    ]
)
def executor(request, inline_executor_fast, dask_executor):
    if request.param == 'inline_executor_fast':
        yield inline_executor_fast
    elif request.param == "dask_executor":
        yield dask_executor
    elif request.param == "delayed_default":
        yield DelayedJobExecutor()
    elif request.param == "delayed_dist":
        with distributed.Client(
                n_workers=2,
                threads_per_worker=4,
                processes=True
        ) as _:
            yield DelayedJobExecutor()
    elif request.param == "threaded_dask_executor":
        with distributed.Client(
                n_workers=2,
                threads_per_worker=4,
                processes=False
        ) as c:
            yield DaskJobExecutor(client=c)
    elif request.param == "concurrent":
        yield ConcurrentJobExecutor.make_local()


@pytest.fixture
def ctx(executor):
    return Context(executor=executor)


@pytest.fixture(scope='session')
def load_kwargs(hdf5, default_raw):
    kwargs = [
        {
            'filetype': 'HDF5',
            'path': hdf5.filename
        },
        {
            'filetype': 'RAW',
            'path': default_raw._path,
            'nav_shape': default_raw.shape.nav,
            'sig_shape': default_raw.shape.sig,
            'dtype': default_raw.dtype
        },
        {
            'filetype': 'memory',
            'data': np.ones((3, 4, 5, 6))
        }
    ]
    testdata_path = get_testdata_path()
    blo_path = os.path.join(testdata_path, 'default.blo')
    if os.path.isfile(blo_path):
        kwargs.append({
            'filetype': 'BLO',
            'path': blo_path,
        })
    dm_files = list(sorted(glob(os.path.join(testdata_path, 'dm', '*.dm4'))))
    if dm_files:
        kwargs.append({
            'filetype': 'dm',
            'files': dm_files
        })
    empad_path = os.path.join(testdata_path, 'EMPAD', 'acquisition_12_pretty.xml')
    if os.path.isfile(empad_path):
        kwargs.append({
            'filetype': 'EMPAD',
            'path': empad_path
        })
    frms6_path = os.path.join(testdata_path, 'frms6', 'C16_15_24_151203_019.hdr')
    if os.path.isfile(frms6_path):
        kwargs.append({
            'filetype': 'frms6',
            'path': frms6_path
        })
    k2is_path = os.path.join(testdata_path, 'Capture52', 'Capture52_.gtg')
    if os.path.isfile(k2is_path):
        kwargs.append({
            'filetype': 'k2is',
            'path': k2is_path
        })
    mib_path = os.path.join(testdata_path, 'default.mib')
    if os.path.isfile(mib_path):
        kwargs.append({
            'filetype': 'mib',
            'path': mib_path,
            'nav_shape': (32, 32)
        })
    mrc_path = os.path.join(testdata_path, 'mrc', '20200821_92978_movie.mrc')
    if os.path.isfile(mrc_path):
        kwargs.append({
            'filetype': 'mrc',
            'path': mrc_path
        })
    seq_path = os.path.join(testdata_path, 'default.seq')
    if os.path.isfile(seq_path):
        kwargs.append({
            'filetype': 'seq',
            'path': seq_path,
            'nav_shape': (8, 8)
        })
    ser_path = os.path.join(testdata_path, 'default.ser')
    if os.path.isfile(ser_path):
        kwargs.append({
            'filetype': 'ser',
            'path': ser_path
        })

    return kwargs


def _make_udfs(ds):
    def factory():
        m = np.zeros(ds.shape.sig)
        m[-1, -1] = 1.3
        return m

    udfs = [
        StdDevUDF(), ApplyMasksUDF(mask_factories=[factory])
    ]
    return udfs


def _calculate(ctx, load_kwargs):
    result = {}
    print(f"calculating with {ctx.executor}")
    for kwargs in load_kwargs:
        ds = ctx.load(**kwargs)
        udfs = _make_udfs(ds)
        roi = None
        roi = np.zeros(
            np.prod(ds.shape.nav, dtype=np.int64),
            dtype=bool
        )
        roi[0] = True
        roi[-1] = True
        roi[len(roi)//2] = True
        roi = roi.reshape(ds.shape.nav)
        print(f"calculating {kwargs['filetype']}")
        result[kwargs['filetype']] = ctx.run_udf(
            dataset=ds, udf=udfs, roi=roi
        )
    return result


@pytest.fixture(scope='session')
def reference(load_kwargs):
    ctx = Context(executor=InlineJobExecutor())
    return _calculate(ctx, load_kwargs)


@pytest.mark.slow
def test_executors(ctx, load_kwargs, reference):
    results = _calculate(ctx, load_kwargs)
    for key, res in results.items():
        print(f"filetype: {key}")
        if isinstance(ctx.executor, DelayedJobExecutor):
            res = res.compute()

        assert len(res) == len(reference[key])
        for i, item in enumerate(reference[key]):
            assert item.keys() == res[i].keys()
            for buf_key in item.keys():
                print(f"buffer {buf_key}")
                assert np.allclose(
                    item[buf_key].raw_data,
                    res[i][buf_key].raw_data
                )
