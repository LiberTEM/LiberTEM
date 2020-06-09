import os
import sys
from unittest import mock

import pytest
import numpy as np

from libertem.udf.base import UDF
import libertem.common.backend as bae

from utils import _mk_random


@pytest.fixture(autouse=True, scope='module')
def mock_cupy():
    old_cupy = sys.modules.get('cupy')
    sys.modules['cupy'] = np
    yield
    if old_cupy is not None:
        sys.modules['cupy'] = old_cupy


class DebugUDF(UDF):
    def get_result_buffers(self):
        return {
            'debug': self.buffer(kind="single", dtype="object"),
            'on_device': self.buffer(kind="sig", dtype=np.float32, where="device"),
            'backend': self.buffer(kind="nav", dtype="object"),
        }

    def preprocess(self):
        self.results.debug[0] = dict()

    def process_partition(self, partition):
        cpu = bae.get_use_cpu()
        cuda = bae.get_use_cuda()
        self.results.debug[0][self.meta.slice] = (cpu, cuda)
        self.results.on_device[:] += self.xp.sum(partition, axis=0)
        self.results.backend[:] = self.meta.device_class
        print(f"meta device_class {self.meta.device_class}")

    def merge(self, dest, src):
        de, sr = dest['debug'][0], src['debug'][0]
        for key, value in sr.items():
            assert key not in de
            de[key] = value

        dest['on_device'][:] += src['on_device']
        dest['backend'][:] = src['backend']

    def get_backends(self):
        return ('cupy', 'numpy')


def test_run_default(lt_ctx):
    # Inline executor sets CPU 0 by default
    data = _mk_random(size=(16, 16, 16, 16))
    ds = lt_ctx.load("memory", data=data)

    res = lt_ctx.run_udf(udf=DebugUDF(), dataset=ds)

    for val in res['debug'].data[0].values():
        assert val == (0, None)

    assert np.all(res['backend'].data == 'cpu')

    assert np.allclose(res['on_device'].data, data.sum(axis=(0, 1)))


class MockCuda:
    class Device:
        def __init__(self, id):
            pass

        def use(self):
            pass


def test_run_cupy(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16))
    ds = lt_ctx.load("memory", data=data)

    use_cpu = bae.get_use_cpu()
    use_cuda = bae.get_use_cuda()
    backend = bae.get_device_class()

    with mock.patch.dict(os.environ, {'LIBERTEM_USE_CUDA': "23"}):
        # This should set the same environment variable as the mock above
        # so that it will be unset after the "with"
        bae.set_use_cuda(23)
        # add `numpy.cuda` so we can make `numpy` work as a mock replacement for `cupy` 
        with mock.patch('numpy.cuda', return_value=MockCuda, create=True):
            res = lt_ctx.run_udf(udf=DebugUDF(), dataset=ds)

    for val in res['debug'].data[0].values():
        assert val == (None, 23)

    # We make sure that the mocking was successful, i.e.
    # restored the previous state
    assert use_cpu == bae.get_use_cpu()
    assert use_cuda == bae.get_use_cuda()
    assert backend == bae.get_device_class()

    assert np.all(res['backend'].data == 'cuda')

    assert np.allclose(res['on_device'].data, data.sum(axis=(0, 1)))


def test_run_numpy(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16))
    ds = lt_ctx.load("memory", data=data)

    use_cpu = bae.get_use_cpu()
    use_cuda = bae.get_use_cuda()
    backend = bae.get_device_class()

    with mock.patch.dict(os.environ, {'LIBERTEM_USE_CPU': "42"}):
        # This should set the same environment variable as the mock above
        # so that it will be unset after the "with"
        bae.set_use_cpu(42)
        res = lt_ctx.run_udf(udf=DebugUDF(), dataset=ds)

    for val in res['debug'].data[0].values():
        assert val == (42, None)

    # We make sure that the mocking was successful, i.e.
    # restored the previous state
    assert use_cpu == bae.get_use_cpu()
    assert use_cuda == bae.get_use_cuda()
    assert backend == bae.get_device_class()

    assert np.all(res['backend'].data == 'cpu')

    assert np.allclose(res['on_device'].data, data.sum(axis=(0, 1)))
