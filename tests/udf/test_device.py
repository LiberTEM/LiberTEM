import sys
import os
from unittest import mock

import numpy as np

from libertem.udf.base import UDF

from utils import _mk_random

# Mock CuPy with NumPy
# FIXME this overrides it for all following tests,
# is there a better approach?
sys.modules['cupy'] = np


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
        cpu = os.environ.get("LIBERTEM_USE_CPU", None)
        cuda = os.environ.get("LIBERTEM_USE_CUDA", None)
        self.results.debug[0][self.meta.slice] = (cpu, cuda)
        self.results.on_device[:] += self.xp.sum(partition, axis=0)
        self.results.backend[:] = self.meta.backend
        print(f"meta backend {self.meta.backend}")

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
        assert val == ("0", None)

    assert np.all(res['backend'].data == 'numpy')

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

    with mock.patch.dict(os.environ, {'LIBERTEM_USE_CUDA': "23"}):
        with mock.patch('numpy.cuda', return_value=MockCuda, create=True):
            res = lt_ctx.run_udf(udf=DebugUDF(), dataset=ds)

    for val in res['debug'].data[0].values():
        assert val == (None, "23")

    assert np.all(res['backend'].data == 'cupy')

    assert np.allclose(res['on_device'].data, data.sum(axis=(0, 1)))


@mock.patch.dict(os.environ, {'LIBERTEM_USE_CPU': "42"})
def test_run_numpy(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16))
    ds = lt_ctx.load("memory", data=data)

    res = lt_ctx.run_udf(udf=DebugUDF(), dataset=ds)

    for val in res['debug'].data[0].values():
        assert val == ("42", None)

    assert np.all(res['backend'].data == 'numpy')

    assert np.allclose(res['on_device'].data, data.sum(axis=(0, 1)))
