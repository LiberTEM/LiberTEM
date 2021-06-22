import os
import sys
from unittest import mock

import pytest
import numpy as np

import libertem.common.backend as bae

from utils import _mk_random, DebugDeviceUDF


@pytest.fixture(autouse=False, scope='module')
def mock_cupy():
    old_cupy = sys.modules.get('cupy')
    sys.modules['cupy'] = np
    yield
    if old_cupy is not None:
        sys.modules['cupy'] = old_cupy
    else:
        sys.modules.pop('cupy')


@pytest.fixture(autouse=False, scope='module')
def mask_cupy():
    old_cupy = sys.modules.get('cupy')
    sys.modules['cupy'] = None
    yield
    if old_cupy is not None:
        sys.modules['cupy'] = old_cupy
    else:
        sys.modules.pop('cupy')


def test_run_default(lt_ctx, mock_cupy):
    data = _mk_random(size=(16, 16, 16, 16))
    ds = lt_ctx.load("memory", data=data)

    res = lt_ctx.run_udf(udf=DebugDeviceUDF(), dataset=ds)
    # Make sure a single string works, common mistype and
    # we can guess what it is supposed to mean
    _ = lt_ctx.run_udf(udf=DebugDeviceUDF(backends='numpy'), dataset=ds)

    for val in res['device_id'].data[0].values():
        # Inline executor uses CPU 0 by default
        assert val['cpu'] == 0
        assert val['cuda'] is None

    # Default to running on CPU
    assert np.all(res['device_class'].data == 'cpu')
    assert np.allclose(res['on_device'].data, data.sum(axis=(0, 1)))


class MockCuda:
    class Device:
        def __init__(self, id):
            pass

        def use(self):
            pass


def test_run_cupy(lt_ctx, mock_cupy):
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
            res = lt_ctx.run_udf(
                udf=DebugDeviceUDF(backends=('cupy', 'numpy')),
                dataset=ds
            )

    for val in res['device_id'].data[0].values():
        assert val['cpu'] is None
        assert val['cuda'] == 23

    # We make sure that the mocking was successful, i.e.
    # restored the previous state
    assert use_cpu == bae.get_use_cpu()
    assert use_cuda == bae.get_use_cuda()
    assert backend == bae.get_device_class()

    assert np.all(res['device_class'].data == 'cuda')
    assert np.allclose(res['on_device'].data, data.sum(axis=(0, 1)))


def test_run_cuda(lt_ctx, mask_cupy):
    # The cupy module is set to None in mask_cupy fixture so that
    # any use of it will raise an error
    with pytest.raises(ModuleNotFoundError):
        import cupy  # NOQA: F401
    data = _mk_random(size=(16, 16, 16, 16))
    ds = lt_ctx.load("memory", data=data)

    use_cpu = bae.get_use_cpu()
    use_cuda = bae.get_use_cuda()
    backend = bae.get_device_class()

    with mock.patch.dict(os.environ, {'LIBERTEM_USE_CUDA': "23"}):
        # This should set the same environment variable as the mock above
        # so that it will be unset after the "with"
        bae.set_use_cuda(23)
        res = lt_ctx.run_udf(
            udf=DebugDeviceUDF(backends=('cuda', 'numpy')),
            dataset=ds
        )

    for val in res['device_id'].data[0].values():
        print(val)
        assert val['cpu'] is None
        assert val['cuda'] == 23

    # We make sure that the mocking was successful, i.e.
    # restored the previous state
    assert use_cpu == bae.get_use_cpu()
    assert use_cuda == bae.get_use_cuda()
    assert backend == bae.get_device_class()

    assert np.all(res['device_class'].data == 'cuda')
    assert np.allclose(res['on_device'].data, data.sum(axis=(0, 1)))


def test_run_numpy(lt_ctx, mask_cupy):
    data = _mk_random(size=(16, 16, 16, 16))
    ds = lt_ctx.load("memory", data=data)

    use_cpu = bae.get_use_cpu()
    use_cuda = bae.get_use_cuda()
    backend = bae.get_device_class()

    with mock.patch.dict(os.environ, {'LIBERTEM_USE_CPU': "42"}):
        # This should set the same environment variable as the mock above
        # so that it will be unset after the "with"
        bae.set_use_cpu(42)
        res = lt_ctx.run_udf(udf=DebugDeviceUDF(), dataset=ds)

    for val in res['device_id'].data[0].values():
        assert val['cpu'] == 42
        assert val['cuda'] is None

    # We make sure that the mocking was successful, i.e.
    # restored the previous state
    assert use_cpu == bae.get_use_cpu()
    assert use_cuda == bae.get_use_cuda()
    assert backend == bae.get_device_class()

    assert np.all(res['device_class'].data == 'cpu')
    assert np.allclose(res['on_device'].data, data.sum(axis=(0, 1)))
