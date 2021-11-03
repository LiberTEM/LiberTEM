import pytest


from libertem.udf.base import get_resources_for_backends


def test_no_common_backends():
    with pytest.raises(ValueError):
        get_resources_for_backends(
            [('numpy',), ('cupy',)], user_backends=None,
        )


def test_no_common_backends_with_user():
    with pytest.raises(ValueError):
        get_resources_for_backends(
            [('numpy', 'cupy'), ('cupy',)], user_backends=('numpy',),
        )


def test_common_backends_cpu_1():
    resources = get_resources_for_backends(
        [('numpy', 'cupy'), ('numpy',)], user_backends=None,
    )
    assert resources == {
        'CPU': 1, 'compute': 1, 'ndarray': 1
    }


def test_common_backends_cpu_2():
    resources = get_resources_for_backends(
        [('numpy', 'cupy'), ('numpy', 'cupy',)], user_backends=('numpy',),
    )
    assert resources == {
        'CPU': 1, 'compute': 1, 'ndarray': 1
    }


def test_common_backends_compute_1():
    resources = get_resources_for_backends(
        [('numpy', 'cupy'), ('numpy', 'cupy',)], user_backends=None,
    )
    assert resources == {
        'compute': 1, 'ndarray': 1
    }


def test_common_backends_compute_2():
    resources = get_resources_for_backends(
        [('numpy', 'cupy'), ('numpy', 'cupy',)], user_backends=('numpy', 'cupy'),
    )
    assert resources == {
        'compute': 1, 'ndarray': 1
    }


def test_common_backends_gpu_1():
    resources = get_resources_for_backends(
        [('cupy',), ('cuda',)], user_backends=None,
    )
    assert resources == {
        'CUDA': 1, 'compute': 1, 'ndarray': 1
    }


def test_common_backends_gpu_2():
    resources = get_resources_for_backends(
        [('cuda',), ('cupy',)], user_backends=('cupy', 'cuda'),
    )
    assert resources == {
        'CUDA': 1, 'compute': 1, 'ndarray': 1
    }


def test_common_backends_gpu_3():
    resources = get_resources_for_backends(
        [('cupy',), ('cupy',)], user_backends=('cupy',),
    )
    assert resources == {
        'CUDA': 1, 'compute': 1, 'ndarray': 1
    }


def test_common_backends_as_string():
    resources = get_resources_for_backends(
        ['cupy', ('cupy',)], user_backends='cupy',
    )
    assert resources == {
        'CUDA': 1, 'compute': 1, 'ndarray': 1
    }
