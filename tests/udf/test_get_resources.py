import pytest

from libertem.udf.base import UDFRunner, UDF


class UDF1Numpy(UDF):
    def get_result_buffers(self):
        return {}

    def process_frame(self):
        pass

    def get_backends(self):
        return ('numpy',)


class UDF2CuPy(UDF):
    def get_result_buffers(self):
        return {}

    def process_frame(self):
        pass

    def get_backends(self):
        return ('cupy',)


class UDF3NumpyCuPy(UDF):
    def get_result_buffers(self):
        return {}

    def process_frame(self):
        pass

    def get_backends(self):
        return ('cupy', 'numpy')


class UDF4CUDA(UDF):
    def get_result_buffers(self):
        return {}

    def process_frame(self):
        pass

    def get_backends(self):
        return ('cuda',)


class UDF5CUDAString(UDF):
    def get_result_buffers(self):
        return {}

    def process_frame(self):
        pass

    def get_backends(self):
        # Single string instead of tuple
        return 'cuda'


@pytest.mark.parametrize(
    "udfs",
    [(UDF1Numpy(), UDF2CuPy()), (UDF1Numpy(), UDF4CUDA())]
)
def test_no_common_backends(default_raw, udfs):
    runner = UDFRunner(udfs)
    tasks = list(runner._make_udf_tasks(
        dataset=default_raw, roi=None, backends=None,
    ))
    for task in tasks:
        with pytest.raises(ValueError) as e:
            task.get_resources()
        assert e.match("^There is no common supported UDF backend")


@pytest.mark.parametrize(
    "udfs,resources",
    [
        (
            [UDF1Numpy()],
            {'CPU': 1, 'compute': 1, 'ndarray': 1}
        ),
        (
            [UDF2CuPy()],
            {'CUDA': 1, 'compute': 1, 'ndarray': 1}
        ),
        (
            [UDF3NumpyCuPy()],
            {'compute': 1, 'ndarray': 1}
        ),
        (
            [UDF4CUDA()],
            {'CUDA': 1, 'compute': 1}
        ),
        (
            [UDF5CUDAString()],
            {'CUDA': 1, 'compute': 1}
        ),
        (
            [UDF1Numpy(), UDF3NumpyCuPy()],
            {'CPU': 1, 'compute': 1, 'ndarray': 1},
        ),
        (
            [UDF2CuPy(), UDF3NumpyCuPy()],
            {'CUDA': 1, 'compute': 1, 'ndarray': 1},
        ),
        (
            [UDF2CuPy(), UDF4CUDA()],
            {'CUDA': 1, 'compute': 1, 'ndarray': 1},
        ),
        (
            [UDF3NumpyCuPy(), UDF4CUDA()],
            {'CUDA': 1, 'compute': 1, 'ndarray': 1},
        ),
        (
            [UDF3NumpyCuPy(), UDF3NumpyCuPy()],
            {'compute': 1, 'ndarray': 1},
        ),
        (
            [UDF4CUDA(), UDF5CUDAString()],
            {'compute': 1, 'CUDA': 1},
        ),
    ]
)
def test_get_resources_combinations(default_raw, udfs, resources):
    runner = UDFRunner(udfs)
    tasks = list(runner._make_udf_tasks(
        dataset=default_raw, roi=None, backends=None,
    ))
    for task in tasks:
        assert task.get_resources() == resources
