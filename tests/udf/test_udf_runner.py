import pytest
import numpy as np

from libertem.udf.base import UDFRunner, UDF
from libertem.io.dataset.memory import MemoryDataSet
from libertem.io.dataset.base import TilingScheme
from libertem.api import Context

from utils import _mk_random


class PixelsumUDF(UDF):
    def get_result_buffers(self):
        return {
            'pixelsum': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_frame(self, frame):
        assert frame.shape == (16, 16)
        assert self.results.pixelsum.shape == (1,)
        self.results.pixelsum[:] = np.sum(frame)


@pytest.mark.asyncio
async def test_async_run_for_dset(async_executor):
    data = _mk_random(size=(16 * 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)

    pixelsum = PixelsumUDF()
    roi = np.zeros((256,), dtype=bool)
    runner = UDFRunner([pixelsum])

    udf_iter = runner.run_for_dataset_async(
        dataset, async_executor, roi=roi, cancel_id="42"
    )

    async for udf_results in udf_iter:
        udf_results.buffers
        pass
    assert "udf_results" in locals(), "must yield at least one result"


class UDF1(UDF):
    def get_result_buffers(self):
        return {}

    def process_frame(self):
        pass

    def get_backends(self):
        return ('numpy',)


class UDF2(UDF):
    def get_result_buffers(self):
        return {}

    def process_frame(self):
        pass

    def get_backends(self):
        return ('cupy',)


class UDF3(UDF):
    def get_result_buffers(self):
        return {}

    def process_frame(self):
        pass

    def get_backends(self):
        return ('cupy', 'numpy')


class UDF4(UDF):
    def get_result_buffers(self):
        return {}

    def process_frame(self):
        pass

    def get_backends(self):
        return ('cuda',)


class UDF5(UDF):
    def get_result_buffers(self):
        return {}

    def process_frame(self):
        pass

    def get_backends(self):
        # Single string instead of tuple
        return 'cuda'


def test_no_common_backends(default_raw, lt_ctx: Context):
    runner = UDFRunner([UDF1(), UDF2()])
    tiling_scheme = TilingScheme.make_for_shape(
        dataset_shape=default_raw.shape,
        tileshape=default_raw.shape.flatten_nav(),
    )
    tasks = list(runner._make_udf_tasks(
        dataset=default_raw, roi=None, backends=None, executor=lt_ctx.executor,
        tiling_scheme=tiling_scheme,
    ))
    for task in tasks:
        with pytest.raises(ValueError) as e:
            task.get_resources()
        assert e.match("^There is no common supported UDF backend")


def test_no_common_backends_2(default_raw, lt_ctx):
    runner = UDFRunner([UDF1(), UDF4()])
    tiling_scheme = TilingScheme.make_for_shape(
        dataset_shape=default_raw.shape,
        tileshape=default_raw.shape.flatten_nav(),
    )
    tasks = list(runner._make_udf_tasks(
        dataset=default_raw, roi=None, backends=None, executor=lt_ctx.executor,
        tiling_scheme=tiling_scheme,
    ))
    for task in tasks:
        with pytest.raises(ValueError) as e:
            task.get_resources()
        assert e.match("^There is no common supported UDF backend")


def test_common_backends_cpu(default_raw, lt_ctx):
    runner = UDFRunner([UDF1(), UDF3()])
    tiling_scheme = TilingScheme.make_for_shape(
        dataset_shape=default_raw.shape,
        tileshape=default_raw.shape.flatten_nav(),
    )
    tasks = list(runner._make_udf_tasks(
        dataset=default_raw, roi=None, backends=None, executor=lt_ctx.executor,
        tiling_scheme=tiling_scheme,
    ))
    for task in tasks:
        assert task.get_resources() == {'CPU': 1, 'compute': 1, 'ndarray': 1}


def test_common_backends_gpu(default_raw, lt_ctx):
    runner = UDFRunner([UDF2(), UDF3()])
    tiling_scheme = TilingScheme.make_for_shape(
        dataset_shape=default_raw.shape,
        tileshape=default_raw.shape.flatten_nav(),
    )
    tasks = list(runner._make_udf_tasks(
        dataset=default_raw, roi=None, backends=None, executor=lt_ctx.executor,
        tiling_scheme=tiling_scheme,
    ))
    for task in tasks:
        assert task.get_resources() == {'CUDA': 1, 'compute': 1, 'ndarray': 1}


def test_common_backends_gpu_2(default_raw, lt_ctx):
    runner = UDFRunner([UDF2(), UDF4()])
    tiling_scheme = TilingScheme.make_for_shape(
        dataset_shape=default_raw.shape,
        tileshape=default_raw.shape.flatten_nav(),
    )
    tasks = list(runner._make_udf_tasks(
        dataset=default_raw, roi=None, backends=None, executor=lt_ctx.executor,
        tiling_scheme=tiling_scheme,
    ))
    for task in tasks:
        assert task.get_resources() == {'CUDA': 1, 'compute': 1, 'ndarray': 1}


def test_common_backends_gpu_3(default_raw, lt_ctx):
    tiling_scheme = TilingScheme.make_for_shape(
        dataset_shape=default_raw.shape,
        tileshape=default_raw.shape.flatten_nav(),
    )
    runner = UDFRunner([UDF3(), UDF4()])
    tasks = list(runner._make_udf_tasks(
        dataset=default_raw, roi=None, backends=None, executor=lt_ctx.executor,
        tiling_scheme=tiling_scheme,
    ))
    for task in tasks:
        assert task.get_resources() == {'CUDA': 1, 'compute': 1, 'ndarray': 1}


def test_common_backends_compute(default_raw, lt_ctx):
    runner = UDFRunner([UDF3(), UDF3()])
    tiling_scheme = TilingScheme.make_for_shape(
        dataset_shape=default_raw.shape,
        tileshape=default_raw.shape.flatten_nav(),
    )
    tasks = list(runner._make_udf_tasks(
        dataset=default_raw, roi=None, backends=None, executor=lt_ctx.executor,
        tiling_scheme=tiling_scheme,
    ))
    for task in tasks:
        assert task.get_resources() == {'compute': 1, 'ndarray': 1}


def test_common_backends_string(default_raw, lt_ctx):
    runner = UDFRunner([UDF4(), UDF5()])
    tiling_scheme = TilingScheme.make_for_shape(
        dataset_shape=default_raw.shape,
        tileshape=default_raw.shape.flatten_nav(),
    )
    tasks = list(runner._make_udf_tasks(
        dataset=default_raw, roi=None, backends=None, executor=lt_ctx.executor,
        tiling_scheme=tiling_scheme,
    ))
    for task in tasks:
        assert task.get_resources() == {'CUDA': 1, 'compute': 1}
