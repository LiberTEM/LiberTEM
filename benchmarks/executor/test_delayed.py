import numpy as np
import pytest

from libertem.udf.base import UDF
from libertem.api import Context
from libertem.executor.delayed import DelayedJobExecutor
from libertem.contrib.daskadapter import make_dask_array


@pytest.fixture(scope="module", params=["float32", "uint16"])
def my_ds(request, medium_raw, medium_raw_float32):
    if request.param == 'float32':
        return medium_raw_float32
    elif request.param == 'uint16':
        return medium_raw
    else:
        raise ValueError()


class MySumUDF(UDF):
    def get_result_buffers(self):
        return {
            'intensity': self.buffer(kind='sig', dtype=self.meta.input_dtype)
        }

    def process_tile(self, tile):
        self.results.intensity[:] += np.sum(tile, axis=0)

    def merge(self, dest, src):
        dest.intensity[:] += src.intensity


class MySumMergeUDF(MySumUDF):
    def merge_all(self, ordered_results):
        intensity = np.stack([b.intensity for b in ordered_results.values()]).sum(axis=0)
        return {
            'intensity': intensity
        }


class MySumSigUDF(UDF):
    def get_result_buffers(self):
        return {
            'intensity': self.buffer(
                kind="nav", dtype=self.meta.input_dtype
            ),
        }

    # Make sure we don't have default merge so that there is no default merge_all
    def merge(self, dest, src):
        return super().merge(dest, src)

    def process_tile(self, tile):
        self.results.intensity[:] += np.sum(tile, axis=tuple(range(1, len(tile.shape))))


class MySumSigMergeUDF(MySumSigUDF):
    def merge_all(self, ordered_results):
        intensity = np.concatenate([b.intensity for b in ordered_results.values()])
        return {
            'intensity': intensity
        }


class EchoUDF(UDF):
    def get_result_buffers(self):
        return {
            'intensity': self.buffer(
                kind='nav', dtype=self.meta.input_dtype, extra_shape=self.meta.dataset_shape.sig
            )
        }

    def process_tile(self, tile):
        sl = (..., *self.meta.sig_slice.get())
        self.results.intensity[sl] = tile

    # Make sure we don't have default merge so that there is no default merge_all
    def merge(self, dest, src):
        return super().merge(dest, src)


class EchoMergeUDF(EchoUDF):
    def merge_all(self, ordered_results):
        intensity = np.concatenate([b.intensity for b in ordered_results.values()])
        return {
            'intensity': intensity
        }


class Test:
    @pytest.mark.benchmark(
        group="sum",
    )
    def test_sum_baseline(self, shared_dist_ctx_globaldask, my_ds, benchmark):
        ctx = shared_dist_ctx_globaldask
        udf = MySumUDF()
        benchmark(
            ctx.run_udf,
            dataset=my_ds,
            udf=udf
        )

    @pytest.mark.benchmark(
        group="sum",
    )
    def test_sum_delayed(self, shared_dist_ctx_globaldask, my_ds, benchmark):
        ctx = Context(executor=DelayedJobExecutor())
        udf = MySumUDF()
        resources = DelayedJobExecutor.get_resources_from_udfs(udf)

        def doit():
            result = ctx.run_udf(dataset=my_ds, udf=udf)
            # Make sure we run on the same number of workers
            return result['intensity'].raw_data.compute(resources=resources)

        benchmark(doit)

    @pytest.mark.benchmark(
        group="sum",
    )
    def test_sum_delayed_merge(self, shared_dist_ctx_globaldask, my_ds, benchmark):
        ctx = Context(executor=DelayedJobExecutor())
        udf = MySumMergeUDF()
        resources = DelayedJobExecutor.get_resources_from_udfs(udf)

        def doit():
            result = ctx.run_udf(dataset=my_ds, udf=udf)
            return result['intensity'].raw_data.compute(resources=resources)

        benchmark(doit)

    @pytest.mark.benchmark(
        group="sumsig",
    )
    def test_sumsig_baseline(self, shared_dist_ctx_globaldask, my_ds, benchmark):
        ctx = shared_dist_ctx_globaldask
        udf = MySumSigUDF()
        benchmark(
            ctx.run_udf,
            dataset=my_ds,
            udf=udf
        )

    @pytest.mark.benchmark(
        group="sumsig",
    )
    def test_sumsig_delayed(self, shared_dist_ctx_globaldask, my_ds, benchmark):
        ctx = Context(executor=DelayedJobExecutor())
        udf = MySumSigUDF()
        resources = DelayedJobExecutor.get_resources_from_udfs(udf)

        def doit():
            result = ctx.run_udf(dataset=my_ds, udf=udf)
            return result['intensity'].raw_data.compute(resources=resources)

        benchmark(doit)

    @pytest.mark.benchmark(
        group="sumsig",
    )
    def test_sumsig_delayed_merge(self, shared_dist_ctx_globaldask, my_ds, benchmark):
        ctx = Context(executor=DelayedJobExecutor())
        udf = MySumSigMergeUDF()
        resources = DelayedJobExecutor.get_resources_from_udfs(udf)

        def doit():
            result = ctx.run_udf(dataset=my_ds, udf=udf)
            return result['intensity'].raw_data.compute(resources=resources)

        benchmark(doit)

    @pytest.mark.benchmark(
        group="large",
    )
    def test_large_baseline(self, shared_dist_ctx_globaldask, my_ds, benchmark):
        ctx = shared_dist_ctx_globaldask
        udf = EchoUDF()

        def doit():
            result = ctx.run_udf(dataset=my_ds, udf=udf)
            return result['intensity'].raw_data.sum(axis=0)

        benchmark(doit)

    @pytest.mark.benchmark(
        group="large",
    )
    def test_large_delayed(self, shared_dist_ctx_globaldask, my_ds, benchmark):
        ctx = Context(executor=DelayedJobExecutor())
        udf = EchoUDF()
        resources = DelayedJobExecutor.get_resources_from_udfs(udf)

        def doit():
            result = ctx.run_udf(dataset=my_ds, udf=udf)
            return result['intensity'].raw_data.sum(axis=0).compute(resources=resources)

        benchmark(doit)

    @pytest.mark.benchmark(
        group="large",
    )
    def test_large_delayed_merge(self, shared_dist_ctx_globaldask, my_ds, benchmark):
        ctx = Context(executor=DelayedJobExecutor())
        udf = EchoMergeUDF()
        resources = DelayedJobExecutor.get_resources_from_udfs(udf)

        def doit():
            result = ctx.run_udf(dataset=my_ds, udf=udf)
            return result['intensity'].raw_data.sum(axis=0).compute(resources=resources)

        benchmark(doit)

    @pytest.mark.benchmark(
        group="large",
    )
    def test_large_dsdirect(self, shared_dist_ctx_globaldask, my_ds, benchmark):
        resources = DelayedJobExecutor.get_resources_from_udfs(EchoUDF())

        def doit():
            # There seems to be some form of caching if the sum is calculated
            # repeatedly on the same dask arrays
            dask_array, workers = make_dask_array(my_ds, dtype=my_ds.dtype)
            assert len(dask_array.shape) == 4
            return dask_array.sum(axis=(0, 1)).compute(resources=resources)

        benchmark(doit)
