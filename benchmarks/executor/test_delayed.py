import numpy as np
import dask.array as da
import pytest

from libertem.udf.base import UDF
from libertem.api import Context
from libertem.executor.delayed import DelayedJobExecutor


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
    def dask_merge(self, ordered_results):
        intensity = da.stack([b.intensity for b in ordered_results.values()]).sum(axis=0)
        self.results.get_buffer('intensity').update_data(intensity)


class MySumSigUDF(UDF):
    def get_result_buffers(self):
        return {
            'intensity': self.buffer(
                kind="nav", dtype=self.meta.input_dtype
            ),
        }

    def process_tile(self, tile):
        self.results.intensity[:] += np.sum(tile, axis=tuple(range(1, len(tile.shape))))


class MySumSigMergeUDF(MySumSigUDF):
    def dask_merge(self, ordered_results):
        intensity = np.concatenate([b.intensity for b in ordered_results.values()])
        self.results.get_buffer('intensity').update_data(intensity)


class EchoUDF(UDF):
    def get_result_buffers(self):
        return {
            'intensity': self.buffer(
                kind='nav', dtype=self.meta.input_dtype, extra_shape=self.meta.dataset_shape.sig
            )
        }

    def process_tile(self, tile):
        self.results.intensity[:] = tile


class EchoMergeUDF(EchoUDF):
    def dask_merge(self, ordered_results):
        intensity = np.concatenate([b.intensity for b in ordered_results.values()])
        self.results.get_buffer('intensity').update_data(intensity)


class Test:
    @pytest.mark.benchmark(
        group="sum",
    )
    def test_sum_baseline(self, shared_dist_ctx_globaldask, medium_raw_float32, benchmark):
        ctx = shared_dist_ctx_globaldask
        ds = medium_raw_float32
        udf = MySumUDF()
        benchmark(
            ctx.run_udf,
            dataset=ds,
            udf=udf
        )

    @pytest.mark.benchmark(
        group="sum",
    )
    def test_sum_delayed(self, shared_dist_ctx_globaldask, medium_raw_float32, benchmark):
        ctx = Context(executor=DelayedJobExecutor())
        ds = medium_raw_float32
        udf = MySumUDF()
        resources = DelayedJobExecutor.get_resources_from_udfs(udf)

        def doit():
            result = ctx.run_udf(dataset=ds, udf=udf)
            # Make sure we run on the same number of workers
            return result['intensity'].raw_data.compute(resources=resources)

        benchmark(doit)

    @pytest.mark.benchmark(
        group="sum",
    )
    def test_sum_delayed_merge(self, shared_dist_ctx_globaldask, medium_raw_float32, benchmark):
        ctx = Context(executor=DelayedJobExecutor())
        ds = medium_raw_float32
        udf = MySumMergeUDF()
        resources = DelayedJobExecutor.get_resources_from_udfs(udf)

        def doit():
            result = ctx.run_udf(dataset=ds, udf=udf)
            return result['intensity'].raw_data.compute(resources=resources)

        benchmark(doit)

    @pytest.mark.benchmark(
        group="sumsig",
    )
    def test_sumsig_baseline(self, shared_dist_ctx_globaldask, medium_raw_float32, benchmark):
        ctx = shared_dist_ctx_globaldask
        ds = medium_raw_float32
        udf = MySumSigUDF()
        benchmark(
            ctx.run_udf,
            dataset=ds,
            udf=udf
        )

    @pytest.mark.benchmark(
        group="sumsig",
    )
    def test_sumsig_delayed(self, shared_dist_ctx_globaldask, medium_raw_float32, benchmark):
        ctx = Context(executor=DelayedJobExecutor())
        ds = medium_raw_float32
        udf = MySumSigUDF()
        resources = DelayedJobExecutor.get_resources_from_udfs(udf)

        def doit():
            result = ctx.run_udf(dataset=ds, udf=udf)
            return result['intensity'].raw_data.compute(resources=resources)

        benchmark(doit)

    @pytest.mark.benchmark(
        group="sumsig",
    )
    def test_sumsig_delayed_merge(self, shared_dist_ctx_globaldask, medium_raw_float32, benchmark):
        ctx = Context(executor=DelayedJobExecutor())
        ds = medium_raw_float32
        udf = MySumSigMergeUDF()
        resources = DelayedJobExecutor.get_resources_from_udfs(udf)

        def doit():
            result = ctx.run_udf(dataset=ds, udf=udf)
            return result['intensity'].raw_data.compute(resources=resources)

        benchmark(doit)

    @pytest.mark.benchmark(
        group="echo",
    )
    def test_echo_baseline(self, shared_dist_ctx_globaldask, medium_raw_float32, benchmark):
        ctx = shared_dist_ctx_globaldask
        ds = medium_raw_float32
        udf = EchoUDF()

        def doit():
            result = ctx.run_udf(dataset=ds, udf=udf)
            return result['intensity'].raw_data.sum(axis=0)

        benchmark(doit)

    @pytest.mark.benchmark(
        group="echo",
    )
    def test_echo_delayed(self, shared_dist_ctx_globaldask, medium_raw_float32, benchmark):
        ctx = Context(executor=DelayedJobExecutor())
        ds = medium_raw_float32
        udf = EchoUDF()
        resources = DelayedJobExecutor.get_resources_from_udfs(udf)

        def doit():
            result = ctx.run_udf(dataset=ds, udf=udf)
            return result['intensity'].raw_data.sum(axis=0).compute(resources=resources)

        benchmark(doit)

    @pytest.mark.benchmark(
        group="echo",
    )
    def test_echo_delayed_merge(self, shared_dist_ctx_globaldask, medium_raw_float32, benchmark):
        ctx = Context(executor=DelayedJobExecutor())
        ds = medium_raw_float32
        udf = EchoMergeUDF()
        resources = DelayedJobExecutor.get_resources_from_udfs(udf)

        def doit():
            result = ctx.run_udf(dataset=ds, udf=udf)
            return result['intensity'].raw_data.sum(axis=0).compute(resources=resources)

        benchmark(doit)
