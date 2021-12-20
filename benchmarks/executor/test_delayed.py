import numpy as np
import pytest

from libertem.udf.base import UDF
from libertem.udf.stddev import StdDevUDF
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


class MyStdDevUDF(StdDevUDF):
    pass


# FIXME check if renamed?
if hasattr(MyStdDevUDF, 'dask_merge'):
    delattr(MyStdDevUDF, 'dask_merge')


class MyStdDevMergeUDF(MyStdDevUDF):
    # Copied and adapted from tests/executor/test_delayed
    # FIXME import instead?
    def dask_merge(self, ordered_results):
        n_frames = np.stack([b.num_frames[0] for b in ordered_results.values()])
        pixel_sums = np.stack([b.sum for b in ordered_results.values()])
        pixel_varsums = np.stack([b.varsum for b in ordered_results.values()])

        # Expand n_frames to be broadcastable
        extra_dims = pixel_sums.ndim - n_frames.ndim
        n_frames = n_frames.reshape(n_frames.shape + (1,) * extra_dims)

        cumulative_frames = np.cumsum(n_frames, axis=0)
        cumulative_sum = np.cumsum(pixel_sums, axis=0)
        sumsum = cumulative_sum[-1, ...]
        total_frames = cumulative_frames[-1, 0]

        mean_0 = cumulative_sum / cumulative_frames
        # Handle the fact that mean_0 is indexed to results from
        # up-to the partition before. We shift everything one to
        # the right, and we don't care about result 0 because it
        # is by definiition replaced with varsum[0, ...]
        mean_0 = np.roll(mean_0, 1, axis=0)

        mean_1 = pixel_sums / n_frames
        delta = mean_1 - mean_0
        mean = mean_0 + (n_frames * delta) / cumulative_frames
        partial_delta = mean_1 - mean
        varsum = pixel_varsums + (n_frames * delta * partial_delta)
        varsum[0, ...] = pixel_varsums[0, ...]
        varsum_total = np.sum(varsum, axis=0)

        self.results.get_buffer('sum').update_data(sumsum)
        self.results.get_buffer('varsum').update_data(varsum_total)
        self.results.get_buffer('num_frames').update_data(total_frames)


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
        intensity = np.stack([b.intensity for b in ordered_results.values()]).sum(axis=0)
        self.results.get_buffer('intensity').update_data(intensity)


class MySumSigUDF(UDF):
    def get_result_buffers(self):
        return {
            'intensity': self.buffer(
                kind="nav", dtype=self.meta.input_dtype
            ),
        }

    # Make sure we don't have default merge so that there is no default dask_merge
    def merge(self, dest, src):
        return super().merge(dest, src)

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
        sl = (..., *self.meta.sig_slice.get())
        self.results.intensity[sl] = tile

    # Make sure we don't have default merge so that there is nod efault dask_merge
    def merge(self, dest, src):
        return super().merge(dest, src)


class EchoMergeUDF(EchoUDF):
    def dask_merge(self, ordered_results):
        intensity = np.concatenate([b.intensity for b in ordered_results.values()])
        self.results.get_buffer('intensity').update_data(intensity)


class Test:
    @pytest.mark.benchmark(
        group="std",
    )
    def test_std_baseline(self, shared_dist_ctx_globaldask, my_ds, benchmark):
        ctx = shared_dist_ctx_globaldask
        udf = MyStdDevUDF()
        benchmark(
            ctx.run_udf,
            dataset=my_ds,
            udf=udf
        )

    @pytest.mark.benchmark(
        group="std",
    )
    def test_std_delayed(self, shared_dist_ctx_globaldask, my_ds, benchmark):
        ctx = Context(executor=DelayedJobExecutor())
        udf = MyStdDevUDF()
        resources = DelayedJobExecutor.get_resources_from_udfs(udf)

        def doit():
            result = ctx.run_udf(dataset=my_ds, udf=udf)
            # Make sure we run on the same number of workers
            return result['std'].raw_data.compute(resources=resources)

        benchmark(doit)

    @pytest.mark.benchmark(
        group="std",
    )
    def test_std_delayed_merge(self, shared_dist_ctx_globaldask, my_ds, benchmark):
        ctx = Context(executor=DelayedJobExecutor())
        udf = MyStdDevMergeUDF()
        resources = DelayedJobExecutor.get_resources_from_udfs(udf)

        def doit():
            result = ctx.run_udf(dataset=my_ds, udf=udf)
            return result['std'].raw_data.compute(resources=resources)

        benchmark(doit)

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
