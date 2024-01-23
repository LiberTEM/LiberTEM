import time
import copy
from unittest import mock
import contextlib

import pytest
import numpy as np
import sparse
import scipy.sparse
from libertem.common.buffers import BufferWrapper
from libertem.common.executor import (
    TaskProtocol, WorkerQueue, TaskCommHandler,
)
from libertem.io.dataset.base import MMapBackend
from libertem.udf import UDF, UDFRunCancelled
from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.base import NoOpUDF, MergeAttrMapping
from libertem.api import Context
from libertem.common.exceptions import ExecutorSpecException


def test_ctx_load(lt_ctx, default_raw):
    lt_ctx.load(
        "raw",
        path=default_raw._path,
        nav_shape=(16, 16),
        dtype="float32",
        sig_shape=(128, 128),
    )


def test_run_empty_udf_list(lt_ctx, default_raw):
    ds = lt_ctx.load(
        "raw",
        path=default_raw._path,
        nav_shape=(16, 16),
        dtype="float32",
        sig_shape=(128, 128),
    )
    with pytest.raises(ValueError) as em:
        lt_ctx.run_udf(dataset=ds, udf=[])
    em.match("^empty list of UDFs - nothing to do!$")


def test_run_udf_with_io_backend(lt_ctx, default_raw):
    io_backend = MMapBackend(enable_readahead_hints=True)
    lt_ctx.load(
        "raw",
        path=default_raw._path,
        io_backend=io_backend,
        dtype="float32",
        nav_shape=(16, 16),
        sig_shape=(128, 128),
    )
    res = lt_ctx.run_udf(dataset=default_raw, udf=SumUDF())
    assert np.array(res['intensity']).shape == (128, 128)


# ignore 'no plottable channels' from NoOpUDF:
@pytest.mark.filterwarnings('ignore:no plottable channels:UserWarning')
@pytest.mark.parametrize('progress', (True, False))
@pytest.mark.parametrize('plots', (None, True))
def test_multi_udf(lt_ctx, default_raw, progress, plots):
    udfs = [NoOpUDF(), SumUDF(), SumSigUDF()]
    combined_res = lt_ctx.run_udf(dataset=default_raw, udf=udfs, progress=progress, plots=plots)
    ref_res = (
        lt_ctx.run_udf(dataset=default_raw, udf=udfs[0]),
        lt_ctx.run_udf(dataset=default_raw, udf=udfs[1]),
        lt_ctx.run_udf(dataset=default_raw, udf=udfs[2]),
    )

    assert isinstance(ref_res[0], dict)
    for index, res in enumerate(ref_res):
        for key in res.keys():
            assert np.all(res[key].data == combined_res[index][key].data)


@pytest.mark.filterwarnings('ignore::UserWarning')  # ignore 'no plottable channels' from NoOpUDF
@pytest.mark.parametrize('progress', (True, False))
@pytest.mark.parametrize('plots', (None, True))
@pytest.mark.asyncio
async def test_multi_udf_async(lt_ctx, default_raw, progress, plots):
    udfs = [NoOpUDF(), SumUDF(), SumSigUDF()]
    combined_res = await lt_ctx.run_udf(
        dataset=default_raw, udf=udfs, progress=progress, plots=plots, sync=False
    )
    single_res = (
        await lt_ctx.run_udf(dataset=default_raw, udf=udfs[0], sync=False),
        await lt_ctx.run_udf(dataset=default_raw, udf=udfs[1], sync=False),
        await lt_ctx.run_udf(dataset=default_raw, udf=udfs[2], sync=False),
    )
    ref_res = (
        lt_ctx.run_udf(dataset=default_raw, udf=udfs[0]),
        lt_ctx.run_udf(dataset=default_raw, udf=udfs[1]),
        lt_ctx.run_udf(dataset=default_raw, udf=udfs[2]),
    )
    assert isinstance(ref_res[0], dict)
    for index, res in enumerate(ref_res):
        for key in res.keys():
            assert np.all(res[key].data == combined_res[index][key].data)
            assert np.all(single_res[index][key].data == combined_res[index][key].data)


@pytest.mark.filterwarnings('ignore::UserWarning')  # ignore 'no plottable channels' from NoOpUDF
@pytest.mark.parametrize('progress', (True, False))
@pytest.mark.parametrize('plots', (None, True))
def test_udf_iter(lt_ctx, default_raw, progress, plots):
    udfs = [NoOpUDF(), SumUDF(), SumSigUDF()]
    for res in lt_ctx.run_udf_iter(dataset=default_raw, udf=udfs, progress=progress, plots=plots):
        ref_res = lt_ctx.run_udf(dataset=default_raw, udf=copy.deepcopy(udfs), roi=res.damage.data)
        # first one is empty, skipping
        # kind='sig'
        for key in ref_res[1].keys():
            ref_item = ref_res[1][key].data
            res_item = res.buffers[1][key].data
            assert np.all(ref_item == res_item)
        # kind='nav'
        for key in ref_res[2].keys():
            ref_item = ref_res[2][key].data[res.damage.data]
            res_item = res.buffers[2][key].data[res.damage.data]
            assert np.all(ref_item == res_item)


@pytest.mark.filterwarnings('ignore::UserWarning')  # ignore 'no plottable channels' from NoOpUDF
@pytest.mark.parametrize('progress', (True, False))
@pytest.mark.parametrize('plots', (None, True))
@pytest.mark.asyncio
async def test_udf_iter_async(lt_ctx, default_raw, progress, plots):
    udfs = [NoOpUDF(), SumUDF(), SumSigUDF()]
    async for res in lt_ctx.run_udf_iter(
            dataset=default_raw, udf=udfs, progress=progress, plots=plots, sync=False):
        # Nested execution of UDFs doesn't work, so we take a copy
        ref_res = lt_ctx.run_udf(dataset=default_raw, udf=copy.deepcopy(udfs), roi=res.damage.data)
        # first one is empty, skipping
        # kind='sig'
        for key in ref_res[1].keys():
            ref_item = ref_res[1][key].data
            res_item = res.buffers[1][key].data
            assert np.all(ref_item == res_item)
        # kind='nav'
        for key in ref_res[2].keys():
            ref_item = ref_res[2][key].data[res.damage.data]
            res_item = res.buffers[2][key].data[res.damage.data]
            assert np.all(ref_item == res_item)


@pytest.mark.filterwarnings('ignore::UserWarning')  # ignore 'no plottable channels' from NoOpUDF
@pytest.mark.parametrize(
    'plots', (
        True,
        [[], ['intensity']],
        [[], [('intensity', np.abs), ('intensity', np.sin)]],
        None
    )
)
def test_plots(lt_ctx, default_raw, plots):
    udfs = [NoOpUDF(), SumUDF(), SumSigUDF()]
    if plots is None:

        def test_channel(udf_result, damage):
            result = udf_result['intensity'].data
            if udf_result['intensity'].kind == 'nav':
                res_damage = damage
            else:
                res_damage = True
            return (result, res_damage)

        real_plots = [
            lt_ctx.plot_class(
                dataset=default_raw, udf=udfs[1], channel=test_channel,
                title="Test title"
            ),
            lt_ctx.plot_class(
                dataset=default_raw, udf=udfs[2], channel=test_channel,
                title="Test title"
            )
        ]
        mock_target = real_plots[0]
    else:
        real_plots = plots
        mock_target = lt_ctx.plot_class
    with mock.patch.object(mock_target, 'update') as plot_update:
        with mock.patch.object(mock_target, 'display') as plot_display:
            if plots is None:
                real_plots[0].display()
                real_plots[1].display()
            combined_res = lt_ctx.run_udf(dataset=default_raw, udf=udfs, plots=real_plots)
            plot_update.assert_called()
            plot_display.assert_called()
    ref_res = (
        lt_ctx.run_udf(dataset=default_raw, udf=udfs[0]),
        lt_ctx.run_udf(dataset=default_raw, udf=udfs[1]),
        lt_ctx.run_udf(dataset=default_raw, udf=udfs[2]),
    )

    assert isinstance(ref_res[0], dict)
    for index, res in enumerate(ref_res):
        for key in res.keys():
            assert np.all(res[key].data == combined_res[index][key].data)


@pytest.mark.parametrize('plots', ([['hello']], [[('hello', np.abs)]]))
def test_plots_fail(lt_ctx, default_raw, plots):
    udfs = [NoOpUDF()]
    with pytest.raises(ValueError):
        lt_ctx.run_udf(dataset=default_raw, udf=udfs, plots=plots)


@pytest.mark.parametrize(
    'use_roi', (False, True)
)
def test_display(lt_ctx, default_raw, use_roi):
    if use_roi:
        roi = np.random.choice((True, False), size=default_raw.shape.nav)
    else:
        roi = None
    udf = SumUDF()
    d = lt_ctx.display(dataset=default_raw, udf=udf, roi=roi)
    print(d._repr_html_())


@pytest.mark.parametrize(
    'dtype', (bool, int, float, None)
)
def test_roi_dtype(lt_ctx, default_raw, dtype):
    roi = np.zeros(default_raw.shape.nav, dtype=dtype)
    roi[0, 0] = 1

    ref_roi = np.zeros(default_raw.shape.nav, dtype=bool)
    ref_roi[0, 0] = True

    udf = SumUDF()
    if roi.dtype is not np.dtype(bool):
        match = f"ROI dtype is {roi.dtype}, expected bool. Attempting cast to bool."
        with pytest.warns(UserWarning, match=match):
            res = lt_ctx.run_udf(dataset=default_raw, udf=udf, roi=roi)
    else:
        res = lt_ctx.run_udf(dataset=default_raw, udf=udf, roi=roi)

    ref = lt_ctx.run_udf(dataset=default_raw, udf=udf, roi=ref_roi)

    assert np.all(res['intensity'].raw_data == ref['intensity'].raw_data)


roi_types = (
    sparse.COO,
    sparse.DOK,
    scipy.sparse.coo_matrix,
    scipy.sparse.csr_matrix,
    scipy.sparse.csr_array,
    list,
    tuple,
    int,
)


@pytest.mark.parametrize(
    'roi_type', roi_types
)
def test_allowed_rois(lt_ctx, default_raw, roi_type):
    roi = np.zeros(default_raw.shape.nav, dtype=bool)
    roi[3, 6] = True

    if issubclass(roi_type, sparse.SparseArray):
        roi_input = roi_type.from_numpy(roi)
    elif issubclass(roi_type, scipy.sparse.spmatrix):
        roi_input = roi_type(roi)
    elif hasattr(scipy.sparse, 'sparray') and issubclass(roi_type, scipy.sparse.sparray):
        roi_input = roi_type(roi)
    elif roi_type in (tuple, list):
        roi_input = np.argwhere(roi)
        roi_input = roi_type(roi_type((roi_type(row), True)) for row in roi_input)
    elif roi_type is int:
        roi_input = tuple(np.argwhere(roi).squeeze().tolist())
    else:
        raise ValueError('Unrecognized roi_type')

    udf = SumUDF()
    res = lt_ctx.run_udf(dataset=default_raw, udf=udf, roi=roi_input)
    ref = lt_ctx.run_udf(dataset=default_raw, udf=udf, roi=roi)
    assert np.all(res['intensity'].raw_data == ref['intensity'].raw_data)


def test_make_with_inline(inline_executor):
    ctx = Context.make_with('inline')
    assert isinstance(ctx.executor, inline_executor.__class__)


def test_make_with_inline_raises():
    with pytest.raises(ExecutorSpecException):
        Context.make_with('inline', cpus=4)


@pytest.mark.parametrize('n_threads', (None, 4, (2, 3)))
def test_make_with_threads(concurrent_executor, n_threads):
    ctx = Context.make_with('threads', cpus=n_threads)
    assert isinstance(ctx.executor, concurrent_executor.__class__)
    # No way to check number of workers in a concurrent.futures.Executor!


@pytest.mark.parametrize('exec_spec', ('threads', 'inline', 'delayed'))
def test_make_with_raises_gpus_no_support(exec_spec):
    with pytest.raises(ExecutorSpecException):
        Context.make_with(exec_spec, gpus=4)


def test_make_with_unrecognized():
    with pytest.raises(ExecutorSpecException):
        Context.make_with('not_an_executor')


def test_udf_cancellation(default_raw):
    from libertem.common.executor import JobCancelledError
    ctx = Context.make_with('inline')

    # As we don't have an API for explicit cancellation by the user yet, we
    # raise `JobCancelledError` from `process_frame` of the UDF.  In reality, it
    # would be raised either from somewhere inside the UDF runner on explicit
    # cancellation, or from the executor when the data source has run dry.

    class DumbUDF(UDF):
        def get_result_buffers(self):
            return {
                'stuff': self.buffer(kind='nav', dtype='float32'),
            }

        def process_frame(self, frame):
            raise JobCancelledError()

    res_iter = ctx.run_udf_iter(dataset=default_raw, udf=DumbUDF())

    # TODO: support `res_iter.cancel_run()` or similar
    with pytest.raises(UDFRunCancelled) as ex:
        for part in res_iter:
            pass

    assert ex.match(r"^UDF run cancelled after \d+ partitions$")


class DynamicParamsUDF(UDF):
    def __init__(self, latest_index, delay=0.0):
        super().__init__(latest_index=latest_index, delay=delay)

    def get_result_buffers(self) -> dict[str, BufferWrapper]:
        return {
            'index': self.buffer(kind='nav', dtype=int),
            'index_merge': self.buffer(kind='nav', dtype=int),
        }

    def process_partition(self, partition):
        print(f"DynamicParamsUDF {self.params.latest_index}")
        # no need to sleep if we have successfully updated the parameter:
        if self.params.latest_index == 0:
            time.sleep(self.params.delay)
        self.results.index[:] = self.params.latest_index

    def merge(self, dest: MergeAttrMapping, src: MergeAttrMapping):
        dest.index[:] = src.index
        dest.index_merge = self.params.latest_index


def test_dynamic_parameter_update_sync(lt_ctx, default_raw):
    result_iter = lt_ctx.run_udf_iter(
        dataset=default_raw, udf=[DynamicParamsUDF(latest_index=0)]
    )
    with contextlib.closing(result_iter) as result_iter:
        # because this is using the inline executor, we can guarantee
        # that the updated parameters are used for the next partition.
        for idx, part_res in enumerate(result_iter):
            result_iter.update_parameters_experimental([
                {"latest_index": idx + 1}
            ])
            print(idx, part_res.buffers[0]['index'].data)
        # `default_raw` has at least two partitions, so there should be
        # something non-zero in the result:
        assert not np.allclose(part_res.buffers[0]['index'], 0)
        assert not np.allclose(part_res.buffers[0]['index_merge'], 0)


@pytest.mark.parametrize('executor', ['pipelined', 'dask', 'inline', 'concurrent'])
def test_dynamic_parameter_update_integration(
    executor, local_cluster_ctx, pipelined_ctx, concurrent_executor,
):

    # we do this dance to re-use the existing executors, so we don't have to
    # mark this test as slow:
    ctx: Context
    if executor == 'dask':
        ctx = local_cluster_ctx
    elif executor == 'pipelined':
        ctx = pipelined_ctx
    elif executor == 'inline':
        ctx = Context.make_with('inline')
    elif executor == 'concurrent':
        ctx = Context(executor=concurrent_executor)
    else:
        raise ValueError('invalid executor name')

    num_workers = sum(
        w.nthreads for w in ctx.executor.get_available_workers()
    )
    parts = 15
    delay = 0.025

    # this should make it more likely that we wait long enough in the main
    # process for the first task to finish, meaning we get to update the index
    # as soon as possible
    delay_udf = delay / 2
    dataset = ctx.load(
        "memory",
        num_partitions=num_workers * parts,
        datashape=(num_workers * parts, 16, 16),
        sig_dims=2,
    )

    class DelayingCommHandler(TaskCommHandler):
        def handle_task(self, task: TaskProtocol, queue: WorkerQueue):
            # our tests only work if the tasks don't all get submitted at the
            # beginning of the `run_tasks` call - this simulates the live
            # processing scenario
            time.sleep(delay)
    dataset.get_task_comm_handler = lambda: DelayingCommHandler()

    result_iter = ctx.run_udf_iter(
        dataset=dataset, udf=[DynamicParamsUDF(latest_index=0, delay=delay_udf)]
    )
    with contextlib.closing(result_iter) as result_iter:
        for idx, part_res in enumerate(result_iter):
            result_iter.update_parameters_experimental([
                {"latest_index": idx + 1}
            ])
            print(idx, part_res.buffers[0]['index'].data)
        assert not np.allclose(part_res.buffers[0]['index'], 0)
        assert not np.allclose(part_res.buffers[0]['index_merge'], 0)

    # second run, to make sure we don't mess up any state:
    result_iter = ctx.run_udf_iter(
        dataset=dataset, udf=[DynamicParamsUDF(latest_index=0, delay=delay_udf)]
    )
    with contextlib.closing(result_iter) as result_iter:
        for idx, part_res in enumerate(result_iter):
            result_iter.update_parameters_experimental([
                {"latest_index": idx + 1}
            ])
            print(idx, part_res.buffers[0]['index'].data)
        assert not np.allclose(part_res.buffers[0]['index'], 0)
        assert not np.allclose(part_res.buffers[0]['index_merge'], 0)


@pytest.mark.asyncio
async def test_dynamic_parameter_update_async(lt_ctx, default_raw):
    result_iter = lt_ctx.run_udf_iter(
        dataset=default_raw,
        udf=[DynamicParamsUDF(latest_index=0)],
        sync=False,
    )
    try:
        # because this is using the inline executor, we can guarantee
        # that the updated parameters are used for the next partition.
        idx = 0  # no aenumerate in stdlib :(
        async for part_res in result_iter:
            await result_iter.update_parameters_experimental([
                {"latest_index": idx + 1}
            ])
            print(idx, part_res.buffers[0]['index'].data)
            idx += 1
        # `default_raw` has at least two partitions, so there should be
        # something non-zero in the result:
        assert not np.allclose(part_res.buffers[0]['index'], 0)
        assert not np.allclose(part_res.buffers[0]['index_merge'], 0)
    finally:
        await result_iter.aclose()


class DynamicParamsAuxUDF(UDF):
    def __init__(self, latest_index):
        super().__init__(latest_index=latest_index)

    def get_result_buffers(self) -> dict[str, BufferWrapper]:
        return {
            'index': self.buffer(kind='nav', dtype=int),
            'index_merge': self.buffer(kind='nav', dtype=int),
        }

    def process_partition(self, partition):
        print(f"DynamicParamsAuxUDF.process_partition {self.params.latest_index}")
        self.results.index[:] = self.params.latest_index[:, 2]
        # the correct view is set:
        assert np.allclose(
            self.meta.coordinates,
            self.params.latest_index[:, :2],
        )

    def merge(self, dest: MergeAttrMapping, src: MergeAttrMapping):
        print(self.params.latest_index.shape)
        print(f"DynamicParamsAuxUDF.merge: {self.params.latest_index}")
        dest.index[:] = src.index
        dest.index_merge = self.params.latest_index[:, 2]


def test_dynamic_parameter_aux_data(lt_ctx, default_raw):
    def _aux_data(idx):
        coords = np.moveaxis(
            np.indices(default_raw.shape.nav),
            0,
            -1,
        ).astype(np.float32).reshape((-1, 2))
        idx_arr = np.zeros((coords.shape[0], 1), dtype=np.float32)
        idx_arr[:] = idx
        aux = DynamicParamsAuxUDF.aux_data(
            data=np.hstack([coords, idx_arr]),
            kind="nav", dtype=np.float32, extra_shape=(3,)
        )
        return aux
    aux = _aux_data(0)
    result_iter = lt_ctx.run_udf_iter(
        dataset=default_raw, udf=[DynamicParamsAuxUDF(latest_index=aux)]
    )
    with contextlib.closing(result_iter) as result_iter:
        # because this is using the inline executor, we can guarantee
        # that the updated parameters are used for the next partition.
        for idx, part_res in enumerate(result_iter):
            aux = _aux_data(idx + 1)
            result_iter.update_parameters_experimental([
                {"latest_index": aux}
            ])
        # `default_raw` has at least two partitions, so there should be
        # something non-zero in the result:
        assert not np.allclose(part_res.buffers[0]['index'], 0)
        assert not np.allclose(part_res.buffers[0]['index_merge'], 0)


@pytest.mark.asyncio
async def test_res_iter_athrow(lt_ctx: Context, default_raw):
    result_iter = lt_ctx.run_udf_iter(
        dataset=default_raw,
        udf=SumUDF(),
        sync=False,
    )

    with pytest.raises(RuntimeError):
        async for res in result_iter:
            await result_iter.athrow(RuntimeError("stuff"))


def test_res_iter_throw(lt_ctx: Context, default_raw):
    result_iter = lt_ctx.run_udf_iter(
        dataset=default_raw,
        udf=SumUDF(),
    )

    with pytest.raises(RuntimeError):
        for res in result_iter:
            result_iter.throw(RuntimeError("stuff"))
