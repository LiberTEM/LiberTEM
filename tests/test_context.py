import pytest
import numpy as np
from unittest import mock
import copy
import sparse
import scipy.sparse

from libertem.io.dataset.base import MMapBackend
from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.base import NoOpUDF
from libertem.api import Context


def test_ctx_load(lt_ctx, default_raw):
    lt_ctx.load(
        "raw",
        path=default_raw._path,
        nav_shape=(16, 16),
        dtype="float32",
        sig_shape=(128, 128),
    )


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


@pytest.mark.parametrize(
    'roi_type', (
        sparse.COO,
        sparse.DOK,
        scipy.sparse.coo_matrix,
        scipy.sparse.csr_matrix,
        scipy.sparse.csr_array,
        list,
        tuple,
    )
)
def test_allowed_rois(lt_ctx, default_raw, roi_type):
    roi = np.zeros(default_raw.shape.nav, dtype=bool)
    roi[0, 0] = True
    roi[3, 6] = True

    if issubclass(roi_type, sparse.SparseArray):
        roi_input = roi_type.from_numpy(roi)
    elif issubclass(roi_type, scipy.sparse.spmatrix):
        roi_input = roi_type(roi)
    elif roi_type in (tuple, list):
        roi_input = np.argwhere(roi)
        roi_input = roi_type(roi_type((roi_type(row), True)) for row in roi_input)
    else:
        raise ValueError('Unrecognized roi_type')

    udf = SumUDF()
    res = lt_ctx.run_udf(dataset=default_raw, udf=udf, roi=roi_input)
    ref = lt_ctx.run_udf(dataset=default_raw, udf=udf, roi=roi)
    assert np.all(res['intensity'].raw_data == ref['intensity'].raw_data)


def test_make_with_inline(inline_executor):
    ctx = Context.make_with('inline')
    assert isinstance(ctx.executor, inline_executor.__class__)


def test_make_with_threads(concurrent_executor):
    ctx = Context.make_with('threads')
    assert isinstance(ctx.executor, concurrent_executor.__class__)


def test_make_with_unrecognized():
    with pytest.raises(ValueError):
        Context.make_with('not_an_executor')
