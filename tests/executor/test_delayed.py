import pytest
from functools import partial
import sys

import dask
import dask.array as da
import numpy as np

from libertem.udf.base import UDF
from libertem.udf.logsum import LogsumUDF
from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.stddev import StdDevUDF
from libertem.udf.masks import ApplyMasksUDF
from libertem.udf.raw import PickUDF

from libertem.io.dataset.memory import MemoryDataSet
from libertem.executor.utils.dask_inplace import DaskInplaceWrapper

from utils import _mk_random


def test_inplace_python36():
    data = _mk_random((55, 55), dtype=np.float32)
    dask_wrapped = DaskInplaceWrapper(da.from_array(data.copy()))

    dask_wrapped.set_slice(np.s_[2, :])
    if sys.version_info < (3, 7):
        with pytest.raises(NotImplementedError):
            dask_wrapped[np.s_[3]] = 55.


class CountUDF(UDF):
    def process_tile(self, tile):
        self.results.count[:] = np.ones(tile.shape[:1], dtype=np.int32)

    def get_result_buffers(self):
        return {'count': self.buffer(kind='nav', dtype=np.int32)}


class StrUDF(UDF):
    def process_tile(self, tile):
        self.results.str[:] = np.full(tile.shape[:1], 'a', dtype=np.dtype('S1'))

    def get_result_buffers(self):
        return {'str': self.buffer(kind='nav', dtype=np.dtype('S1'))}


class SumUDFDask(SumUDF):
    def dask_merge(self, ordered_results):
        intensity_chunks = [b.intensity for b in ordered_results.values()]
        intensity_sum = da.stack(intensity_chunks, axis=0).sum(axis=0)
        self.results.get_buffer('intensity').update_data(intensity_sum)


class SumSigUDFDask(SumSigUDF):
    def dask_merge(self, ordered_results):
        intensity = da.concatenate([b.intensity for b in ordered_results.values()])
        self.results.get_buffer('intensity').update_data(intensity)


class StdDevUDFDask(StdDevUDF):
    def dask_merge(self, ordered_results):
        n_frames = da.concatenate([[b.num_frames[0] for b in ordered_results.values()]])
        pixel_sums = da.concatenate([[b.sum for b in ordered_results.values()]])
        pixel_varsums = da.concatenate([[b.varsum for b in ordered_results.values()]])

        n_frames = da.rechunk(n_frames, (-1,) * n_frames.ndim)
        pixel_sums = da.rechunk(pixel_sums, (-1,) * pixel_sums.ndim)
        pixel_varsums = da.rechunk(pixel_varsums, (-1,) * pixel_varsums.ndim)

        # Expand n_frames to be broadcastable
        extra_dims = pixel_sums.ndim - n_frames.ndim
        n_frames = n_frames.reshape(n_frames.shape + (1,) * extra_dims)

        cumulative_frames = da.cumsum(n_frames, axis=0)
        cumulative_sum = da.cumsum(pixel_sums, axis=0)
        sumsum = cumulative_sum[-1, ...]
        total_frames = cumulative_frames[-1, 0]

        mean_0 = cumulative_sum / cumulative_frames
        # Handle the fact that mean_0 is indexed to results from
        # up-to the partition before. We shift everything one to
        # the right, and we don't care about result 0 because it
        # is by definiition replaced with varsum[0, ...]
        mean_0 = da.roll(mean_0, 1, axis=0)

        mean_1 = pixel_sums / n_frames
        delta = mean_1 - mean_0
        mean = mean_0 + (n_frames * delta) / cumulative_frames
        partial_delta = mean_1 - mean
        varsum = pixel_varsums + (n_frames * delta * partial_delta)
        varsum[0, ...] = pixel_varsums[0, ...]
        varsum_cumulative = da.cumsum(varsum, axis=0)
        varsum_total = varsum_cumulative[-1, ...]

        self.results.get_buffer('sum').update_data(sumsum)
        self.results.get_buffer('varsum').update_data(varsum_total)
        self.results.get_buffer('num_frames').update_data(total_frames)


def get_dataset(ctx, shape, tileshape, num_partitions, sig_dims, use_roi=False):
    data = _mk_random(shape)
    dataset = MemoryDataSet(data=data, tileshape=tileshape,
                            num_partitions=num_partitions, sig_dims=sig_dims)
    dataset.initialize(ctx)
    if use_roi:
        roi = np.random.choice([True, False], dataset._shape.nav)
    else:
        roi = None
    return {'data': data, 'dataset': dataset, 'roi': roi}


def ds_dims(dataset):
    ds_shape = dataset._shape
    n_sig_dims = ds_shape.sig.dims
    nav_dims = tuple(range(ds_shape.dims - n_sig_dims))
    sig_dims = tuple(range(ds_shape.nav.dims, ds_shape.dims))
    return ds_shape, nav_dims, sig_dims


def logsum(udf_params, ds_dict):
    data = ds_dict['data']
    dataset = ds_dict['dataset']
    roi = ds_dict.get('roi', None)
    ds_shape, nav_dims, sig_dims = ds_dims(dataset)

    flat_nav_data, flat_sig_dims = flatten_with_roi(data, roi, ds_shape)

    udf = LogsumUDF(**udf_params)
    minima = np.min(flat_nav_data, axis=flat_sig_dims)
    for _ in sig_dims:
        minima = np.expand_dims(minima, -1)
    naive_result = np.log(flat_nav_data - minima + 1).sum(axis=0)
    return {'udf': udf, 'naive_result': {'logsum': naive_result}}


def count(udf_params, ds_dict):
    data = ds_dict['data']
    dataset = ds_dict['dataset']
    roi = ds_dict.get('roi', None)
    ds_shape, nav_dims, sig_dims = ds_dims(dataset)

    flat_nav_data, flat_sig_dims = flatten_with_roi(data, roi, ds_shape)

    udf = CountUDF(**udf_params)
    naive_result = np.ones(flat_nav_data.shape[:1], dtype=np.int32)
    naive_result = fill_nav_with_roi(ds_shape, naive_result, roi, fill=0)
    return {'udf': udf, 'naive_result': {'count': naive_result}}


def string(udf_params, ds_dict):
    data = ds_dict['data']
    dataset = ds_dict['dataset']
    roi = ds_dict.get('roi', None)
    ds_shape, nav_dims, sig_dims = ds_dims(dataset)

    flat_nav_data, flat_sig_dims = flatten_with_roi(data, roi, ds_shape)

    udf = StrUDF(**udf_params)
    naive_result = np.full(flat_nav_data.shape[:1], 'a', dtype=np.dtype('S1'))
    naive_result = fill_nav_with_roi(ds_shape, naive_result, roi, fill='')
    return {'udf': udf, 'naive_result': {'str': naive_result}}


def pick(udf_params, ds_dict):
    data = ds_dict['data']
    dataset = ds_dict['dataset']
    ds_shape, nav_dims, sig_dims = ds_dims(dataset)

    roi = np.zeros(tuple(ds_shape.nav), dtype=bool)
    roi[-1, -1] = True

    flat_nav_data, flat_sig_dims = flatten_with_roi(data, roi, ds_shape)

    udf = PickUDF(**udf_params)
    naive_result = flat_nav_data
    return {'udf': udf, 'naive_result': {'intensity': naive_result}, 'roi': roi}


def _stddev(udf_class, udf_params, ds_dict):
    data = ds_dict['data']
    dataset = ds_dict['dataset']
    roi = ds_dict.get('roi', None)
    ds_shape, nav_dims, sig_dims = ds_dims(dataset)

    flat_nav_data, flat_sig_dims = flatten_with_roi(data, roi, ds_shape)

    udf = udf_class(**udf_params)

    direct_results = {}
    direct_results['sum'] = flat_nav_data.sum(axis=0)
    direct_results['varsum'] = None
    direct_results['num_frames'] = flat_nav_data.shape[0]
    direct_results['var'] = np.var(flat_nav_data, axis=0, dtype=np.float64)
    direct_results['std'] = np.std(flat_nav_data, axis=0, dtype=np.float64)
    direct_results['mean'] = flat_nav_data.mean(axis=0, dtype=np.float64)

    return {'udf': udf, 'naive_result': direct_results, 'tolerance': 1e-3}


def stddev(udf_params, ds_dict):
    return _stddev(StdDevUDF, udf_params, ds_dict)


def stddevdask(udf_params, ds_dict):
    return _stddev(StdDevUDFDask, udf_params, ds_dict)


def _navsum(udf_class, udf_params, ds_dict):
    data = ds_dict['data']
    dataset = ds_dict['dataset']
    roi = ds_dict.get('roi', None)
    ds_shape, nav_dims, sig_dims = ds_dims(dataset)

    flat_nav_data, _ = flatten_with_roi(data, roi, ds_shape)

    udf = udf_class(**udf_params)
    naive_result = flat_nav_data.sum(axis=0)
    return {'udf': udf, 'naive_result': {'intensity': naive_result}}


def navsum(udf_params, ds_dict):
    return _navsum(SumUDF, udf_params, ds_dict)


def navsumdask(udf_params, ds_dict):
    return _navsum(SumUDFDask, udf_params, ds_dict)


def _sigsum(udf_class, udf_params, ds_dict):
    data = ds_dict['data']
    dataset = ds_dict['dataset']
    roi = ds_dict.get('roi', None)
    ds_shape, nav_dims, sig_dims = ds_dims(dataset)

    flat_nav_data, flat_sig_dims = flatten_with_roi(data, roi, ds_shape)

    udf = udf_class(**udf_params)
    naive_result = flat_nav_data.sum(axis=flat_sig_dims)
    naive_result = fill_nav_with_roi(ds_shape, naive_result, roi)

    return {'udf': udf, 'naive_result': {'intensity': naive_result}}


def sigsum(udf_params, ds_dict):
    return _sigsum(SumSigUDF, udf_params, ds_dict)


def sigsumdask(udf_params, ds_dict):
    return _sigsum(SumSigUDFDask, udf_params, ds_dict)


def _mask_fac(mask_slice, mask_value, ds_shape):
    m = np.zeros(ds_shape.sig)
    m[mask_slice] = mask_value
    return m


def mask(udf_params, ds_dict):
    data = ds_dict['data']
    dataset = ds_dict['dataset']
    roi = ds_dict.get('roi', None)
    ds_shape, nav_dims, sig_dims = ds_dims(dataset)

    flat_nav_data, flat_sig_dims = flatten_with_roi(data, roi, ds_shape)

    mask_slice = udf_params.pop('slices', [np.s_[-1, -1]])
    mask_value = udf_params.pop('values', [1.3])

    factories = []
    for sl, val in zip(mask_slice, mask_value):
        factories.append(partial(_mask_fac, sl, val, ds_shape))

    udf = ApplyMasksUDF(mask_factories=factories)

    results = []
    for fac in factories:
        _mask = fac()
        masked_result = (flat_nav_data * _mask[np.newaxis, ...]).sum(axis=flat_sig_dims)
        results.append(fill_nav_with_roi(ds_shape, masked_result, roi))
    naive_result = np.stack(results, axis=-1)

    return {'udf': udf, 'naive_result': {'intensity': naive_result}}


def fill_nav_with_roi(ds_shape, flat_result, roi, fill=np.nan):
    nav_shape = tuple(ds_shape.nav)
    if roi is None:
        return flat_result.reshape(nav_shape)
    else:
        full = np.full(nav_shape, fill_value=fill, dtype=flat_result.dtype).ravel()
        full[roi.ravel()] = flat_result
        return full.reshape(nav_shape)


def flatten_with_roi(data, roi, ds_shape):
    n_sig_dims = ds_shape.sig.dims
    flat_nav_data = data.reshape(-1, *tuple(ds_shape.sig))
    flat_sig_dims = tuple(range(-n_sig_dims, 0))
    if roi is not None:
        flat_nav_data = flat_nav_data[roi.ravel()]
    return flat_nav_data, flat_sig_dims


def build_udf_dict(udf_config, ds_dict):
    udf_dict = {'udf': [], 'naive_result': []}
    for udf in udf_config:
        factory = udf['factory']
        udf_params = udf['params']
        _udf_dict = factory(udf_params, ds_dict)

        udf_dict['udf'].append(_udf_dict['udf'])
        udf_dict['naive_result'].append(_udf_dict['naive_result'])

        if 'roi' in _udf_dict.keys():
            assert 'roi' not in udf_dict.keys(), 'Cannot define more than one custom ROI'
            udf_dict['roi'] = _udf_dict['roi']

    return udf_dict


def unpack_all(udf_results_dask):
    return tuple({k: v.data for k, v in udf_result.items()}
                 for udf_result in udf_results_dask)


def alldask(udf_results_dask):
    if not isinstance(udf_results_dask, tuple):
        udf_results_dask = (udf_results_dask,)
    for udf_result in udf_results_dask:
        for daskbuffer in udf_result.values():
            assert isinstance(daskbuffer.raw_data, da.Array)
            assert isinstance(daskbuffer.data, da.Array)


def try_convert_py(obj):
    try:
        return obj.item()
    except AttributeError:
        return obj


def allclose_with_nan(array1, array2, tol=None):
    if not isinstance(array1, np.ndarray) or not isinstance(array2, np.ndarray):
        assert try_convert_py(array1) == try_convert_py(array2)
        return
    assert array1.shape == array2.shape
    try:
        a1_nan = np.isnan(array1)
        a2_nan = np.isnan(array2)
        assert (a1_nan == a2_nan).all()
    except TypeError:
        # Cannot isnan non-numeric arrays
        pass
    tol_dict = {}
    if tol is not None:
        tol_dict = {'atol': tol}
    if np.issubdtype(array2.dtype, np.number):
        assert np.allclose(array1, array2, **tol_dict, equal_nan=True)
    else:
        assert (array1 == array2).all()


@pytest.mark.parametrize(
    "ds_config", ({'shape': (16, 8, 32, 64), 'tileshape': (8, 32, 64),
                   'num_partitions': 2, 'sig_dims': 2},
                  {'shape': (8, 16, 64, 64), 'tileshape': (8, 32, 64),
                   'num_partitions': 7, 'sig_dims': 2},
                  {'shape': (16, 16, 8, 64), 'tileshape': (8, 8, 64),
                   'num_partitions': 1, 'sig_dims': 2}),
    ids=['2part', '8part', '1part']
)
@pytest.mark.parametrize(
    "udf_config", (
        [
            {'factory': logsum, 'params': {}}],
        [
            {'factory': sigsum, 'params': {}}],
        [
            {'factory': navsum, 'params': {}}],
        [
            {'factory': stddev, 'params': {}}],
        [
            {'factory': pick, 'params': {}}],
        [
            {'factory': mask, 'params': {}}],
        [
            {'factory': count, 'params': {}}],
        [
            {'factory': string, 'params': {}}],
        [
            {'factory': mask, 'params': {'slices': [np.s_[:, :], np.s_[0:3, :]],
                                         'values': [1, 1.7]}}],
        [
            {'factory': sigsumdask, 'params': {}}],
        [
            {'factory': navsumdask, 'params': {}}],
        [
            {'factory': stddevdask, 'params': {}}],
        [
            {'factory': navsum, 'params': {}},
            {'factory': sigsum, 'params': {}}],
        [
            {'factory': sigsum, 'params': {}},
            {'factory': navsumdask, 'params': {}}],
        [
            {'factory': sigsumdask, 'params': {}},
            {'factory': navsum, 'params': {}},
            {'factory': stddevdask, 'params': {}}],
                   ),
    ids=['logsum', 'sigsum', 'navsum', 'stddev', 'pick', 'mask',
         'count', 'string', 'dualmask',
         'sigsumdask', 'navsumdask', 'stddevdask',
         'navsum+sigsum', 'sigsum+navsumdask',
         'sigsumdask+navsum+stddevdask']
)
@pytest.mark.parametrize(
    "use_roi", (True, False),
    ids=['w/ ROI', 'no ROI']
)
@pytest.mark.skipif(sys.version_info < (3, 7), reason="Requires python3.7 or higher")
def test_udfs(delayed_ctx, ds_config, udf_config, use_roi):
    ds_dict = get_dataset(delayed_ctx, **ds_config, use_roi=use_roi)
    udf_dict = build_udf_dict(udf_config, ds_dict)
    if 'roi' in udf_dict.keys():
        ds_dict['roi'] = udf_dict['roi']

    dataset = ds_dict['dataset']
    udf = udf_dict['udf']
    roi_array = ds_dict['roi']
    if not use_roi and 'roi' not in udf_dict.keys():
        assert roi_array is None
    result_dask = delayed_ctx.run_udf(dataset=dataset, udf=udf, roi=roi_array)

    alldask(result_dask)
    computed_results = delayed_ctx.executor.compute(result_dask)
    for udf_computed_results, naive_results in zip(computed_results, udf_dict['naive_result']):
        for k, result in udf_computed_results.items():
            direct_result = naive_results[k]
            if direct_result is None:
                continue
            allclose_with_nan(result, direct_result, tol=udf_dict.get('tolerance', None))


def test_run_wrap(delayed_ctx):
    wrapped_delayed = delayed_ctx.executor.run_wrap(lambda x: x**2, 4)
    assert wrapped_delayed.compute() == 16


def test_map(delayed_ctx):
    iterable = range(5)
    wrapped_delayed = delayed_ctx.executor.map(lambda x: x**2, iterable)
    assert np.allclose(wrapped_delayed, np.asarray(iterable)**2)


@pytest.mark.skipif(sys.version_info < (3, 7), reason="Requires python3.7 or higher")
def test_bare_compute(delayed_ctx):
    ds_dict = get_dataset(delayed_ctx, (16, 8, 32, 32), (8, 32, 32), 4, 2)
    dataset = ds_dict['dataset']
    udf = SumSigUDF()
    result_dask = delayed_ctx.run_udf(dataset=dataset, udf=udf)
    dask_res = result_dask['intensity'].data
    # Check one chunk per partition
    assert len(dask_res.chunks[0]) == 4
    res = dask.compute(dask_res)
    res = res[0]
    assert np.allclose(res, ds_dict['data'].sum(axis=(2, 3)))


def test_unwrap_null_case(delayed_ctx):
    nest = {'a': [5, 6, 7], 'b': (1, 2, {'x': 6, 'y': 'string'})}
    unwrapped = delayed_ctx.executor.unwrap_results(nest)
    for k, v in nest.items():
        assert isinstance(v, type(unwrapped[k]))
        for _i, _v in enumerate(v):
            assert isinstance(_v, type(unwrapped[k][_i]))
            if isinstance(_v, dict):
                for __k, __v in _v.items():
                    assert unwrapped[k][_i][__k] == __v
            else:
                assert unwrapped[k][_i] == _v
