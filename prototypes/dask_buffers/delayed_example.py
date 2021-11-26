import dask.array as da
import numpy as np
from collections import namedtuple
from functools import partial

from libertem.common.shape import Shape
from libertem.common.slice import Slice
from libertem.io.dataset.base.file import File
from libertem.io.dataset.base.fileset import FileSet
from libertem.io.dataset.base.meta import DataSetMeta
from libertem.io.dataset.base.partition import BasePartition
from libertem.io.dataset.base.backend_mmap import MMapBackend
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.sum import SumUDF
import libertem.common.buffers

from dask_inplace import DaskInplaceBufferWrapper
from libertem.udf.base import UDFMeta

_buf_chunking = -1


def dask_allocate(self, lib=None):
    _z = partial(da.zeros, chunks=_buf_chunking)
    self._data = _z(self._shape, dtype=self._dtype)

libertem.common.buffers.BufferWrapper.allocate = dask_allocate

def dask_get_slice(self, slice: Slice):
    real_slice = slice.get()
    inplace_wrapper = DaskInplaceBufferWrapper(self._data)
    inplace_wrapper.set_slice(real_slice)
    return inplace_wrapper

libertem.common.buffers.BufferWrapper._get_slice = dask_get_slice


def run_for_part(udf_class, array, partition):
    global _buf_chunking
    udf = udf_class()
    _buf_chunking = array.shape[0]
    udf.init_result_buffers()
    udf.allocate_for_part(partition, roi=None)
    udf.process_tile(array)
    udf.clear_views()
    return udf.results


def run_udf(udf_class, dataset, sig_dims):
    global _buf_chunking

    ds_shape = Shape(dataset.shape, sig_dims=sig_dims)

    f = File('.', -np.inf, np.inf, dataset.dtype, ds_shape.sig)
    fs = FileSet([f])
    ds_meta = DataSetMeta(ds_shape,
                          image_count=np.prod(ds_shape.nav, dtype=np.int32),
                          raw_dtype=dataset.dtype)

    udf_meta = UDFMeta(partition_slice=None,
                       dataset_shape=ds_shape,
                       roi=None,
                       dataset_dtype=dataset.dtype,
                       input_dtype=dataset.dtype)

    merge_udf = udf_class()
    merge_udf.set_meta(udf_meta)
    _buf_chunking = (dataset.shape[1],) * dataset.shape[0]
    merge_udf.init_result_buffers()
    FakeDS = namedtuple('FakeDS', ['shape'])
    merge_udf.allocate_for_full(FakeDS(shape=ds_shape), roi=None)

    slice_boundaries = (0,) + tuple(np.cumsum(dataset.chunks[0], dtype=int))
    slice_rest = dataset.shape[1:]
    for s0, s1 in zip(slice_boundaries[:-1], slice_boundaries[1:]):
        flat_shape = Shape(((s1 - s0) * slice_rest[0],) + slice_rest[1:], sig_dims=sig_dims)
        flat_origin = (s0 * slice_rest[0], 0, 0)
        part_slice = Slice(flat_origin, flat_shape)
        dask_chunk = dataset[s0:s1, ...].squeeze(axis=0)

        start_frame = part_slice.origin[0]
        num_frames = part_slice.shape[0]
        partition = BasePartition(ds_meta, part_slice, fs, start_frame, num_frames, MMapBackend)
        part_results = run_for_part(udf_class, dask_chunk, partition)

        merge_udf.set_views_for_partition(partition)
        merge_udf.merge(
            dest=merge_udf.results.get_proxy(),
            src=part_results.get_proxy()
        )
        merge_udf.clear_views()

    return merge_udf._do_get_results()


def build_increasing_ds(data, axis, mode='arange'):
    ds_shape = data.shape
    multishape = tuple(v if idx == axis else 1 for idx, v in enumerate(ds_shape))
    if mode == 'arange':
        multi = np.arange(ds_shape[axis])
    elif mode == 'linspace':
        multi = np.linspace(0., 1., num=ds_shape[axis], endpoint=True)
    else:
        raise
    return data * multi.reshape(multishape)


if __name__ == '__main__':
    ds_shape = Shape((5, 10, 64, 64), sig_dims=2)
    data = np.ones(tuple(ds_shape))
    for i, mode in enumerate(['arange'] * 2 + ['linspace'] * 2):
        data = build_increasing_ds(data, i, mode=mode)
    dar = da.from_array(data, chunks=(1, -1, -1, -1))
    
    _buf_chunking = (dar.shape[1],) * dar.shape[0]
    results = run_udf(SumSigUDF, dar, sig_dims=len(ds_shape.sig))
