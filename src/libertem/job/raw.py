import sys
import logging

import numpy as np

from libertem.udf.base import Task
from .base import BaseJob, ResultTile
from libertem.common import Shape
from libertem.common.buffers import zeros_aligned
from libertem.io.dataset.base import TilingScheme


log = logging.getLogger(__name__)


class PickFrameJob(BaseJob):
    '''
    .. deprecated:: 0.4.0
        Use :meth:`libertem.api.Context.create_pick_analysis`, :class:`libertem.udf.raw.PickUDF`,
        :class:`libertem.udf.masks.ApplyMasksUDF`or a custom UDF (:ref:`user-defined functions`)
        as a replacement. See also :ref:`job deprecation`.
    '''
    def __init__(self, slice_, squeeze=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slice = slice_
        self._squeeze = squeeze
        assert slice_.shape.nav.dims == 1, "slice must have flat nav"

    def _make_roi(self):
        roi = np.zeros((self.dataset.shape.nav), dtype=bool)
        roi.reshape((-1,))[self._slice.get(nav_only=True)] = True
        return roi

    def get_tasks(self):
        roi = self._make_roi()
        for idx, partition in enumerate(self.dataset.get_partitions()):
            if self._slice.intersection_with(partition.slice).is_null():
                continue
            yield PickFrameTask(partition=partition, slice_=self._slice, idx=idx, roi=roi)

    def get_result_shape(self):
        if self._squeeze:
            return tuple(part for part in self._slice.shape
                         if part > 1)
        return self._slice.shape


class PickFrameTask(Task):
    def __init__(self, slice_, roi, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slice = slice_
        self._roi = roi

    def __call__(self):
        # NOTE: this is a stop-gap solution that should work until this is deprecated
        # it is not optimized for performance...
        shape = self.partition.shape
        tileshape = Shape(
            (1,) + tuple(shape.sig),  # single frames
            sig_dims=shape.sig.dims
        )
        tiling_scheme = TilingScheme.make_for_shape(
            tileshape=tileshape,
            dataset_shape=self.partition.meta.shape,
        )
        dtype = np.dtype(self.partition.dtype).newbyteorder(sys.byteorder)
        result = zeros_aligned(self._slice.shape, dtype=dtype)
        result = result.reshape((np.count_nonzero(self._roi)), -1)

        tiles = self.partition.get_tiles(
            tiling_scheme=tiling_scheme,
            dest_dtype=dtype,
            roi=self._roi,
        )

        for tile in tiles:
            result[
                tile.tile_slice.origin[0]
            ] = tile[(...,) + self._slice.get(sig_only=True)].reshape((-1,))
        return [PickFrameResultTile(data=result)]


class PickFrameResultTile(ResultTile):
    def __init__(self, data):
        self.data = data

    @property
    def dtype(self):
        return self.data.dtype

    def reduce_into_result(self, result):
        out = result.reshape(self.data.shape)
        out += self.data
        return result
