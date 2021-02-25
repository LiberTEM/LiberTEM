from typing import List, Generator, Tuple, Union

import numpy as np

from libertem.common import Shape, Slice
from libertem.executor.base import JobExecutor
from libertem.corrections import CorrectionSet
from libertem.io.dataset.base import (
    DataSet, Partition, DataSetMeta, DataTile, TilingScheme,
)


class RoiDataSet(DataSet):
    def __init__(self, wrapped: DataSet, rois: List[np.ndarray], io_backend=None):
        self._wrapped = wrapped
        self._rois = rois
        our_nav_shape = sum(
            roi.sum()
            for roi in rois
        )
        shape = Shape(
            (our_nav_shape,) + wrapped.shape.sig,
            sig_dims=wrapped.shape.sig.dims,
        )
        self._meta = DataSetMeta(
            shape=shape,
            raw_dtype=wrapped.meta.raw_dtype,
            sync_offset=0,
            image_count=wrapped.meta.image_count,
        )
        if io_backend is not None:
            raise ValueError(
                "RoiDataSet always uses the io_backend of the "
                "wrapped DataSet."
            )

    def initialze(self, executor: JobExecutor):
        # Most likely nothing to do? self._wrapped should already be initialzed
        pass

    def _roi_to_range(self, roi: np.ndarray) -> Tuple[int, int]:
        """
        Convert a ROI array to [start, stop) intervals
        on the flat navigation axis.
        """
        flat_roi = roi.reshape((-1,))
        coords = np.nonzero(flat_roi)
        return (
            np.min(coords),
            np.max(coords) + 1,
        )

    def get_partitions(self, ranges=None) -> Generator["RoiPartition", None, None]:
        if ranges is not None:
            raise ValueError(
                "cannot use pre-defined ranges for this DataSet"
            )
        ranges = [
            self._roi_to_range(roi)
            for roi in self._rois
        ]
        wrapped_partitions = self._wrapped.get_partitions(ranges=ranges)
        wrapped_partitions = list(wrapped_partitions)
        # FIXME: create sub-partitions from the ROIs?
        # They can possibly be too large otherwise...
        # This is currently the responsibility of the user
        start = 0
        for part_range, roi, part in zip(ranges, self._rois, wrapped_partitions):
            shape = self.shape
            # part_slice is in flattened coordinates of
            # the resulting `RoiDataSet`
            part_slice = Slice(
                origin=(start,) + tuple([0] * shape.sig.dims),
                shape=Shape((roi.sum(),) + tuple(shape.sig),
                            sig_dims=shape.sig.dims)
            )
            yield RoiPartition(
                meta=self._wrapped.meta,
                wrapped_partition=part,
                roi=roi,
                partition_slice=part_slice,
            )
            start += roi.sum()

    @property
    def raw_dtype(self):
        """
        the underlying data type
        """
        return self._wrapped.raw_dtype

    @property
    def dtype(self):
        return self._wrapped.dtype

    @property
    def shape(self):
        # flattened shape:
        return self.meta.shape

    @property
    def meta(self):
        return self._meta

    def check_valid(self):
        return True

    @classmethod
    def detect_params(cls, path, executor):
        # as this is a "virtual" DataSet, it can't be "detected" from a path...
        pass


class RoiPartition(Partition):
    """
    """
    def __init__(
        self,
        meta: DataSetMeta,
        wrapped_partition: Partition,
        roi: np.ndarray,
        partition_slice: Slice,
    ):
        super().__init__(meta=meta, io_backend=None)
        self._wrapped = wrapped_partition
        self._roi = roi
        self.slice = partition_slice

    @property
    def shape(self):
        return self.slice.shape.flatten_nav()

    def shape_for_roi(self, roi: Union[np.ndarray, None]):
        return self.slice.adjust_for_roi(roi).shape

    def need_decode(self, read_dtype, roi, corrections):
        # FIXME: is there any case where we don't need to decode?
        return True

    def adjust_tileshape(self, tileshape):
        return self._wrapped.adjust_tileshape(tileshape)

    def set_corrections(self, corrections: CorrectionSet):
        self._wrapped.set_corrections(corrections)
        self._corrections = corrections

    def get_base_shape(self):
        return self._wrapped.get_base_shape()

    def get_io_backend(self):
        return self._wrapped.get_io_backend()

    def get_locations(self):
        return self._wrapped.get_locations()

    def get_macrotile(self, dest_dtype="float32", roi=None) -> DataTile:
        '''
        Return a single tile for the entire partition.

        This is useful to support process_partiton() in UDFs and to construct dask arrays
        from datasets.
        '''

        tiling_scheme = TilingScheme.make_for_shape(
            tileshape=self.shape,
            dataset_shape=self.meta.shape,
        )

        try:
            return next(self.get_tiles(
                tiling_scheme=tiling_scheme,
                dest_dtype=dest_dtype,
                roi=roi,
            ))
        except StopIteration:
            sig_dims = self.shape.sig.dims
            tile_slice = Slice(
                origin=(self.slice.origin[0],) + tuple([0] * sig_dims),
                shape=Shape(
                    (0,) + tuple(self.shape.sig),
                    sig_dims=sig_dims,
                ),
            )
            return DataTile(
                np.zeros(tile_slice.shape, dtype=dest_dtype),
                tile_slice=tile_slice,
                scheme_idx=0,
            )

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None):
        if roi is not None:
            # TODO
            raise ValueError("can't combine ROIs yet,")
        offset = self.slice.origin
        for tile in self._wrapped.get_tiles(
            tiling_scheme=tiling_scheme,
            dest_dtype=dest_dtype,
            roi=self._roi,
        ):
            tile.tile_slice = tile.tile_slice.translate(offset)
            yield tile

    def __repr__(self):
        start = self.slice.origin[0]
        num_frames = self.slice.shape[0]
        return "<%s [%d:%d]>" % (
            self.__class__.__name__,
            start, start + num_frames
        )
