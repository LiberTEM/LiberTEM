from typing import List, Generator, Tuple, Union

import numpy as np

from libertem.common import Shape, Slice
from libertem.executor.base import JobExecutor
from libertem.io.dataset.base import (
    DataSet, Partition, DataSetMeta, DataTile, TilingScheme,
)


class RoiDataSet(DataSet):
    def __init__(self, wrapped: DataSet, rois: List[np.ndarray], io_backend=None):
        self._wrapped = wrapped
        self._rois = rois
        self._io_backend = io_backend

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
        # they can possibly be too large otherwise...
        for roi in self._rois:
            yield RoiPartition(
                meta=self._wrapped.meta,
                wrapped_partitions=wrapped_partitions,
                roi=roi,
                io_backend=self.get_io_backend(),
            )

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
        return self._wrapped.shape

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
        self, meta: DataSetMeta, wrapped_partitions: List[Partition], roi: np.ndarray, io_backend
    ):
        super().__init__(meta=meta, io_backend=io_backend)
        self._wrapped = wrapped_partitions
        self._roi = roi

    @property
    def shape(self):
        sig_shape = self.meta.shape.sig
        return Shape(
            (np.count_nonzero(self._roi),) + tuple(sig_shape),
            sig_dims=sig_shape.dims,
        )

    def shape_for_roi(self, roi: Union[np.ndarray, None]):
        return self.slice.adjust_for_roi(roi).shape

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
        for partition in self._wrapped:
            yield from partition.get_tiles(
                tiling_scheme=tiling_scheme,
                dest_dtype=dest_dtype,
                roi=roi,
            )
