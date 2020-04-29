from typing import List

import numpy as np

from libertem.io.dataset.base import DataSet, Partition, DataSetMeta


class RoiDataSet(DataSet):
    def __init__(self, wrapped: DataSet, rois: List[np.ndarray]):
        self._wrapped = wrapped
        self._rois = rois

    def initialze(self, executor):
        # Most likely nothing to do? self._wrapped should already be initialzed
        pass

    def get_partitions(self):
        wrapped_partitions = list(self._wrapped.get_partitions())
        # FIXME: create sub-partitions from the ROIs?
        # they can possibly be too large otherwise...
        for roi in self._rois:
            yield RoiPartition(
                meta=self._wrapped.meta,
                wrapped_partitions=wrapped_partitions,
                roi=roi,
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
    def __init__(self, meta: DataSetMeta, wrapped_partitions: List[Partition], roi: np.ndarray):
        super().__init__(meta=meta)
        self._wrapped = wrapped_partitions
        self._roi = roi

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
