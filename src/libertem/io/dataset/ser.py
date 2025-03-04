import os
import logging
from typing import Optional
import warnings
import contextlib

import numpy as np
from ncempy.io.ser import fileSER
from sparseconverter import CUDA, NUMPY, ArrayBackend

from libertem.common.math import prod, flat_nonzero
from libertem.common import Shape, Slice
from libertem.io.dataset.base.tiling_scheme import TilingScheme
from libertem.common.messageconverter import MessageConverter
from .base import (
    DataSet, FileSet, BasePartition, DataSetException, DataSetMeta,
    DataTile,
)

log = logging.getLogger(__name__)


class SERDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/SERDatasetParams.schema.json",
        "title": "SERDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "SER"},
            "path": {"type": "string"},
            "nav_shape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
                "maxItems": 2
            },
            "sig_shape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
                "maxItems": 2
            },
            "sync_offset": {"type": "number"},
        },
        "required": ["type", "path"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path"]
        }
        if "nav_shape" in raw_data:
            data["nav_shape"] = tuple(raw_data["nav_shape"])
        if "sig_shape" in raw_data:
            data["sig_shape"] = tuple(raw_data["sig_shape"])
        if "sync_offset" in raw_data:
            data["sync_offset"] = raw_data["sync_offset"]
        return data


class SERFile:
    def __init__(self, path, num_frames):
        self._path = path
        self._num_frames = num_frames

    def _get_handle(self):
        return fileSER(self._path)

    @contextlib.contextmanager
    def get_handle(self):
        with self._get_handle() as f:
            yield f

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def start_idx(self):
        return 0

    @property
    def end_idx(self):
        return self.num_frames


class SERFileSet(FileSet):
    pass


class SERDataSet(DataSet):
    """
    Read TIA SER files.

    Examples
    --------

    >>> ds = ctx.load("ser", path="/path/to/file.ser")  # doctest: +SKIP

    Parameters
    ----------
    path: str
        Path to the .ser file

    nav_shape: tuple of int, optional
        A n-tuple that specifies the size of the navigation region ((y, x), but
        can also be of length 1 for example for a line scan, or length 3 for
        a data cube, for example)

    sig_shape: tuple of int, optional
        Signal/detector size (height, width)

    sync_offset: int, optional
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start

    num_partitions: int, optional
        Override the number of partitions. This is useful if the
        default number of partitions, chosen based on common workloads,
        creates partitions which are too large (or small) for the UDFs
        being run on this dataset.
    """
    def __init__(
        self,
        path,
        emipath=None,
        nav_shape=None,
        sig_shape=None,
        sync_offset=0,
        io_backend=None,
        num_partitions=None,
    ):
        super().__init__(
            io_backend=io_backend,
            num_partitions=num_partitions,
        )
        if io_backend is not None:
            raise ValueError("SERDataSet currently doesn't support alternative I/O backends")
        self._path = path
        self._meta = None
        self._filesize = None
        self._num_frames = None
        if emipath is not None:
            warnings.warn(
                "emipath is not used anymore, as it was removed from ncempy", DeprecationWarning
            )
        self._nav_shape = tuple(nav_shape) if nav_shape else nav_shape
        self._sig_shape = tuple(sig_shape) if sig_shape else sig_shape
        self._sync_offset = sync_offset

    def _do_initialize(self):
        self._filesize = os.stat(self._path).st_size
        reader = SERFile(path=self._path, num_frames=None)

        with reader.get_handle() as f1:
            self._num_frames = f1.head['ValidNumberElements']
            if f1.head['ValidNumberElements'] == 0:
                raise DataSetException("no data found in file")

            data, meta_data = f1.getDataset(0)
            dtype = f1._dictDataType[meta_data['DataType']]
            nav_dims = tuple(
                reversed([
                    int(dim['DimensionSize'])
                    for dim in f1.head['Dimensions']
                ])
            )
            self._image_count = int(self._num_frames)
            if self._nav_shape is None:
                self._nav_shape = nav_dims
            if self._sig_shape is None:
                self._sig_shape = tuple(data.shape)
            elif int(prod(self._sig_shape)) != int(prod(data.shape)):
                raise DataSetException(
                    "sig_shape must be of size: %s" % int(prod(data.shape))
                )
            self._nav_shape_product = int(prod(self._nav_shape))
            self._sync_offset_info = self.get_sync_offset_info()
            self._shape = Shape(self._nav_shape + self._sig_shape, sig_dims=len(self._sig_shape))
            self._meta = DataSetMeta(
                shape=self._shape,
                raw_dtype=dtype,
                sync_offset=self._sync_offset,
                image_count=self._image_count,
            )
        return self

    def initialize(self, executor):
        return executor.run_function(self._do_initialize)

    @classmethod
    def get_msg_converter(cls):
        return SERDatasetParams

    @classmethod
    def get_supported_extensions(cls):
        return {"ser"}

    @classmethod
    def get_supported_io_backends(self):
        return []

    @classmethod
    def detect_params(cls, path, executor):
        if path.lower().endswith(".ser"):
            ds = cls(path)
            ds = ds.initialize(executor)
            return {
                "parameters": {
                    "path": path,
                    "nav_shape": tuple(ds.shape.nav),
                    "sig_shape": tuple(ds.shape.sig),
                },
                "info": {
                    "image_count": int(prod(ds.shape.nav)),
                    "native_sig_shape": tuple(ds.shape.sig),
                }
            }
        return False

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    def check_valid(self):
        try:
            with fileSER(self._path) as f1:
                if f1.head['ValidNumberElements'] == 0:
                    raise DataSetException("no data found in file")
                if f1.head['DataTypeID'] not in (0x4120, 0x4122):
                    raise DataSetException("unknown datatype id: %s" % f1.head['DataTypeID'])
            return True
        except OSError as e:
            raise DataSetException("invalid dataset: %s" % e) from e

    def get_cache_key(self):
        return {
            "path": self._path,
            "shape": tuple(self.shape),
            "sync_offset": self._sync_offset,
        }

    def _get_fileset(self):
        assert self._num_frames is not None
        return SERFileSet([
            SERFile(
                path=self._path,
                num_frames=self._num_frames,
            )
        ])

    def get_base_shape(self, roi):
        return (1,) + tuple(self.shape.sig)

    def adjust_tileshape(self, tileshape, roi):
        # force single-frame tiles
        return (1,) + tileshape[1:]

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in self.get_slices():
            yield SERPartition(
                path=self._path,
                meta=self._meta,
                partition_slice=part_slice,
                fileset=fileset,
                start_frame=start,
                num_frames=stop - start,
                io_backend=self.get_io_backend(),
                decoder=None,
            )

    def __repr__(self):
        return f"<SERDataSet for {self._path}>"


class SERPartition(BasePartition):
    def __init__(self, path, *args, **kwargs):
        self._path = path
        super().__init__(*args, **kwargs)

    def validate_tiling_scheme(self, tiling_scheme):
        if tiling_scheme.shape.sig != self.shape.sig:
            raise ValueError(
                f"invalid tiling scheme ({tiling_scheme.shape!r}): sig shape must match"
            )

    def _preprocess(self, tile_data, tile_slice):
        if self._corrections is None:
            return
        self._corrections.apply(tile_data, tile_slice)

    def get_tiles(self, tiling_scheme: TilingScheme, dest_dtype="float32", roi=None,
            array_backend: Optional[ArrayBackend] = None):
        if array_backend is None:
            array_backend = self.meta.array_backends[0]
        assert array_backend in (NUMPY, CUDA)
        sync_offset = self.meta.sync_offset
        shape = Shape((1,) + tuple(self.shape.sig), sig_dims=self.shape.sig.dims)
        tiling_scheme = tiling_scheme.adjust_for_partition(self)
        self.validate_tiling_scheme(tiling_scheme)

        start = self._start_frame
        if start < self.meta.image_count:
            stop = min(start + self._num_frames, self.meta.image_count)
            if roi is None:
                indices = np.arange(max(0, start), stop)
                # in case of a negative sync_offset, 'start' can be negative
                if start < 0:
                    offset = abs(sync_offset)
                else:
                    offset = start - sync_offset
            else:
                # The following is taken (effectively) from _default_get_read_ranges
                roi_nonzero = flat_nonzero(roi)

                shifted_roi = roi_nonzero + sync_offset
                roi_mask = np.logical_and(shifted_roi >= max(0, start),
                                          shifted_roi < stop)
                indices = shifted_roi[roi_mask]

                # in case of a negative sync_offset, 'start' can be negative
                if start < 0:
                    offset = np.sum(roi_nonzero < abs(sync_offset))
                else:
                    offset = np.sum(roi_nonzero < start - sync_offset)

            with fileSER(self._path) as f:
                for num, idx in enumerate(indices):
                    origin = (num + offset,) + tuple([0] * self.shape.sig.dims)
                    tile_slice = Slice(origin=origin, shape=shape)
                    data, metadata = f.getDataset(int(idx))
                    if data.dtype != np.dtype(dest_dtype):
                        data = data.astype(dest_dtype)
                    data = data.reshape(shape)
                    self._preprocess(data, tile_slice)
                    yield DataTile(
                        data,
                        tile_slice=tile_slice,
                        scheme_idx=0,
                    )
