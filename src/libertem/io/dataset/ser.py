import os
import math
import logging
import itertools

import numpy as np
from ncempy.io.ser import fileSER

from libertem.common import Slice, Shape
from .base import DataSet, Partition, DataTile, DataSetException, DataSetMeta

log = logging.getLogger(__name__)


class SERReader(object):
    def __init__(self, meta, path, emipath=None):
        self._path = path
        self._emipath = emipath
        self._meta = meta

    def _get_handle(self):
        return fileSER(self._path, emifile=self._emipath)

    def read_images(self, start, stop, out, crop_to=None):
        """
        read [`start`, `stop`) images from this file into `out`
        """
        with self._get_handle() as f1:
            num_images = f1.head['ValidNumberElements']
            assert start < num_images
            assert stop <= num_images
            assert stop >= start
            for ii in range(start, stop):
                data0, metadata0 = f1.getDataset(ii)
                if crop_to is not None:
                    # TODO: maybe limit I/O to the cropped region, too?
                    data0 = data0[crop_to.get(sig_only=True)]
                out[ii - start, ...] = data0


class SERDataSet(DataSet):
    def __init__(self, path, emipath=None):
        self._path = path
        self._emipath = emipath
        self._meta = None
        self._filesize = None

    def initialize(self):
        self._filesize = os.stat(self._path).st_size
        with fileSER(self._path, emifile=self._emipath) as f1:
            if f1.head['ValidNumberElements'] == 0:
                raise DataSetException("no data found in file")

            data, meta_data = f1.getDataset(0)
            dtype = f1._dictDataType[meta_data['DataType']]
            if f1.head['DataTypeID'] == 0x4120:
                # Spectra as 1D single spectra, 2D line scan or 3D spectrum image
                spectra_size = data.shape[0]
                shape = (f1.head['ValidNumberElements'], spectra_size)
                raw_shape = shape
                if f1.head['NumberDimensions'] > 1:
                    scan_x = f1.head['Dimensions'][0]['DimensionSize']
                    scan_y = f1.head['Dimensions'][1]['DimensionSize']
                    shape = (scan_x, scan_y, spectra_size)

                self._meta = DataSetMeta(
                    shape=Shape(shape, sig_dims=1),
                    raw_shape=Shape(raw_shape, sig_dims=1),
                    dtype=dtype
                )
            elif f1.head['DataTypeID'] == 0x4122:
                shape = (f1.head['ValidNumberElements'],) + tuple(data.shape)
                sig_dims = len(data.shape)
                self._meta = DataSetMeta(
                    shape=Shape(shape, sig_dims=sig_dims),
                    raw_shape=Shape(shape, sig_dims=sig_dims),
                    dtype=dtype
                )
            else:
                raise DataSetException("unknown DataTypeID: %s" % f1.head['DataTypeID'])
        return self

    @classmethod
    def detect_params(cls, path):
        raise NotImplementedError()

    @property
    def dtype(self):
        return self._meta.dtype

    @property
    def shape(self):
        return self._meta.shape

    @property
    def raw_shape(self):
        return self._meta.raw_shape

    def check_valid(self):
        try:
            with fileSER(self._path, emifile=self._emipath) as f1:
                if f1.head['ValidNumberElements'] == 0:
                    raise DataSetException("no data found in file")
                if f1.head['DataTypeID'] not in (0x4120, 0x4122):
                    raise DataSetException("unknown datatype id: %s" % f1.head['DataTypeID'])
            return True
        except (IOError, OSError) as e:
            raise DataSetException("invalid dataset: %s" % e) from e

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 512MB per partition
        res = max(1, self._filesize // (512*1024*1024))
        return res

    def get_partitions(self):
        num_frames = self.shape.nav.size
        f_per_part = num_frames // self._get_num_partitions()

        c0 = itertools.count(start=0, step=f_per_part)
        c1 = itertools.count(start=f_per_part, step=f_per_part)
        for (start, stop) in zip(c0, c1):
            if start >= num_frames:
                break
            stop = min(stop, num_frames)
            part_slice = Slice(
                origin=(
                    start, 0, 0,
                ),
                shape=Shape(((stop - start),) + tuple(self.shape.sig),
                            sig_dims=self.shape.sig.dims)
            )
            yield SERPartition(
                meta=self._meta,
                partition_slice=part_slice,
                reader=SERReader(path=self._path, emipath=self._emipath, meta=self._meta),
                start_frame=start,
                num_frames=stop - start,
            )

    def __repr__(self):
        return "<SERDataSet for %s>" % (self._path,)


class SERPartition(Partition):
    def __init__(self, reader, start_frame, num_frames, *args, **kwargs):
        self._reader = reader
        self._start_frame = start_frame
        self._num_frames = num_frames
        super().__init__(*args, **kwargs)

    def _get_stackheight(self, target_size=2 * 1024 * 1024):
        # FIXME: centralize this decision and make it tunable
        framesize = self.meta.shape.sig.size * self.dtype.itemsize
        return min(1, math.floor(target_size / framesize))

    def get_tiles(self, crop_to=None):
        start_at_frame = self._start_frame
        num_frames = self._num_frames
        stackheight = self._get_stackheight()
        dtype = self.dtype
        sig_shape = self.meta.shape.sig
        sig_origin = tuple([0] * len(sig_shape))
        if crop_to is not None:
            sig_origin = tuple(crop_to.origin[-sig_shape.dims:])
            sig_shape = crop_to.shape.sig
        tile_buf_full = np.zeros((stackheight,) + tuple(sig_shape), dtype=dtype)

        tileshape = (
            stackheight,
        ) + tuple(sig_shape)

        for outer_frame in range(start_at_frame, start_at_frame + num_frames, stackheight):
            if start_at_frame + num_frames - outer_frame < stackheight:
                end_frame = start_at_frame + num_frames
                current_stackheight = end_frame - outer_frame
                current_tileshape = (
                    current_stackheight,
                ) + tuple(sig_shape)
                tile_buf = np.zeros(current_tileshape, dtype=dtype)
            else:
                current_stackheight = stackheight
                current_tileshape = tileshape
                tile_buf = tile_buf_full
            tile_slice = Slice(
                origin=(outer_frame,) + sig_origin,
                shape=Shape(current_tileshape, sig_dims=sig_shape.dims)
            )
            if crop_to is not None:
                intersection = tile_slice.intersection_with(crop_to)
                if intersection.is_null():
                    continue
            self._reader.read_images(
                start=outer_frame,
                stop=outer_frame + current_stackheight,
                out=tile_buf,
                crop_to=crop_to,
            )
            yield DataTile(
                data=tile_buf,
                tile_slice=tile_slice
            )
