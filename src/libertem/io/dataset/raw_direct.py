import contextlib
import itertools
import os

import numpy as np

from libertem.common import Slice, Shape
from .base import DataSet, Partition, DataTile, DataSetException, DataSetMeta
from libertem.io.direct import open_direct, empty_aligned, readinto_direct


class DirectRawFileReader(object):
    def __init__(self, meta, path):
        self._path = path
        self._meta = meta
        self._file = None

    @contextlib.contextmanager
    def open_file(self):
        with open_direct(self._path) as f:
            self._file = f
            yield self
        self._file = None

    def _calc_offset(self, idx):
        frame_size_px = self._meta.shape.sig.size
        frame_size = frame_size_px * self._meta.dtype.itemsize
        offset = idx * frame_size
        return offset

    def get_buffer(self, stackheight):
        frame_size_px = self._meta.shape.sig.size
        size = frame_size_px * stackheight
        return empty_aligned(size, dtype=self._meta.dtype)

    def readinto(self, out):
        return readinto_direct(self._file, out)

    def seek_frame(self, idx):
        self._file.seek(self._calc_offset(idx))


class DirectRawFileDataSet(DataSet):
    def __init__(self, path, scan_size, dtype, detector_size, stackheight):
        self._path = path
        self._scan_size = tuple(scan_size)
        self._detector_size = detector_size
        self._stackheight = stackheight
        self._sig_dims = len(self._detector_size)
        shape = Shape(self._scan_size + self._detector_size, sig_dims=self._sig_dims)
        self._meta = DataSetMeta(
            shape=shape,
            raw_shape=Shape((shape.nav.size,) + self._detector_size, sig_dims=self._sig_dims),
            dtype=np.dtype(dtype)
        )
        self._filesize = None

    def initialize(self):
        self._filesize = os.stat(self._path).st_size
        return self

    @property
    def dtype(self):
        return self._meta.dtype

    @property
    def shape(self):
        return self._meta.shape

    @property
    def raw_shape(self):
        return self._meta.raw_shape

    def get_reader(self):
        return DirectRawFileReader(
            meta=self._meta,
            path=self._path,
        )

    def check_valid(self):
        try:
            reader = self.get_reader()
            reader.open_file()
            # TODO: check file size match
            # TODO: try to read from file?
            return True
        except (IOError, OSError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 1024MB per partition
        res = max(1, self._filesize // (1024*1024*1024))
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
            yield DirectRawFilePartition(
                stackheight=self._stackheight,
                meta=self._meta,
                reader=self.get_reader(),
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
            )

    def __repr__(self):
        return "<DirectRawFileDataSet of %s shape=%s>" % (self.dtype, self.shape)


class DirectRawFilePartition(Partition):
    def __init__(self, stackheight, start_frame, num_frames, reader, *args, **kwargs):
        self.stackheight = stackheight
        self.reader = reader
        self.start_frame = start_frame
        self.num_frames = num_frames
        super().__init__(*args, **kwargs)

    def get_tiles(self, crop_to=None):
        if crop_to is not None:
            if crop_to.shape.sig != self.meta.shape.sig:
                raise DataSetException(
                    "DirectRawFileDataSet only supports whole-frame crops for now"
                )
        stackheight = self.stackheight
        start_frame = self.start_frame
        num_frames = self.num_frames
        shape_sig = tuple(self.shape.sig)
        sig_dims = self.shape.sig.dims
        sig_size = self.shape.sig.size
        stop = start_frame + num_frames
        with self.reader.open_file() as reader:
            buf = reader.get_buffer(stackheight)
            c0 = itertools.count(start=start_frame, step=stackheight)
            for tile_start in c0:
                if tile_start >= stop:
                    break
                reader.seek_frame(tile_start)
                # tile_height is the real height, which may be smaller than stackheight
                # at the end
                tile_height = min(stackheight, stop - tile_start)
                tileslice = Slice(
                    origin=(
                        tile_start, 0, 0,
                    ),
                    shape=Shape((tile_height,) + shape_sig,
                                sig_dims=sig_dims)
                )
                if crop_to is not None:
                    intersection = tileslice.intersection_with(crop_to)
                    if intersection.is_null():
                        continue
                data = reader.readinto(buf)
                # at partition boundary we may read more than requested, cut it off:
                data = data[:tile_height * sig_size]
                yield DataTile(
                    data=data.reshape((tile_height,) + shape_sig),
                    tile_slice=tileslice
                )
