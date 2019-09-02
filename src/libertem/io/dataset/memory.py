import time
import logging

import psutil
import numpy as np

from libertem.web.messages import MessageConverter
from libertem.io.dataset.base import (
    FileSet3D, Partition3D, DataSet, DataSetMeta, DataTile
)
from libertem.common import Shape


log = logging.getLogger(__name__)


class MemDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/MEMDatasetParams.schema.json",
        "title": "MEMDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "memory"},
            "tileshape": {
                "type": "array",
                "items": {"type": "number"},
            },
            "datashape": {
                "type": "array",
                "items": {"type": "number"},
            },
            "num_partitions": {"type": "number"},
            "sig_dims": {"type": "number"},
            "check_cast": {"type": "boolean"},
            "crop_frames": {"type": "boolean"},
            "tiledelay": {"type": "number"},
        },
        "required": ["type", "tileshape", "num_partitions"],
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["tileshape", "num_partitions", "sig_dims", "check_cast",
                      "crop_frames", "tiledelay", "datashape"]
            if k in raw_data
        }
        return data


class MemoryFile3D(object):
    def __init__(self, data, check_cast=True):
        self.num_frames = data.shape[0]
        self.start_idx = 0
        self.end_idx = self.num_frames
        self._data = data
        self._check_cast = check_cast

    def open(self):
        pass

    def close(self):
        pass

    def readinto(self, start, stop, out, crop_to=None):
        slice_ = (...,)
        if crop_to is not None:
            slice_ = crop_to.get(sig_only=True)
        if self._check_cast:
            assert np.can_cast(self._data.dtype, out.dtype, casting='safe'),\
                "cannot cast safely between %s and %s" % (self._data.dtype, out.dtype)
        out[:] = self._data[(slice(start, stop),) + slice_]


class MemoryDataSet(DataSet):
    def __init__(self, tileshape=None, num_partitions=None, data=None, sig_dims=2,
                 check_cast=True, crop_frames=False, tiledelay=None, datashape=None):
        # For HTTP API testing purposes: Allow to create empty dataset with given shape
        if data is None:
            assert datashape is not None
            data = np.zeros(datashape, dtype=np.float32)
        if num_partitions is None:
            num_partitions = psutil.cpu_count(logical=False)
        if tileshape is None:
            sig_shape = data.shape[-sig_dims:]
            target = 2**20
            framesize = np.prod(sig_shape)
            framecount = max(1, min(np.prod(data.shape[:-sig_dims]), int(target / framesize)))
            tileshape = (framecount, ) + sig_shape
        assert len(tileshape) == sig_dims + 1
        self.data = data
        self.tileshape = Shape(tileshape, sig_dims=sig_dims)
        self.num_partitions = num_partitions
        self.sig_dims = sig_dims
        self._meta = DataSetMeta(
            shape=self.shape,
            raw_dtype=self.data.dtype,
        )
        self._check_cast = check_cast
        self._crop_frames = crop_frames
        self._tiledelay = tiledelay

    def initialize(self):
        return self

    @classmethod
    def get_msg_converter(cls):
        return MemDatasetParams

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return Shape(self.data.shape, sig_dims=self.sig_dims)

    def check_valid(self):
        return True

    def get_partitions(self):
        fileset = FileSet3D([
            MemoryFile3D(self.data.reshape(self.shape.flatten_nav()),
                         check_cast=self._check_cast)
        ])

        stackheight = int(np.product(self.tileshape[:-self.sig_dims]))
        for part_slice, start, stop in Partition3D.make_slices(
                shape=self.shape,
                num_partitions=self.num_partitions):
            log.debug("creating partition", part_slice, start, stop, stackheight)
            if self._crop_frames:
                yield CropFramesMemPartition(
                    meta=self._meta,
                    partition_slice=part_slice,
                    fileset=fileset.get_for_range(start, stop),
                    start_frame=start,
                    num_frames=stop - start,
                    stackheight=stackheight,
                    tileshape=self.tileshape,
                    tiledelay=self._tiledelay,
                )
            else:
                yield MemPartition(
                    meta=self._meta,
                    partition_slice=part_slice,
                    fileset=fileset.get_for_range(start, stop),
                    start_frame=start,
                    num_frames=stop - start,
                    stackheight=stackheight,
                    tiledelay=self._tiledelay,
                )


class MemPartition(Partition3D):
    def __init__(self, tiledelay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tiledelay = tiledelay

    def get_tiles(self, *args, **kwargs):
        tiles = super().get_tiles(*args, **kwargs)
        if self._tiledelay:
            log.debug("delayed get_tiles, tiledelay=%.3f" % self._tiledelay)
            for tile in tiles:
                yield tile
                time.sleep(self._tiledelay)
        else:
            yield from tiles


class CropFramesMemPartition(Partition3D):
    def __init__(self, tileshape, tiledelay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tileshape = tileshape
        self._tiledelay = tiledelay

    def get_tiles(self, *args, **kwargs):
        crop = self._tileshape[1:]
        for tile in super().get_tiles(*args, **kwargs):
            log.debug("tile with slice %r" % tile.tile_slice)
            for subslice in tile.tile_slice.subslices((self._stackheight,) + crop):
                if self._tiledelay:
                    time.sleep(self._tiledelay)
                yield DataTile(
                    data=subslice.shift(tile.tile_slice).get(tile.data),
                    tile_slice=subslice,
                )
