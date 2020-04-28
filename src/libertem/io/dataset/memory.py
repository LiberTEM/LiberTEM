import time
import logging

import psutil
import numpy as np

from libertem.web.messages import MessageConverter
from libertem.io.dataset.base import (
    FileSet, BasePartition, DataSet, DataSetMeta, TilingScheme,
    LocalFile, LocalFSMMapBackend,
)
from libertem.common import Shape, Slice
from libertem.io.dataset.base import DataTile


log = logging.getLogger(__name__)


class MemBackend(LocalFSMMapBackend):
    def _set_readahead_hints(self, roi, fileset):
        pass

    def _get_tiles_roi(self, tiling_scheme, fileset, read_ranges, roi):
        ds_sig_shape = tiling_scheme.dataset_shape.sig
        sig_dims = tiling_scheme.shape.sig.dims
        slices, ranges, scheme_indices = read_ranges

        fh = fileset[0]
        memmap = fh.mmap().reshape((fh.num_frames,) + tuple(ds_sig_shape))
        data_w_roi = memmap[roi.reshape((-1,))]

        for idx in range(slices.shape[0]):
            origin, shape = slices[idx]
            scheme_idx = scheme_indices[idx]

            tile_slice = Slice(
                origin=origin,
                shape=Shape(shape, sig_dims=sig_dims)
            )
            data_slice = tile_slice.get()
            data = data_w_roi[data_slice]
            yield DataTile(
                data,
                tile_slice=tile_slice,
                scheme_idx=scheme_idx,
            )

    def get_tiles(self, tiling_scheme, fileset, read_ranges, roi, native_dtype, read_dtype):
        if roi is None:
            # support arbitrary tiling in case of no roi
            with fileset:
                for tile in self._get_tiles_straight(tiling_scheme, fileset, read_ranges):
                    yield tile.astype(read_dtype)
        else:
            with fileset:
                for tile in self._get_tiles_roi(tiling_scheme, fileset, read_ranges, roi):
                    yield tile.astype(read_dtype)


class MemDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/MEMDatasetParams.schema.json",
        "title": "MEMDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "MEMORY"},
            "tileshape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
            },
            "datashape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
            },
            "num_partitions": {"type": "number", "minimum": 1},
            "sig_dims": {"type": "number", "minimum": 1},
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


class MemoryFile(LocalFile):
    def __init__(self, data, check_cast=True, *args, **kwargs):
        self._data = data
        self._check_cast = check_cast
        super().__init__(*args, **kwargs)

    def open(self):
        pass

    def close(self):
        pass

    def mmap(self):
        return self._data

    def raw_mmap(self):
        return self._data.view(np.uint8)

    def fileno(self):
        raise NotImplementedError()


class MemoryDataSet(DataSet):
    '''
    This dataset is constructed from a NumPy array in memory for testing
    purposes. It is not recommended for production use since it performs poorly with a
    distributed executor.

    Examples
    --------

    >>> from libertem.io.dataset.memory import MemoryDataSet
    >>>
    >>> data = np.zeros((2, 2, 128, 128))
    >>> ds = MemoryDataSet(data=data)
    '''
    def __init__(self, tileshape=None, num_partitions=None, data=None, sig_dims=2,
                 check_cast=True, tiledelay=None, datashape=None, base_shape=None,
                 force_need_decode=False):
        # For HTTP API testing purposes: Allow to create empty dataset with given shape
        if data is None:
            assert datashape is not None
            data = np.zeros(datashape, dtype=np.float32)
        if num_partitions is None:
            num_partitions = psutil.cpu_count(logical=False)
        # if tileshape is None:
        #     sig_shape = data.shape[-sig_dims:]
        #     target = 2**20
        #     framesize = np.prod(sig_shape)
        #     framecount = max(1, min(np.prod(data.shape[:-sig_dims]), int(target / framesize)))
        #     tileshape = (framecount, ) + sig_shape
        self.data = data
        if tileshape is None:
            self.tileshape = None
        else:
            assert len(tileshape) == sig_dims + 1
            self.tileshape = Shape(tileshape, sig_dims=sig_dims)
        self._base_shape = base_shape
        self.num_partitions = num_partitions
        self.sig_dims = sig_dims
        self._meta = DataSetMeta(
            shape=self.shape,
            raw_dtype=self.data.dtype,
        )
        self._check_cast = check_cast
        self._tiledelay = tiledelay
        self._force_need_decode = force_need_decode

    def initialize(self, executor):
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

    def get_cache_key(self):
        return TypeError("memory data set is not cacheable yet")

    def get_partitions(self):
        fileset = FileSet([
            MemoryFile(
                path=None,
                start_idx=0,
                end_idx=self.shape.nav.size,
                native_dtype=self.data.dtype,
                sig_shape=self.shape.sig,
                data=self.data.reshape(self.shape.flatten_nav()),
                check_cast=self._check_cast,
            )
        ])

        for part_slice, start, stop in BasePartition.make_slices(
                shape=self.shape,
                num_partitions=self.num_partitions):
            log.debug(
                "creating partition slice %s start %s stop %s",
                part_slice, start, stop,
            )
            yield MemPartition(
                meta=self._meta,
                partition_slice=part_slice,
                fileset=fileset.get_for_range(start, stop),
                start_frame=start,
                num_frames=stop - start,
                tiledelay=self._tiledelay,
                tileshape=self.tileshape,
                base_shape=self._base_shape,
                force_need_decode=self._force_need_decode,
            )


class MemPartition(BasePartition):
    def __init__(self, tiledelay, tileshape, base_shape=None, force_need_decode=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tiledelay = tiledelay
        self._tileshape = tileshape
        self._force_tileshape = True
        self._base_shape = base_shape
        self._force_need_decode = force_need_decode

    def _get_decoder(self):
        return None

    def _get_io_backend(self):
        return MemBackend(decoder=self._get_decoder())

    def get_macrotile(self, *args, **kwargs):
        self._force_tileshape = False
        mt = super().get_macrotile(*args, **kwargs)
        self._force_tileshape = True
        return mt

    def get_base_shape(self):
        if self._base_shape is not None:
            return self._base_shape
        return super().get_base_shape()

    def need_decode(self, read_dtype, roi):
        if self._force_need_decode:
            return True
        return super().need_decode(read_dtype, roi)

    def get_tiles(self, *args, **kwargs):
        # force our own tiling_scheme, if a tileshape is given:
        if self._tileshape is not None and self._force_tileshape:
            tiling_scheme = TilingScheme.make_for_shape(
                tileshape=self._tileshape,
                dataset_shape=self.meta.shape,
            )
            kwargs.update({"tiling_scheme": tiling_scheme})
        tiles = super().get_tiles(*args, **kwargs)
        if self._tiledelay:
            log.debug("delayed get_tiles, tiledelay=%.3f" % self._tiledelay)
            for tile in tiles:
                yield tile
                time.sleep(self._tiledelay)
        else:
            yield from tiles
