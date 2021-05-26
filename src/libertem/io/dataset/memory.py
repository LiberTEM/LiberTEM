import time
import logging

import psutil
import numpy as np

from libertem.web.messages import MessageConverter
from libertem.io.dataset.base import (
    FileSet, BasePartition, DataSet, DataSetMeta, TilingScheme,
    LocalFile, MMapBackend,
)
from libertem.io.dataset.base.backend_mmap import MMapBackendImpl
from libertem.common import Shape, Slice
from libertem.io.dataset.base import DataTile


log = logging.getLogger(__name__)


class MemBackend(MMapBackend):
    def get_impl(self):
        return MemBackendImpl()


class MemBackendImpl(MMapBackendImpl):
    def _set_readahead_hints(self, roi, fileset):
        pass

    def _get_tiles_roi(self, tiling_scheme, fileset, read_ranges, roi, sync_offset):
        ds_sig_shape = tiling_scheme.dataset_shape.sig
        sig_dims = tiling_scheme.shape.sig.dims
        slices, ranges, scheme_indices = read_ranges

        fh = fileset[0]
        memmap = fh.mmap().reshape((fh.num_frames,) + tuple(ds_sig_shape))
        if sync_offset == 0:
            flat_roi = roi.reshape((-1,))
        elif sync_offset > 0:
            flat_roi = np.full(roi.reshape((-1,)).shape, False)
            flat_roi[:sync_offset] = False
            flat_roi[sync_offset:] = roi.reshape((-1,))[:-sync_offset]
        else:
            flat_roi = np.full(roi.reshape((-1,)).shape, False)
            flat_roi[sync_offset:] = False
            flat_roi[:sync_offset] = roi.reshape((-1,))[-sync_offset:]
        data_w_roi = memmap[flat_roi]
        for idx in range(slices.shape[0]):
            origin, shape = slices[idx]
            scheme_idx = scheme_indices[idx]

            tile_slice = Slice(
                origin=origin,
                shape=Shape(shape, sig_dims=sig_dims)
            )
            if sync_offset >= 0:
                data_slice = tile_slice.get()
            else:
                frames_to_skip = np.count_nonzero(roi.reshape((-1,))[:abs(sync_offset)])
                data_slice = Slice(
                    origin=(origin[0] - frames_to_skip,) + tuple(origin[-sig_dims:]),
                    shape=Shape(shape, sig_dims=sig_dims)
                )
                data_slice = data_slice.get()
            data = data_w_roi[data_slice]
            yield DataTile(
                data,
                tile_slice=tile_slice,
                scheme_idx=scheme_idx,
            )

    def get_tiles(
        self, decoder, tiling_scheme, fileset, read_ranges, roi, native_dtype, read_dtype,
        sync_offset, corrections,
    ):
        if roi is None:
            # support arbitrary tiling in case of no roi
            with fileset:
                if sync_offset >= 0:
                    for tile in self._get_tiles_straight(
                        tiling_scheme, fileset, read_ranges, sync_offset
                    ):
                        data = tile.astype(read_dtype)
                        self.preprocess(data, tile.tile_slice, corrections)
                        yield data
                else:
                    for tile in self._get_tiles_w_copy(
                        tiling_scheme=tiling_scheme,
                        fileset=fileset,
                        read_ranges=read_ranges,
                        read_dtype=read_dtype,
                        native_dtype=native_dtype,
                        decoder=decoder,
                        corrections=corrections,
                    ):
                        yield tile
        else:
            with fileset:
                for tile in self._get_tiles_roi(
                    tiling_scheme=tiling_scheme,
                    fileset=fileset,
                    read_ranges=read_ranges,
                    roi=roi,
                    sync_offset=sync_offset,
                ):
                    data = tile.astype(read_dtype)
                    self.preprocess(data, tile.tile_slice, corrections)
                    yield data


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
        "required": ["type", "tileshape", "num_partitions"],
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["tileshape", "num_partitions", "sig_dims", "check_cast",
                      "crop_frames", "tiledelay", "datashape"]
            if k in raw_data
        }
        if "nav_shape" in raw_data:
            data["nav_shape"] = tuple(raw_data["nav_shape"])
        if "sig_shape" in raw_data:
            data["sig_shape"] = tuple(raw_data["sig_shape"])
        if "sync_offset" in raw_data:
            data["sync_offset"] = raw_data["sync_offset"]
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
                 force_need_decode=False, io_backend=None,
                 nav_shape=None, sig_shape=None, sync_offset=0):
        super().__init__(io_backend=io_backend)
        if io_backend is not None:
            raise ValueError("MemoryDataSet currently doesn't support alternative I/O backends")
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
        self._base_shape = base_shape
        self.num_partitions = num_partitions
        self.sig_dims = sig_dims
        if nav_shape is None:
            self._nav_shape = self.shape.nav
        else:
            self._nav_shape = tuple(nav_shape)
        if sig_shape is None:
            self._sig_shape = self.shape.sig
        else:
            self._sig_shape = tuple(sig_shape)
            self.sig_dims = len(self._sig_shape)
        self._sync_offset = sync_offset
        self._check_cast = check_cast
        self._tiledelay = tiledelay
        self._force_need_decode = force_need_decode
        self._image_count = int(np.prod(self._nav_shape))
        self._shape = Shape(
            tuple(self._nav_shape) + tuple(self._sig_shape), sig_dims=self.sig_dims
        )
        if tileshape is None:
            self.tileshape = None
        else:
            assert len(tileshape) == self.sig_dims + 1
            self.tileshape = Shape(tileshape, sig_dims=self.sig_dims)
        self._nav_shape_product = int(np.prod(self._nav_shape))
        self._sync_offset_info = self.get_sync_offset_info()
        self._meta = DataSetMeta(
            shape=self._shape,
            raw_dtype=self.data.dtype,
            sync_offset=self._sync_offset,
            image_count=self._image_count,
        )

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

    def get_num_partitions(self):
        return self.num_partitions

    def get_partitions(self):
        fileset = FileSet([
            MemoryFile(
                path=None,
                start_idx=0,
                end_idx=self._image_count,
                native_dtype=self.data.dtype,
                sig_shape=self.shape.sig,
                data=self.data.reshape(self.shape.flatten_nav()),
                check_cast=self._check_cast,
            )
        ])

        for part_slice, start, stop in self.get_slices():
            log.debug(
                "creating partition slice %s start %s stop %s",
                part_slice, start, stop,
            )
            yield MemPartition(
                meta=self._meta,
                partition_slice=part_slice,
                fileset=fileset,
                start_frame=start,
                num_frames=stop - start,
                tiledelay=self._tiledelay,
                tileshape=self.tileshape,
                base_shape=self._base_shape,
                force_need_decode=self._force_need_decode,
                io_backend=self.get_io_backend(),
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

    def get_io_backend(self):
        return MemBackend()

    def get_macrotile(self, *args, **kwargs):
        self._force_tileshape = False
        mt = super().get_macrotile(*args, **kwargs)
        self._force_tileshape = True
        return mt

    def get_base_shape(self, roi):
        if self._base_shape is not None:
            return self._base_shape
        return super().get_base_shape(roi)

    def need_decode(self, read_dtype, roi, corrections):
        if self._force_need_decode:
            return True
        return super().need_decode(read_dtype, roi, corrections)

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
