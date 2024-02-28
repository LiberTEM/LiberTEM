import time
import logging
from typing import Optional, TYPE_CHECKING
from collections.abc import Sequence

import psutil
import numpy as np
from sparseconverter import (
    BACKENDS, ArrayBackend, for_backend, get_backend, SPARSE_BACKENDS, get_device_class,
    conversion_cost
)

from libertem.common.math import prod, count_nonzero, flat_nonzero
from libertem.common.messageconverter import MessageConverter
from libertem.io.dataset.base import (
    FileSet, BasePartition, DataSet, DataSetMeta, TilingScheme,
    File, MMapBackend, DataSetException
)
from libertem.io.dataset.base.backend_mmap import MMapBackendImpl, MMapFile
from libertem.common import Shape, Slice
from libertem.io.dataset.base import DataTile

if TYPE_CHECKING:
    from libertem.common.executor import JobExecutor

log = logging.getLogger(__name__)


class FakeMMapFile(MMapFile):
    """
    Implementing the same interface as MMapFile, without filesystem backing
    """
    def open(self):
        self._arr = self.desc._data
        self._mmap = self.desc._data
        return self

    def close(self):
        self._arr = None
        self._mmap = None


class MemBackend(MMapBackend):
    def get_impl(self):
        return MemBackendImpl()


class MemBackendImpl(MMapBackendImpl):
    FILE_CLS = FakeMMapFile

    def _set_readahead_hints(self, roi, fileset):
        pass

    def _get_tiles_roi(self, tiling_scheme, open_files, read_ranges, roi, sync_offset):
        ds_sig_shape = tiling_scheme.dataset_shape.sig
        sig_dims = tiling_scheme.shape.sig.dims
        slices, ranges, scheme_indices = read_ranges

        fh = open_files[0]
        memmap = fh.array.reshape((fh.desc.num_frames,) + tuple(ds_sig_shape))
        flat_roi = np.full(roi.reshape((-1,)).shape, False)
        roi_nonzero = flat_nonzero(roi)
        offset_roi = np.clip(roi_nonzero + sync_offset, 0, flat_roi.size)
        flat_roi[offset_roi] = True
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
                frames_to_skip = count_nonzero(roi.reshape((-1,))[:abs(sync_offset)])
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
        sync_offset, corrections, array_backend: ArrayBackend,
    ):
        if roi is None:
            # support arbitrary tiling in case of no roi
            with self.open_files(fileset) as open_files:
                if sync_offset >= 0:
                    for tile in self._get_tiles_straight(
                        tiling_scheme, open_files, read_ranges, sync_offset
                    ):
                        if tile.dtype != read_dtype or tile.c_contiguous is False:
                            data = tile.data.astype(read_dtype)
                        else:
                            data = tile.data
                        self.preprocess(data, tile.tile_slice, corrections)
                        data = for_backend(data, array_backend)
                        yield DataTile(data, tile.tile_slice, tile.scheme_idx)
                else:
                    for tile in self._get_tiles_w_copy(
                        tiling_scheme=tiling_scheme,
                        open_files=open_files,
                        read_ranges=read_ranges,
                        read_dtype=read_dtype,
                        native_dtype=native_dtype,
                        decoder=decoder,
                        corrections=corrections,
                    ):
                        data = for_backend(tile.data, array_backend)
                        yield DataTile(data, tile.tile_slice, tile.scheme_idx)
        else:
            with self.open_files(fileset) as open_files:
                for tile in self._get_tiles_roi(
                    tiling_scheme=tiling_scheme,
                    open_files=open_files,
                    read_ranges=read_ranges,
                    roi=roi,
                    sync_offset=sync_offset,
                ):
                    data = tile.data.astype(read_dtype)
                    self.preprocess(data, tile.tile_slice, corrections)
                    data = for_backend(data, array_backend)
                    yield DataTile(data, tile.tile_slice, tile.scheme_idx)


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
            "array_backend": {"type": "string"},
        },
        "required": ["type", "tileshape", "num_partitions"],
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["tileshape", "num_partitions", "sig_dims", "check_cast",
                      "crop_frames", "tiledelay", "datashape", "array_backend"]
            if k in raw_data
        }
        if "nav_shape" in raw_data:
            data["nav_shape"] = tuple(raw_data["nav_shape"])
        if "sig_shape" in raw_data:
            data["sig_shape"] = tuple(raw_data["sig_shape"])
        if "sync_offset" in raw_data:
            data["sync_offset"] = raw_data["sync_offset"]
        return data


class MemoryFile(File):
    def __init__(self, data, check_cast=True, *args, **kwargs):
        self._data = data
        self._check_cast = check_cast
        super().__init__(*args, **kwargs)

    @property
    def data(self):
        return self._data


class MemoryDataSet(DataSet):
    '''
    This dataset is constructed from a NumPy array in memory for testing
    purposes. It is not recommended for production use since it performs poorly with a
    distributed executor.

    Examples
    --------

    >>> data = np.zeros((2, 2, 64, 64), dtype=np.float32)
    >>> ds = ctx.load('memory', data=data, sig_dims=2)
    '''
    def __init__(self, tileshape=None, num_partitions=None, data=None, sig_dims=None,
                 check_cast=True, tiledelay=None, datashape=None, base_shape=None,
                 force_need_decode=False, io_backend=None,
                 nav_shape=None, sig_shape=None, sync_offset=0, array_backends=None):
        super().__init__(io_backend=io_backend)
        if io_backend is not None:
            raise ValueError("MemoryDataSet currently doesn't support alternative I/O backends")
        # For HTTP API testing purposes: Allow to create empty dataset with given shape
        if data is None:
            if datashape is None:
                raise DataSetException('MemoryDataSet can be created from either data [np.ndarray],'
                                       ' or datashape [tuple | Shape], both arguments are None')
            data = np.zeros(datashape, dtype=np.float32)
        if num_partitions is None:
            num_partitions = psutil.cpu_count(logical=False)
        # if tileshape is None:
        #     sig_shape = data.shape[-sig_dims:]
        #     target = 2**20
        #     framesize = prod(sig_shape)
        #     framecount = max(1, min(prod(data.shape[:-sig_dims]), int(target / framesize)))
        #     tileshape = (framecount, ) + sig_shape
        self.data = data
        self._base_shape = base_shape
        self.num_partitions = num_partitions

        if sig_dims is None:
            if sig_shape is not None:
                sig_dims = len(sig_shape)
            elif nav_shape is not None:
                sig_dims = len(data.shape) - len(nav_shape)
            else:
                sig_dims = 2
        else:
            if sig_shape is not None and len(sig_shape) != sig_dims:
                raise ValueError(
                    f"Length of sig_shape {sig_shape} not matching sig_dims {sig_dims}."
                )

        self.sig_dims = sig_dims
        if nav_shape is None:
            nav_shape = data.shape[:-sig_dims]
        else:
            nav_shape = tuple(nav_shape)
        if sig_shape is None:
            sig_shape = data.shape[-sig_dims:]
        else:
            sig_shape = tuple(sig_shape)
        if self.data.size % prod(sig_shape) != 0:
            raise ValueError("Data size is not a multiple of sig shape")
        self._image_count = self.data.size // prod(sig_shape)
        self._nav_shape = nav_shape
        self._sig_shape = sig_shape
        self._sync_offset = sync_offset
        self._array_backends = array_backends
        self._check_cast = check_cast
        self._tiledelay = tiledelay
        self._force_need_decode = force_need_decode
        self._nav_shape_product = int(prod(nav_shape))
        self._shape = Shape(
            nav_shape + sig_shape, sig_dims=self.sig_dims
        )
        if tileshape is None:
            self.tileshape = None
        else:
            assert len(tileshape) == self.sig_dims + 1
            self.tileshape = Shape(tileshape, sig_dims=self.sig_dims)
        self._sync_offset_info = self.get_sync_offset_info()
        self._meta = DataSetMeta(
            shape=self._shape,
            array_backends=self.array_backends,
            raw_dtype=self.data.dtype,
            sync_offset=self._sync_offset,
            image_count=self._image_count,
        )

    def initialize(self, executor):
        return self

    @classmethod
    def get_msg_converter(cls):
        return MemDatasetParams

    @classmethod
    def detect_params(cls, data: np.ndarray, executor: "JobExecutor"):
        try:
            _ = data.shape
            return {
                "parameters": {
                    "data": data,
                },
                "info": {}
            }
        except AttributeError:
            return False

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self._shape

    @property
    def array_backends(self) -> Sequence[ArrayBackend]:
        """
        All backends can be returned on request

        .. versionadded:: 0.11.0
        """
        if self._array_backends is None:
            native = get_backend(self.data)
            is_sparse = native in SPARSE_BACKENDS
            native_device_class = get_device_class(native)
            cost_metric = {}
            # Sort by tuple (cost_override, conversion_cost),
            # meaning preference for native backend, same sparsity
            # and same device class take precedence over measured conversion cost
            for backend in BACKENDS:
                cost_metric[backend] = [5, conversion_cost(native, backend)]
                if backend == native:
                    cost_metric[backend][0] -= 2
                # sparse==sparse or dense==dense
                if (backend in SPARSE_BACKENDS) == is_sparse:
                    cost_metric[backend][0] -= 2
                # Same device class
                if get_device_class(backend) == native_device_class:
                    cost_metric[backend][0] -= 1
            return tuple(sorted(BACKENDS, key=lambda k: cost_metric[k]))
        else:
            return self._array_backends

    def check_valid(self):
        return True

    def get_cache_key(self):
        return TypeError("memory data set is not cacheable yet")

    def get_num_partitions(self):
        return self.num_partitions

    def get_base_shape(self, roi):
        if self.tileshape is not None:
            return self.tileshape
        if self._base_shape is not None:
            return self._base_shape
        return super().get_base_shape(roi)

    def adjust_tileshape(
        self, tileshape: tuple[int, ...], roi: Optional[np.ndarray],
    ) -> tuple[int, ...]:
        if self.tileshape is not None:
            return tuple(self.tileshape)
        return super().adjust_tileshape(tileshape, roi)

    def need_decode(self, read_dtype, roi, corrections):
        if self._force_need_decode:
            return True
        return super().need_decode(read_dtype, roi, corrections)

    def get_io_backend(self):
        return MemBackend()

    def get_partitions(self):
        fileset = FileSet([
            MemoryFile(
                path=None,
                start_idx=0,
                end_idx=self._image_count,
                native_dtype=self.data.dtype,
                sig_shape=self.shape.sig,
                data=self.data.reshape((-1, *self.shape.sig)),
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
                force_need_decode=self._force_need_decode,
                io_backend=self.get_io_backend(),
                decoder=self.get_decoder(),
            )


class MemPartition(BasePartition):
    def __init__(self, tiledelay, tileshape, force_need_decode=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tiledelay = tiledelay
        self._tileshape = tileshape
        self._force_tileshape = True
        self._force_need_decode = force_need_decode

    def get_io_backend(self):
        return MemBackend()

    def get_macrotile(self, *args, **kwargs):
        self._force_tileshape = False
        mt = super().get_macrotile(*args, **kwargs)
        self._force_tileshape = True
        return mt

    def get_tiles(self, *args, **kwargs):
        if args and isinstance(args[0], TilingScheme):
            tiling_scheme = args[0]
            args = args[1:]
            intent = tiling_scheme.intent
        elif 'tiling_scheme' in kwargs:
            tiling_scheme = kwargs.pop('tiling_scheme')
            intent = tiling_scheme.intent
        else:
            # In this case we require the next if-block to execute
            intent = None
        # force our own tiling_scheme, if a tileshape is given:
        if self._tileshape is not None and self._force_tileshape:
            tiling_scheme = TilingScheme.make_for_shape(
                tileshape=self._tileshape,
                dataset_shape=self.meta.shape,
                intent=intent,
            )
        tiles = super().get_tiles(tiling_scheme, *args, **kwargs)
        if self._tiledelay:
            log.debug("delayed get_tiles, tiledelay=%.3f" % self._tiledelay)
            for tile in tiles:
                yield tile
                time.sleep(self._tiledelay)
        else:
            yield from tiles
