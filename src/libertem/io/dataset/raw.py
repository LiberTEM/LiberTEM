import numpy as np

from libertem.common.slice import Slice
from .base import DataSet, Partition, DataTile, DataSetException


class RawFileDataSet(DataSet):
    def __init__(self, path, scan_size, dtype, detector_size_raw, crop_detector_to, tileshape=None):
        self._path = path
        self._scan_size = tuple(scan_size)
        self._dtype = dtype
        assert len(detector_size_raw) == 2
        self._detector_size_raw = tuple(detector_size_raw)  # example: (130, 128)
        self._detector_size = tuple(crop_detector_to)                # example: (128, 128)
        self._min_num_partitions = None  # FIXME
        if tileshape is None:
            # raw files are memory mapped -> works well with large tiles
            # (actual tiles are then as large as the partitions)
            tileshape = self._scan_size + self._detector_size
        self._tileshape = tuple(tileshape)

    def open_file(self):
        f = np.memmap(self._path, dtype=self.dtype, mode='r',
                      shape=self._scan_size + self._detector_size_raw)
        ds_slice = Slice(origin=(0, 0, 0, 0), shape=self.shape)
        return f[ds_slice.get()]  # crop off the two extra rows

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._scan_size + self._detector_size

    def check_valid(self):
        try:
            self.open_file()
            return True  # TODO: try to read from file? anything else?
        except (IOError, OSError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def get_partitions(self):
        ds_slice = Slice(origin=(0, 0, 0, 0), shape=self.shape)
        partition_shape = Slice.partition_shape(
            datashape=self.shape,
            framesize=self._detector_size[0] * self._detector_size[1],
            dtype=self.dtype,
            target_size=256*1024*1024,
            min_num_partitions=self._min_num_partitions,
        )
        for pslice in ds_slice.subslices(partition_shape):
            # TODO: where should the tileshape be set? let the user choose for now
            yield RawFilePartition(
                tileshape=self._tileshape,
                dataset=self,
                dtype=self.dtype,
                partition_slice=pslice,
            )

    def __repr__(self):
        return "<RawFileDataSet of %s shape=%s>" % (self.dtype, self.shape)


class RawFilePartition(Partition):
    def __init__(self, tileshape, *args, **kwargs):
        self.tileshape = tileshape
        super().__init__(*args, **kwargs)

    def get_tiles(self, crop_to=None):
        if crop_to is not None:
            if crop_to.shape[2:] != self.dataset.shape[2:]:
                raise DataSetException("RawFileDataSet only supports whole-frame crops for now")
        f = self.dataset.open_file()
        subslices = list(self.slice.subslices(shape=self.tileshape))
        for tile_slice in subslices:
            if crop_to is not None:
                intersection = tile_slice.intersection_with(crop_to)
                if intersection.is_null():
                    continue
            # NOTE: no need to re-use buffer, as there is none (mmap!)
            yield DataTile(
                data=f[tile_slice.get()],
                tile_slice=tile_slice
            )

    def get_locations(self):
        return "127.0.1.1"  # FIXME
