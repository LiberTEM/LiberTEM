import contextlib

import numpy as np

from libertem.common import Slice, Shape
from .base import DataSet, Partition, DataTile, DataSetException

MAGIC_EXPECT = 258


# stolen from hyperspy
def get_header_dtype_list(endianess='<'):
    end = endianess
    dtype_list = \
        [
            ('ID', (bytes, 6)),
            ('MAGIC', end + 'u2'),
            ('Data_offset_1', end + 'u4'),      # Offset VBF
            ('Data_offset_2', end + 'u4'),      # Offset DPs
            ('UNKNOWN1', end + 'u4'),           # Flags for ASTAR software?
            ('DP_SZ', end + 'u2'),              # Pixel dim DPs
            ('DP_rotation', end + 'u2'),        # [degrees ( * 100 ?)]
            ('NX', end + 'u2'),                 # Scan dim 1
            ('NY', end + 'u2'),                 # Scan dim 2
            ('Scan_rotation', end + 'u2'),      # [100 * degrees]
            ('SX', end + 'f8'),                 # Pixel size [nm]
            ('SY', end + 'f8'),                 # Pixel size [nm]
            ('Beam_energy', end + 'u4'),        # [V]
            ('SDP', end + 'u2'),                # Pixel size [100 * ppcm]
            ('Camera_length', end + 'u4'),      # [10 * mm]
            ('Acquisition_time', end + 'f8'),   # [Serial date]
        ] + [
            ('Centering_N%d' % i, 'f8') for i in range(8)
        ] + [
            ('Distortion_N%02d' % i, 'f8') for i in range(14)
        ]

    return dtype_list


class BloDataSet(DataSet):
    def __init__(self, path, tileshape, endianess='<'):
        self._tileshape = tileshape
        self._path = path
        self._header = None
        self._endianess = endianess

    @classmethod
    def detect_params(cls, path):
        try:
            ds = cls(path, tileshape=(1, 1, 144, 144), endianess='<')
            if not ds.check_valid():
                return False
            return {
                "path": path,
                "tileshape": (1, 8) + ds.shape.sig,  # FIXME: maybe adjust number of frames?
                "endianess": "<",
            }
        except Exception:
            return False

    @property
    def dtype(self):
        return np.dtype("u1")

    @property
    def raw_shape(self):
        h = self.header
        NY = int(h['NY'])
        NX = int(h['NX'])
        DP_SZ = int(h['DP_SZ'])
        return Shape((NY, NX, DP_SZ, DP_SZ), sig_dims=2)

    def _read_header(self):
        with open(self._path, 'rb') as f:
            self._header = np.fromfile(f, dtype=get_header_dtype_list(self._endianess), count=1)

    @property
    def header(self):
        if self._header is None:
            self._read_header()
        return self._header

    def check_valid(self):
        try:
            self._read_header()
            magic = self.header['MAGIC'][0]
            if magic != MAGIC_EXPECT:
                raise DataSetException("invalid magic number: %x != %x" % (magic, MAGIC_EXPECT))
            return True
        except (IOError, OSError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def get_partitions(self):
        ds_slice = Slice(origin=(0, 0, 0, 0), shape=self.shape)
        partition_shape = self.partition_shape(
            datashape=self.shape,
            framesize=self.shape[2] * self.shape[3],
            dtype=self.dtype,
            target_size=256*1024*1024,
        )
        for pslice in ds_slice.subslices(partition_shape):
            yield BloPartition(
                tileshape=self._tileshape,
                dataset=self,
                dtype=self.dtype,
                partition_slice=pslice,
            )

    @contextlib.contextmanager
    def get_data(self):
        with open(self._path, 'rb') as f:
            data = np.memmap(f, mode='r', offset=int(self.header['Data_offset_2']),
                             dtype=self._endianess + 'u1')
            NY, NX, DP_SZ, _ = self.shape
            data = data.reshape((NY, NX, DP_SZ * DP_SZ + 6))
            data = data[:, :, 6:]
            data = data.reshape(self.shape)
            yield data


class BloPartition(Partition):
    def __init__(self, tileshape, *args, **kwargs):
        self.tileshape = tileshape
        super().__init__(*args, **kwargs)

    def get_tiles(self, crop_to=None):
        if crop_to is not None:
            if crop_to.shape.sig != self.dataset.shape.sig:
                raise DataSetException("BloDataSet only supports whole-frame crops for now")
        with self.dataset.get_data() as data:
            subslices = list(self.slice.subslices(shape=self.tileshape))
            for tile_slice in subslices:
                if crop_to is not None:
                    intersection = tile_slice.intersection_with(crop_to)
                    if intersection.is_null():
                        continue
                # NOTE: no need to re-use buffer, as there is none (mmap!)
                yield DataTile(
                    data=data[tile_slice.get()],
                    tile_slice=tile_slice
                )
