import os

import numpy as np

from libertem.common import Shape
from .base import (
    DataSet, DataSetException, DataSetMeta,
    Partition3D, File3D, FileSet3D
)
from libertem.web.messages import MessageConverter

MAGIC_EXPECT = 258


class BLODatasetParams(MessageConverter):
    SCHEMA = {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "$id": "http://libertem.org/BLODatasetParams.schema.json",
      "title": "BLODatasetParams",
      "type": "object",
      "properties": {
        "type": {"const": "blo"},
        "path": {"type": "string"},
        "tileshape": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4
        },
      },
      "required": ["type", "path", "tileshape"],
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path", "tileshape"]
        }
        return data


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


class BloFile(File3D):
    def __init__(self, path, endianess, meta, offset_2):
        self._path = path
        self._offset_2 = offset_2
        self._endianess = endianess
        self._meta = meta
        self._num_frames = meta.shape.nav.nav.size

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def start_idx(self):
        return 0

    def open(self):
        self._file = open(self._path, 'rb')
        data = np.memmap(self._file, mode='r', offset=self._offset_2,
                         dtype=self._endianess + 'u1')
        NY, NX, DP_SZ, _ = self._meta.shape
        data = data.reshape((NY * NX, DP_SZ * DP_SZ + 6))
        data = data[:, 6:]
        data = data.reshape((NY * NX, DP_SZ, DP_SZ))
        self._mmap = data

    def close(self):
        self._file.close()
        self._file = None
        self._mmap = None

    def mmap(self):
        return self._mmap

    def readinto(self, start, stop, out, crop_to=None):
        slice_ = (...,)
        if crop_to is not None:
            slice_ = crop_to.get(sig_only=True)
        data = self.mmap()
        out[:] = data[(slice(start, stop),) + slice_]


class BloFileSet(FileSet3D):
    pass


class BloDataSet(DataSet):
    # FIXME include sample file for doctest, see Issue #86
    """
    Read Nanomegas .blo files

    Examples
    --------

    >>> ds = ctx.load("blo", path="/path/to/file.blo")  # doctest: +SKIP

    Parameters
    ----------
    path: str
        Path to the file

    tileshape: tuple of int
        Tuning parameter, specifying the size of the smallest data unit
        we are reading and working on. Will be automatically determined
        if left None.

    endianess: str
        either '<' or '>' for little or big endian
    """
    def __init__(self, path, tileshape=None, endianess='<'):
        super().__init__()
        if tileshape is None:
            tileshape = (1, 8, 144, 144)
        self._tileshape = tileshape
        self._path = path
        self._header = None
        self._endianess = endianess
        self._shape = None
        self._filesize = None

    def initialize(self):
        self._read_header()
        h = self.header
        NY = int(h['NY'])
        NX = int(h['NX'])
        DP_SZ = int(h['DP_SZ'])
        self._shape = Shape((NY, NX, DP_SZ, DP_SZ), sig_dims=2)
        self._meta = DataSetMeta(
            shape=self._shape,
            raw_dtype=np.dtype("u1"),
            iocaps={"MMAP", "FULL_FRAMES", "FRAME_CROPS"},
        )
        self._filesize = os.stat(self._path).st_size
        return self

    @classmethod
    def detect_params(cls, path):
        try:
            ds = cls(path, endianess='<')
            ds = ds.initialize()
            if not ds.check_valid():
                return False
            return {
                "path": path,
                "tileshape": (1, 8) + tuple(ds.shape.sig),
                "endianess": "<",
            }
        except Exception:
            return False

    @classmethod
    def get_msg_converter(cls):
        return BLODatasetParams

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        if self._shape is None:
            raise RuntimeError("please call initialize() before using the dataset")
        return self._shape

    def _read_header(self):
        with open(self._path, 'rb') as f:
            self._header = np.fromfile(f, dtype=get_header_dtype_list(self._endianess), count=1)

    @property
    def header(self):
        if self._header is None:
            raise RuntimeError("please call initialize() before using the dataset")
        return self._header

    def check_valid(self):
        try:
            self._read_header()
            magic = self.header['MAGIC'][0]
            if magic != MAGIC_EXPECT:
                raise DataSetException("invalid magic number: %x != %x" % (magic, MAGIC_EXPECT))
            return True
        except (IOError, OSError) as e:
            raise DataSetException("invalid dataset: %s" % e) from e

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 512MB per partition
        res = max(self._cores, self._filesize // (512*1024*1024))
        return res

    def _get_blo_file(self):
        return BloFile(
            path=self._path,
            offset_2=int(self.header['Data_offset_2']),
            endianess=self._endianess,
            meta=self._meta,
        )

    def get_partitions(self):
        fileset = BloFileSet([
            self._get_blo_file()
        ])

        for part_slice, start, stop in Partition3D.make_slices(
                shape=self.shape,
                num_partitions=self._get_num_partitions()):
            yield Partition3D(
                meta=self._meta,
                partition_slice=part_slice,
                fileset=fileset.get_for_range(start, stop),
                start_frame=start,
                num_frames=stop - start,
                stackheight=self._tileshape[0] * self._tileshape[1],
            )
