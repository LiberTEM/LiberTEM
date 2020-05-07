import os
import warnings

import numpy as np

from libertem.common import Shape
from .base import (
    DataSet, DataSetException, DataSetMeta,
    BasePartition, FileSet, LocalFile,
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
        "type": {"const": "BLO"},
        "path": {"type": "string"},
      },
      "required": ["type", "path"],
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path"]
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


class BloFileSet(FileSet):
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

    endianess: str
        either '<' or '>' for little or big endian
    """
    def __init__(self, path, tileshape=None, endianess='<'):
        super().__init__()
        # handle backwards-compatability:
        if tileshape is not None:
            warnings.warn(
                "tileshape argument is ignored and will be removed after 0.6.0",
                FutureWarning
            )
        self._path = path
        self._header = None
        self._endianess = endianess
        self._shape = None
        self._filesize = None

    def initialize(self, executor):
        self._header = h = executor.run_function(self._read_header)
        NY = int(h['NY'])
        NX = int(h['NX'])
        DP_SZ = int(h['DP_SZ'])
        self._shape = Shape((NY, NX, DP_SZ, DP_SZ), sig_dims=2)
        self._meta = DataSetMeta(
            shape=self._shape,
            raw_dtype=np.dtype("u1"),
        )
        self._filesize = executor.run_function(self._get_filesize)
        return self

    @classmethod
    def detect_params(cls, path, executor):
        try:
            ds = cls(path, endianess='<')
            ds = ds.initialize(executor)
            if not executor.run_function(ds.check_valid):
                return False
            return {
                "parameters": {
                    "path": path,
                    "tileshape": (1, 8) + tuple(ds.shape.sig),
                    "endianess": "<",
                },
            }
        except Exception:
            return False

    @classmethod
    def get_msg_converter(cls):
        return BLODatasetParams

    @classmethod
    def get_supported_extensions(cls):
        return set(["blo"])

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        if self._shape is None:
            raise RuntimeError("please call initialize() before using the dataset")
        return self._shape

    def _read_header(self):
        # FIXME: do this read via the IO backend!
        with open(self._path, 'rb') as f:
            return np.fromfile(f, dtype=get_header_dtype_list(self._endianess), count=1)

    def _get_filesize(self):
        return os.stat(self._path).st_size

    @property
    def header(self):
        if self._header is None:
            raise RuntimeError("please call initialize() before using the dataset")
        return self._header

    def check_valid(self):
        try:
            header = self._read_header()
            magic = header['MAGIC'][0]
            if magic != MAGIC_EXPECT:
                raise DataSetException("invalid magic number: %x != %x" % (magic, MAGIC_EXPECT))
            return True
        except (IOError, OSError) as e:
            raise DataSetException("invalid dataset: %s" % e) from e

    def get_cache_key(self):
        return {
            "path": self._path,
            "endianess": self._endianess,
        }

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 512MB (converted float data) per partition
        partition_size_px = 512 * 1024 * 1024 // 4
        total_size_px = np.prod(self.shape)
        res = max(self._cores, total_size_px // partition_size_px)
        return res

    def _get_fileset(self):
        return BloFileSet([
            LocalFile(
                path=self._path,
                start_idx=0,
                end_idx=self.shape.nav.size,
                native_dtype=self._endianess + "u1",
                sig_shape=self.shape.sig,
                frame_header=6,
                file_header=int(self.header['Data_offset_2']),
            )
        ], frame_header_bytes=6)

    def get_partitions(self):
        fileset = self._get_fileset()

        for part_slice, start, stop in BasePartition.make_slices(
                shape=self.shape,
                num_partitions=self._get_num_partitions()):
            yield BasePartition(
                meta=self._meta,
                fileset=fileset.get_for_range(start, stop),
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
            )
