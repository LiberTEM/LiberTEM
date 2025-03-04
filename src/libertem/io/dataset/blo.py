import os
import warnings

import numpy as np

from libertem.common.math import prod
from libertem.common import Shape
from .base import (
    DataSet, DataSetException, DataSetMeta,
    BasePartition, FileSet, File, IOBackend,
)
from libertem.common.messageconverter import MessageConverter

MAGIC_EXPECT = (258, 259)


class BLODatasetParams(MessageConverter):
    SCHEMA = {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "$id": "http://libertem.org/BLODatasetParams.schema.json",
      "title": "BLODatasetParams",
      "type": "object",
      "properties": {
        "type": {"const": "BLO"},
        "path": {"type": "string"},
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
        "io_backend": {
            "enum": IOBackend.get_supported(),
        },
      },
      "required": ["type", "path"],
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path"]
        }
        if "nav_shape" in raw_data:
            data["nav_shape"] = tuple(raw_data["nav_shape"])
        if "sig_shape" in raw_data:
            data["sig_shape"] = tuple(raw_data["sig_shape"])
        if "sync_offset" in raw_data:
            data["sync_offset"] = raw_data["sync_offset"]
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

    nav_shape: tuple of int, optional
        A n-tuple that specifies the size of the navigation region ((y, x), but
        can also be of length 1 for example for a line scan, or length 3 for
        a data cube, for example)

    sig_shape: tuple of int, optional
        Signal/detector size (height, width)

    sync_offset: int, optional
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start

    num_partitions: int, optional
        Override the number of partitions. This is useful if the
        default number of partitions, chosen based on common workloads,
        creates partitions which are too large (or small) for the UDFs
        being run on this dataset.
    """
    def __init__(
        self,
        path,
        tileshape=None,
        endianess='<',
        nav_shape=None,
        sig_shape=None,
        sync_offset=0,
        io_backend=None,
        num_partitions=None,
    ):
        super().__init__(
            io_backend=io_backend,
            num_partitions=num_partitions,
        )
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
        self._nav_shape = tuple(nav_shape) if nav_shape else nav_shape
        self._sig_shape = tuple(sig_shape) if sig_shape else sig_shape
        self._sync_offset = sync_offset

    def initialize(self, executor):
        self._header = h = executor.run_function(self._read_header)
        metadata = executor.run_function(self._read_metadata)
        dtype = self._try_read_bitdepth(metadata)
        NY = int(h['NY'][0])
        NX = int(h['NX'][0])
        DP_SZ = int(h['DP_SZ'][0])
        self._image_count = NY * NX
        if self._nav_shape is None:
            self._nav_shape = (NY, NX)
        if self._sig_shape is None:
            self._sig_shape = (DP_SZ, DP_SZ)
        elif int(prod(self._sig_shape)) != (DP_SZ * DP_SZ):
            raise DataSetException(
                "sig_shape must be of size: %s" % (DP_SZ * DP_SZ)
            )
        self._nav_shape_product = int(prod(self._nav_shape))
        self._sync_offset_info = self.get_sync_offset_info()
        self._shape = Shape(self._nav_shape + self._sig_shape, sig_dims=len(self._sig_shape))
        self._meta = DataSetMeta(
            shape=self._shape,
            raw_dtype=np.dtype(dtype),
            sync_offset=self._sync_offset,
            image_count=self._image_count,
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
                    "nav_shape": tuple(ds.shape.nav),
                    "sig_shape": tuple(ds.shape.sig),
                    "tileshape": (1, 8) + tuple(ds.shape.sig),
                    "endianess": "<",
                },
                "info": {
                    "image_count": int(prod(ds.shape.nav)),
                    "native_sig_shape": tuple(ds.shape.sig),
                }
            }
        except Exception:
            return False

    @classmethod
    def get_msg_converter(cls):
        return BLODatasetParams

    @classmethod
    def get_supported_extensions(cls):
        return {"blo"}

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

    def _read_metadata(self) -> tuple[str]:
        """
        Read metadata from the block after the inital 64 byte header
        but before the first Data section (as interpreted from the header).

        In the files alrady seen, this is a Windows-delimited sequence
        of lines of text starting at byte 240.

        As we aren't confident in the format this function doesn't try
        to decode the metadata, it just loads it and cleans up any null
        bytes or extra whitespace.
        """
        if self._header['MAGIC'] == 259:
            metadata_start_pos = 240
            data_start_byte = int(self._header['Data_offset_1'].item())
            with open(self._path, 'rb') as f:
                f.seek(metadata_start_pos)
                metadata_bytes = f.read(data_start_byte - metadata_start_pos)
            metadata = metadata_bytes.decode(errors='ignore').strip('\x00')
            metadata = (m.strip() for m in metadata.splitlines())
            return tuple(m for m in metadata if len(m) > 0)
        return tuple()

    @staticmethod
    def _try_read_bitdepth(metadata: tuple[str, ...]) -> str:
        """
        Try to extract a bitdepth from the metadata items
        using the key in the post-header section of the file
        like:

            Blo Bit Depth: 16 bits

        Falls back to u1 dtype if metadata is missing
        or does not exactly conform to the format already seen.
        If metadata is missing (older format) this is a no-op.

        The alternative to this function is to infer the bit depth
        using the filesize, data_offset and nav/sig shapes.
        """
        for m in metadata:
            if not m.lower().startswith('blo bit depth:'):
                continue
            for p in m.split(':'):
                p = p.strip().lower()
                if not p.endswith(' bits'):
                    continue
                try:
                    bitdepth = int(p.replace(' bits', ''))
                    return f"u{bitdepth // 8}"
                except ValueError:
                    break
        return "u1"

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
            if magic not in MAGIC_EXPECT:
                raise DataSetException(
                    f"invalid magic number: {magic:x} "
                    f"not in {tuple(hex(x) for x in MAGIC_EXPECT)}"
                )
            return True
        except OSError as e:
            raise DataSetException("invalid dataset: %s" % e) from e

    def get_cache_key(self):
        return {
            "path": self._path,
            "endianess": self._endianess,
            "shape": tuple(self.shape),
            "sync_offset": self._sync_offset,
        }

    def _get_fileset(self):
        return BloFileSet([
            File(
                path=self._path,
                start_idx=0,
                end_idx=self._image_count,
                native_dtype=(
                    self._endianess
                    + self.meta.raw_dtype.kind
                    + str(self.meta.raw_dtype.itemsize)
                ),
                sig_shape=self.shape.sig,
                frame_header=6,
                file_header=int(self.header['Data_offset_2'][0]),
            )
        ], frame_header_bytes=6)

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in self.get_slices():
            yield BasePartition(
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
                io_backend=self.get_io_backend(),
            )

    def __repr__(self):
        return f"<BloDataSet of {self.dtype} shape={self.shape}>"
