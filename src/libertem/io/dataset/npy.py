import io
import os
import sys
import typing
from typing import Optional, Tuple
import logging

import numpy as np
from numpy.lib.utils import safe_eval
from numpy.lib.format import read_magic
from numpy.compat import long, asstr
from libertem.common.messageconverter import MessageConverter

from libertem.io.dataset.base import (
    DataSet, FileSet, BasePartition, DataSetException, DataSetMeta, File, IOBackend,
)
from libertem.common import Shape
from libertem.common.math import prod

log = logging.getLogger(__name__)


class NPYDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/NPYDatasetParams.schema.json",
        "title": "NPYDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "NPY"},
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
            data["sync_offset"] = int(raw_data["sync_offset"])
        return data


class NPYInfo(typing.NamedTuple):
    dtype: str
    shape: typing.Tuple[int]
    count: int
    offset: int


# `_read_bytes`, `_read_array_header`, `_filter_header` are
# stolen from `numpy.lib.format`, as they don't appear to be
# part of the public API.

def _read_bytes(fp, size, error_template="ran out of data"):
    """
    Read from file-like object until size bytes are read.
    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.

    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.
    """
    data = b''
    while True:
        # io files (default in python3) return None or raise on
        # would-block, python2 file will truncate, probably nothing can be
        # done about that.  note that regular files can't be non-blocking
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data


def _read_array_header(fp, version):
    """
    see read_array_header_1_0
    """
    # Read an unsigned, little-endian short int which has the length of the
    # header.
    import struct
    if version == (1, 0):
        hlength_str = _read_bytes(fp, 2, "array header length")
        header_length = struct.unpack('<H', hlength_str)[0]
        header = _read_bytes(fp, header_length, "array header")
    elif version == (2, 0):
        hlength_str = _read_bytes(fp, 4, "array header length")
        header_length = struct.unpack('<I', hlength_str)[0]
        header = _read_bytes(fp, header_length, "array header")
    else:
        raise ValueError("Invalid version %r" % version)

    # The header is a pretty-printed string representation of a literal
    # Python dictionary with trailing newlines padded to a 16-byte
    # boundary. The keys are strings.
    #   "shape" : tuple of int
    #   "fortran_order" : bool
    #   "descr" : dtype.descr
    header = _filter_header(header)
    try:
        d = safe_eval(header)
    except SyntaxError as e:
        msg = "Cannot parse header: %r\nException: %r"
        raise ValueError(msg % (header, e))
    if not isinstance(d, dict):
        msg = "Header is not a dictionary: %r"
        raise ValueError(msg % d)
    keys = sorted(d.keys())
    if keys != ['descr', 'fortran_order', 'shape']:
        msg = "Header does not contain the correct keys: %r"
        raise ValueError(msg % (keys,))

    # Sanity-check the values.
    if (not isinstance(d['shape'], tuple)
            or not np.all([isinstance(x, (int, long)) for x in d['shape']])):
        msg = "shape is not valid: %r"
        raise ValueError(msg % (d['shape'],))
    if not isinstance(d['fortran_order'], bool):
        msg = "fortran_order is not a valid bool: %r"
        raise ValueError(msg % (d['fortran_order'],))
    try:
        dtype = np.dtype(d['descr'])
    except TypeError:
        msg = "descr is not a valid dtype descriptor: %r"
        raise ValueError(msg % (d['descr'],))

    return d['shape'], d['fortran_order'], dtype


def _filter_header(s):
    """Clean up 'L' in npz header ints.

    Cleans up the 'L' in strings representing integers. Needed to allow npz
    headers produced in Python2 to be read in Python3.

    Parameters
    ----------
    s : byte string
        Npy file header.

    Returns
    -------
    header : str
        Cleaned up header.

    """
    import tokenize
    if sys.version_info[0] >= 3:
        from io import StringIO
    else:
        from StringIO import StringIO

    tokens = []
    last_token_was_number = False
    for token in tokenize.generate_tokens(StringIO(asstr(s)).read):
        token_type = token[0]
        token_string = token[1]
        if (last_token_was_number
                and token_type == tokenize.NAME
                and token_string == "L"):
            continue
        else:
            tokens.append(token)
        last_token_was_number = (token_type == tokenize.NUMBER)
    return tokenize.untokenize(tokens)


def read_npy_info(path: str) -> NPYInfo:
    with open(path, "rb") as fp:
        version = read_magic(fp)
        shape, fortran_order, dtype = _read_array_header(fp, version)
        if fortran_order:
            raise DataSetException('Unable to process Fortran-ordered NPY arrays, '
                                   'consider converting with np.ascontiguousarray().')
        if len(shape) == 0:
            count = 1
        else:
            count = int(np.multiply.reduce(shape, dtype=np.int64))
        offset = fp.tell()
        return NPYInfo(dtype=dtype, shape=shape, count=count, offset=offset)


class NPYDataSet(DataSet):
    """
    .. versionadded:: 0.10.0

    Read data stored in a NumPy .npy binary file. Dataset shape
    and dtype are inferred from the file header unless overridden
    by the arguments to this class.

    As of this time Fortran-ordered .npy files are not supported

    Parameters
    ----------
    path : str
        The path to the .npy file
    sig_dims : int, optional, by default 2
        The number of dimensions from the end of the full shape
        to interpret as signal dimensions. If None
        will be inferred from the sig_shape argument when present.
    nav_shape : Tuple[int, int], optional
        A nav_shape to apply to the dataset overriding the shape
        value read from the .npy header, by default None. This can
        be used to read a subset of the .npy file, or reshape the
        contained data. Frames are read in C-order from the beginning
        of the file.
    sig_shape : Tuple[int, int], optional
        A sig_shape to apply to the dataset overriding the shape
        value read from the .npy header, by default None. Pixels are
        read in C-order from the beginning of the file.
    sync_offset : int, optional, by default 0
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start
    io_backend : IOBackend, optional
        The I/O backend to use, see :ref:`io backends`, by default None.

    Raises
    ------
    DataSetException
        If sig_dims is not an integer and cannot be inferred from sig_shape
    DataSetException
        If the supplied nav_shape + sig_shape describe an array larger
        than the contents of the .npy file
    DataSetException
        If the .npy file is Fortran-ordered
    """
    def __init__(
        self,
        path: str,
        sig_dims: Optional[int] = 2,
        nav_shape: Optional[Tuple[int, int]] = None,
        sig_shape: Optional[Tuple[int, int]] = None,
        sync_offset: int = 0,
        io_backend: Optional[IOBackend] = None,
    ):
        super().__init__(io_backend=io_backend)
        self._meta = None
        self._nav_shape = tuple(nav_shape) if nav_shape else nav_shape
        self._sig_shape = tuple(sig_shape) if sig_shape else sig_shape
        self._sig_dims = sig_dims
        if self._sig_shape is not None:
            if self._sig_dims is None:
                self._sig_dims = len(self._sig_shape)
            if len(self._sig_shape) != self._sig_dims:
                raise DataSetException(f'Mismatching sig_dims (= {self._sig_dims}) '
                                       f'and sig_shape {self._sig_shape} arguments')
        if self._sig_dims is None or not isinstance(self._sig_dims, int):
            raise DataSetException('Must supply one of sig_dims or sig_shape to NPYDataSet')
        self._path = path
        self._sync_offset = sync_offset
        self._npy_info: typing.Optional[NPYInfo] = None

    def _get_filesize(self):
        return os.stat(self._path).st_size

    def initialize(self, executor) -> "DataSet":
        self._filesize = executor.run_function(self._get_filesize)
        npyinfo = executor.run_function(read_npy_info, self._path)
        self._npy_info = npyinfo
        np_shape = Shape(npyinfo.shape, sig_dims=self._sig_dims)
        sig_shape = self._sig_shape if self._sig_shape else np_shape.sig
        nav_shape = self._nav_shape if self._nav_shape else np_shape.nav
        shape = Shape(tuple(nav_shape) + tuple(sig_shape), sig_dims=self._sig_dims)
        # Trying to follow implementation of RawFileDataSet i.e. the _image_count
        # is the whole block of data interpreted as N frames of sig_shape, noting that
        # here sig_shape can be either user-supplied or from the npy metadata
        # if prod(sig_shape) is not a factor then bytes will be dropped at the end
        self._image_count = np_shape.size // prod(sig_shape)
        self._nav_shape_product = shape.nav.size
        self._sync_offset_info = self.get_sync_offset_info()
        self._meta = DataSetMeta(
            shape=shape,
            raw_dtype=np.dtype(npyinfo.dtype),
            sync_offset=self._sync_offset or 0,
            image_count=self._image_count,
        )
        return self

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    @classmethod
    def detect_params(cls, path, executor):
        try:
            npy_info = executor.run_function(read_npy_info, path)
            # FIXME: assumption about number of sig dims
            shape = Shape(npy_info.shape, sig_dims=2)
            return {
                "parameters": {
                    "path": path,
                    "nav_shape": tuple(shape.nav),
                    "sig_shape": tuple(shape.sig),
                },
                "info": {
                    "image_count": int(prod(shape.nav)),
                    "native_sig_shape": tuple(shape.sig),
                }
            }
        except Exception as e:
            print(e)
            return False

    @classmethod
    def get_supported_extensions(cls):
        return {"npy"}

    @classmethod
    def get_msg_converter(cls):
        return NPYDatasetParams

    def _get_fileset(self):
        assert self._npy_info is not None
        return FileSet([
            File(
                path=self._path,
                start_idx=0,
                end_idx=self._meta.image_count,
                sig_shape=self.shape.sig,
                native_dtype=self._meta.raw_dtype,
                file_header=self._npy_info.offset,
            )
        ])

    def check_valid(self):
        try:
            fileset = self._get_fileset()
            backend = self.get_io_backend().get_impl()
            with backend.open_files(fileset):
                return True
        except (OSError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def get_cache_key(self):
        return {
            "path": self._path,
            # nav_shape + sig_shape; included because changing nav_shape will change
            # the partition structure and cause errors
            "shape": tuple(self.shape),
            "dtype": str(self.dtype),
            "sync_offset": self._sync_offset,
        }

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
        return f"<NPYDataSet of {self.dtype} shape={self.shape}>"
