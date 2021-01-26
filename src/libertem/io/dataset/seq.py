# Based on the SEQ reader of the PIMS project, with the following license:
#
#    Copyright (c) 2013-2014 PIMS contributors
#    https://github.com/soft-matter/pims
#    All rights reserved
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the soft-matter organization nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import struct
import warnings
from typing import Tuple

import numpy as np
from ncempy.io.mrc import mrcReader

from libertem.common import Shape
from libertem.web.messages import MessageConverter
from libertem.io.dataset.base import (
    FileSet, DataSet, BasePartition, DataSetMeta, DataSetException,
    LocalFile,
)
from libertem.corrections import CorrectionSet


DWORD = 'L'
LONG = 'l'
DOUBLE = 'd'
USHORT = 'H'


HEADER_FIELDS = [
    ('magic', DWORD),
    ('name', '24s'),
    ('version', LONG),
    ('header_size', LONG),
    ('description', '512s'),
    ('width', DWORD),
    ('height', DWORD),
    ('bit_depth', DWORD),
    ('bit_depth_real', DWORD),
    ('image_size_bytes', DWORD),
    ('image_format', DWORD),
    ('allocated_frames', DWORD),
    ('origin', DWORD),
    ('true_image_size', DWORD),
    ('suggested_frame_rate', DOUBLE),
    ('description_format', LONG),
    ('reference_frame', DWORD),
    ('fixed_size', DWORD),
    ('flags', DWORD),
    ('bayer_pattern', LONG),
    ('time_offset_us', LONG),
    ('extended_header_size', LONG),
    ('compression_format', DWORD),
    ('reference_time_s', LONG),
    ('reference_time_ms', USHORT),
    ('reference_time_us', USHORT)
    # More header values not implemented
]

HEADER_SIZE = sum([
    struct.Struct('<' + field[1]).size
    for field in HEADER_FIELDS
])


def _read_header(path, fields):
    with open(path, "rb") as fh:
        fh.seek(0)
        return _unpack_header(fh.read(HEADER_SIZE), fields)


def _unpack_header(header_bytes, fields):
    str_fields = {"name", "description"}
    tmp = dict()
    pos = 0
    for name, fmt in fields:
        header_part = header_bytes[pos:]
        val, size = _unpack_header_field(header_part, fmt)
        if name in str_fields:
            val = _decode_str(val)
        tmp[name] = val
        pos += size

    return tmp


def _unpack_header_field(header_part, fs):
    s = struct.Struct('<' + fs)
    vals = s.unpack(header_part[:s.size])
    if len(vals) == 1:
        return vals[0], s.size
    else:
        return vals, s.size


def _decode_str(str_in):
    end_pos = len(str_in)
    end_mark = b'\x00\x00'
    if end_mark in str_in:
        end_pos = str_in.index(end_mark) + 1
    return str_in[:end_pos].decode("utf16")


def _get_image_offset(header):
    if header['version'] >= 5:
        return 8192
    else:
        return 1024


class SEQDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/SEQDatasetParams.schema.json",
        "title": "SEQDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "SEQ"},
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
        },
        "required": ["type", "path", "nav_shape"],
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path", "nav_shape"]
        }
        if "sig_shape" in raw_data:
            data["sig_shape"] = tuple(raw_data["sig_shape"])
        if "sync_offset" in raw_data:
            data["sync_offset"] = raw_data["sync_offset"]
        return data


class SEQFileSet(FileSet):
    pass


class SEQDataSet(DataSet):
    """
    Read data from Norpix SEQ files.

    Examples
    --------

    >>> ds = ctx.load("seq", path="/path/to/file.seq", nav_shape=(1024, 1024))  # doctest: +SKIP

    Parameters
    ----------
    path
        Path to the .seq file

    nav_shape: tuple of int
        A n-tuple that specifies the size of the navigation region ((y, x), but
        can also be of length 1 for example for a line scan, or length 3 for
        a data cube, for example)

    sig_shape: tuple of int, optional
        Signal/detector size (height, width)

    sync_offset: int, optional
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start
    """
    def __init__(self, path: str, scan_size: Tuple[int] = None, nav_shape: Tuple[int] = None,
                 sig_shape: Tuple[int] = None, sync_offset: int = 0, io_backend=None):
        super().__init__(io_backend=io_backend)
        self._path = path
        self._nav_shape = tuple(nav_shape) if nav_shape else nav_shape
        self._sig_shape = tuple(sig_shape) if sig_shape else sig_shape
        self._sync_offset = sync_offset
        # handle backwards-compatability:
        if scan_size is not None:
            warnings.warn(
                "scan_size argument is deprecated. please specify nav_shape instead",
                FutureWarning
            )
            if nav_shape is not None:
                raise ValueError("cannot specify both scan_size and nav_shape")
            self._nav_shape = tuple(scan_size)
        if self._nav_shape is None:
            raise TypeError("missing 1 required argument: 'nav_shape'")
        self._header = None
        self._meta = None
        self._footer_size = None
        self._filesize = None
        self._dark = None
        self._gain = None

    def _do_initialize(self):
        header = self._header = _read_header(self._path, HEADER_FIELDS)
        self._image_offset = _get_image_offset(header)
        if header['version'] >= 5:  # StreamPix version 6
            # Timestamp = 4-byte unsigned long + 2-byte unsigned short (ms)
            #   + 2-byte unsigned short (us)
            self._timestamp_micro = True
        else:  # Older versions
            self._timestamp_micro = False
        try:
            dtype = np.dtype('uint%i' % header['bit_depth'])
        except TypeError:
            raise DataSetException("unsupported bit depth: %s" % header['bit_depth'])
        frame_size_bytes = header['width'] * header['height'] * dtype.itemsize
        self._footer_size = header['true_image_size'] - frame_size_bytes
        self._filesize = os.stat(self._path).st_size
        self._image_count = int((self._filesize - self._image_offset) / header['true_image_size'])

        if self._sig_shape is None:
            self._sig_shape = tuple((header['height'], header['width']))
        elif int(np.prod(self._sig_shape)) != (header['height'] * header['width']):
            raise DataSetException(
                "sig_shape must be of size: %s" % (header['height'] * header['width'])
            )

        self._nav_shape_product = int(np.prod(self._nav_shape))
        self._sync_offset_info = self.get_sync_offset_info()
        shape = Shape(self._nav_shape + self._sig_shape, sig_dims=len(self._sig_shape))

        self._meta = DataSetMeta(
            shape=shape,
            raw_dtype=dtype,
            dtype=dtype,
            metadata=header,
            sync_offset=self._sync_offset,
            image_count=self._image_count,
        )
        self._maybe_load_dark_gain()
        return self

    def _maybe_load_mrc(self, path):
        if not os.path.exists(path):
            return None
        data_dict = mrcReader(path)
        return np.squeeze(data_dict['data'])

    def _maybe_load_dark_gain(self):
        self._dark = self._maybe_load_mrc(self._path + ".dark.mrc")
        self._gain = self._maybe_load_mrc(self._path + ".gain.mrc")

    def get_correction_data(self):
        return CorrectionSet(
            dark=self._dark,
            gain=self._gain,
        )

    def initialize(self, executor):
        return executor.run_function(self._do_initialize)

    def get_diagnostics(self):
        header = self._header
        return [
            {"name": k, "value": str(v)}
            for k, v in header.items()
        ] + [
            {"name": "Footer size",
             "value": str(self._footer_size)},
            {"name": "Dark frame included",
             "value": str(self._dark is not None)},
            {"name": "Gain map included",
             "value": str(self._gain is not None)},
        ]

    @property
    def meta(self):
        return self._meta

    @classmethod
    def get_msg_converter(cls):
        return SEQDatasetParams

    @classmethod
    def detect_params(cls, path, executor):
        try:
            header = _read_header(path, HEADER_FIELDS)
            if header['magic'] != 0xFEED:
                return False
            image_offset = _get_image_offset(header)
            filesize = os.stat(path).st_size
            image_count = int(
                (filesize - image_offset) / header['true_image_size']
            )
            sig_shape = tuple((header['height'], header['width']))
            return {
                "parameters": {
                    "path": path,
                    "nav_shape": tuple((image_count,)),
                    "sig_shape": sig_shape,
                },
                "info": {
                    "image_count": image_count,
                    "native_sig_shape": sig_shape,
                }
            }
        except (OSError, UnicodeDecodeError):
            return False

    @classmethod
    def get_supported_extensions(cls):
        return set(["seq"])

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    def _get_fileset(self):
        return SEQFileSet(files=[
            LocalFile(
                path=self._path,
                start_idx=0,
                end_idx=self._image_count,
                native_dtype=self._meta.raw_dtype,
                sig_shape=self._meta.shape.sig,
                frame_footer=self._footer_size,
                frame_header=0,
                file_header=self._image_offset,
            )
        ], frame_footer_bytes=self._footer_size)

    def check_valid(self):
        if self._header['magic'] != 0xFEED:
            raise DataSetException('The format of this .seq file is unrecognized')
        if self._header['compression_format'] != 0:
            raise DataSetException('Only uncompressed images are supported in .seq files')
        if self._header['image_format'] != 100:
            raise DataSetException('Non-monochrome images are not supported')

    def get_cache_key(self):
        return {
            "path": self._path,
            "shape": self.shape,
            "sync_offset": self._sync_offset,
        }

    def get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        res = max(self._cores, self._filesize // (512*1024*1024))
        return res

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
        return "<SEQDataSet of %s shape=%s>" % (self.dtype, self.shape)
