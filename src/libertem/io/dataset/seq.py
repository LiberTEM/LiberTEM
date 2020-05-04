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

import numpy as np
from libertem.io.dataset.base import (
    FileSet, DataSet, BasePartition, DataSetMeta, DataSetException,
    LocalFile,
)
from libertem.common import Shape


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


def _read_header(path, fields, offset=0):
    with open(path, "rb") as _file:
        _file.seek(offset)
        tmp = dict()
        for name, format in fields:
            val = _unpack_header(_file, format)
            tmp[name] = val

    return tmp


def _unpack_header(_file, fs, offset=None):
    if offset is not None:
        _file.seek(offset)
    s = struct.Struct('<' + fs)
    vals = s.unpack(_file.read(s.size))
    if len(vals) == 1:
        return vals[0]
    else:
        return vals


class SEQFileSet(FileSet):
    pass


class SEQDataSet(DataSet):
    def __init__(self, path, scan_size):
        self._path = path
        self._scan_size = scan_size
        self._header = None
        self._meta = None
        self._footer_size = None
        self._filesize = None
        self._image_count = None

    def _do_initialize(self):
        header = self._header = _read_header(self._path, HEADER_FIELDS)
        if header['version'] >= 5:  # StreamPix version 6
            self._image_offset = 8192
            # Timestamp = 4-byte unsigned long + 2-byte unsigned short (ms)
            #   + 2-byte unsigned short (us)
            self._timestamp_micro = True
        else:  # Older versions
            self._image_offset = 1024
            self._timestamp_micro = False
        try:
            dtype = np.dtype('uint%i' % header['bit_depth'])
        except TypeError:
            raise DataSetException("unsupported bit depth: %s" % header['bit_depth'])
        frame_size_bytes = header['width'] * header['height'] * dtype.itemsize
        self._footer_size = header['true_image_size'] - frame_size_bytes
        self._filesize = os.stat(self._filename).st_size
        self._image_count = int(
            (self._filesize - self._image_offset) / header['true_image_size']
        )
        if self._image_count != np.prod(self._scan_size):
            raise DataSetException("scan_size doesn't match number of frames")
        shape = Shape(
            (self._scan_size,) + (header['height'], header['width']),
            sig_dims=2,
        )
        self._meta = DataSetMeta(
            shape=shape,
            raw_dtype=dtype,
            dtype=dtype,
            metadata=header,
        )

    def initialize(self, executor):
        return executor.run_function(self._do_initialize)

    @property
    def meta(self):
        return self._meta

    @classmethod
    def detect_params(cls, path, executor):
        # TODO: check the MAGIC
        raise NotImplementedError()

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
                file_header=0,
            )
        ])

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
        }

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        res = max(self._cores, self._filesize // (512*1024*1024))
        return res

    def get_partitions(self):
        slices = BasePartition.make_slices(
            shape=self.shape,
            num_partitions=self._get_num_partitions(),
        )
        fileset = self._get_fileset()
        for part_slice, start, stop in slices:
            yield BasePartition(
                meta=self._meta,
                fileset=fileset.get_for_range(start, stop),
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
            )

    def __repr__(self):
        return "<SEQDataSet of %s shape=%s>" % (self.dtype, self.shape)
