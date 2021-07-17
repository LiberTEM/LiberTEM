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
import sparse
from ncempy.io.mrc import mrcReader
import xml.etree.ElementTree as ET
import math

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
        self._excluded_pixels=None

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
        self._excluded_pixels=self._maybe_load_xml(self._path + ".Config.Metadata.xml")

    def cropping(self, a, start_size, req_size):
        '''

        :param a: an array we want to appply the cropping to
        :param start_size: the original size of image
        :param req_size: equals with the signal shape, this is the size we want to crop to
        :return: the middle of 'a' with the size req_size, 2d array
        '''
        if start_size > req_size and math.log(start_size, 2).is_integer() and math.log(req_size, 2).is_integer():
            rep = math.log2(start_size) - math.log2(req_size)
        else:
            return []
        ac = a
        for i in range(0, int(rep)):
            quarter = ac.shape[0] // 4
            ac = ac[quarter:(3 * quarter), quarter:(3 * quarter)]

        return ac

    def _maybe_load_xml(self, path):
        if not os.path.exists(self._path):
            return None
        tree = ET.parse(path)
        root = tree.getroot()
        num_of_cat = len(root[2])  # the number of sizes (2048,4096....)
        num_of_Rowz = []  # the num of rows in different category
        num_of_Colz = []  # the number of cols in different category
        num_of_Pixels = []  # sort the number of pixels in different categories
        mop = {}  # dummy dictionary to store the elements of root[2] wich are the BadPixelMaps
        for z in root[2]:
            mop[z] = z.attrib
        Rowz = []
        Colz = []
        Pixels = []  # store the elements in [row, col] format
        row_counter = 0
        col_counter = 0
        pix_counter = 0
        coo_shape_x = []  # the list that contains information regarding the shape of the sparse.COO array, if u pick e.g. the 0. index element and it has the value of 4096
        # that means u will also pick the 0. index of the rows, cols and pixels list
        coo_shape_y = []
        coo_bin_val = []  # the binning values cateorized(by inedex)
        '''
        the program can search for the appropriate data based on the position of the sig_shape's value
        in this matrix, later it will use it's index to get the rows, cols and pixels's values
         '''

        for b in mop:
            chk = 0

            coo_shape_x.append(int(b.attrib['Rows']))

            coo_shape_y.append(int(b.attrib['Columns']))
            if 'Binning' in b.attrib:
                coo_bin_val.append([int(b.attrib['Binning'])])
                chk += 1
            elif (len(b.attrib) == 2) and chk == 0:
                coo_bin_val.append([])
            for c in b:

                tmp_cnt = 0  # to determine if we are dealing with a single pixel
                tmp_dict = {}
                tmp_dict.update(c.attrib)  # still unsplitted dictionary
                tmp_block = []
                for i in tmp_dict:
                    # just to convert, but sometimes (in case of pixels) it can be used to iterate throught that dict too meaning we shuold use a counter to determine whether
                    # the current inspected element is a pixel or just simply a row or col
                    tmp = ''

                    if (len(tmp_dict) == 2):
                        tmp = tmp_dict[i]
                        tmp_block.append(tmp)

                        tmp_cnt += 1
                        if tmp_cnt == 2:
                            Pixels.append(tmp_block)
                            pix_counter += 1
                            tmp_cnt = 0
                            break
                    else:
                        if (i == 'Rows') or (i == 'Row'):
                            tmp = tmp_dict[i]
                            Rowz.append(tmp.split('-'))
                            row_counter += 1

                        if (i == 'Columns') or (i == 'Column'):
                            tmp = tmp_dict[i]
                            Colz.append(tmp.split('-'))
                            col_counter += 1
            num_of_Rowz.append(row_counter)
            num_of_Colz.append(col_counter)
            num_of_Pixels.append(pix_counter)
            row_counter = 0
            col_counter = 0
            pix_counter = 0

        # splitting the results by category
        rows_by_category = {}
        cols_by_category = {}
        pixels_by_category = {}
        end_id = 0
        str_ind = 0
        end_id_col = 0
        str_id_col = 0
        end_id_pixel = 0
        str_id_pixel = 0
        for x in range(0, num_of_cat):
            if (x == 0):
                rows_by_category[x] = Rowz[0:num_of_Rowz[x]]
                cols_by_category[x] = Colz[0:num_of_Colz[x]]
                pixels_by_category[x] = Pixels[0:num_of_Pixels[x]]

                end_id = num_of_Rowz[x]
                str_ind = num_of_Rowz[x]
                end_id_col = num_of_Colz[x]
                str_id_col = num_of_Colz[x]
                end_id_pixel = num_of_Pixels[x]
                str_id_pixel = num_of_Pixels[x]
            else:
                end_id += num_of_Rowz[x]
                end_id_col += num_of_Colz[x]
                end_id_pixel += num_of_Pixels[x]

                rows_by_category[x] = Rowz[str_ind:end_id]
                cols_by_category[x] = Colz[str_id_col:end_id_col]
                pixels_by_category[x] = Pixels[str_id_pixel:end_id_pixel]

                str_ind += num_of_Rowz[x]
                str_id_col += num_of_Colz[x]
                str_id_pixel += num_of_Pixels[x]

        Defect_ID = 0  # determine wich index should be used for further calculations
        if (self._sig_shape[0] in coo_shape_x):
            for index in coo_shape_x:
                Defect_ID += 1
                if int(index) == int(self._sig_shape[0]):
                    break
        else:
            Defect_ID = num_of_cat + 1
            coo_bin_val.append([])
            if (len(coo_bin_val[0]) == 0):

                dummy_transformation_m = np.zeros((coo_shape_x[0], coo_shape_y[0]))
                for i in rows_by_category[0]:
                    if len(i) == 2:
                        dummy_transformation_m[int(i[0]):int(i[1]) + 1] = 2
                    if len(i) == 1:
                        dummy_transformation_m[int(i[0])] = 2

                res2 = []
                c = self.cropping(dummy_transformation_m, coo_shape_x[0], self._sig_shape[0])
                for a in range(0, c.shape[0]):
                    for b in range(0, c.shape[1]):
                        if (c[a, b] > 1):
                            if [a] not in res2:
                                res2.append([a])
                dummy_transformation_m = np.zeros((coo_shape_x[0], coo_shape_y[0]))
                for i in cols_by_category[0]:
                    if (len(i) == 2):
                        dummy_transformation_m[:, int(i[0]):(int(i[1]) + 1)] = 2
                    if len(i) == 1:
                        dummy_transformation_m[:, int(i[0])] = 2

                res3 = []
                c = self.cropping(dummy_transformation_m, coo_shape_x[0], self._sig_shape[0])
                for a in range(0, c.shape[0]):
                    for b in range(0, c.shape[1]):
                        if (c[a, b] > 1):
                            if [b] not in res3 and [a] not in res2:
                                res3.append([b])

                dummy_transformation_m = np.zeros((coo_shape_x[0], coo_shape_y[0]))
                for i in pixels_by_category[0]:
                    dummy_transformation_m[int(i[0]), int(i[1])] = 2

                res4 = []
                c = self.cropping(dummy_transformation_m, coo_shape_x[0], self._sig_shape[0])
                for a in range(0, c.shape[0]):
                    for b in range(0, c.shape[1]):
                        if (c[a, b] > 1):
                            if [a, b] not in res4 and [a] not in res2 and [b] not in res3:
                                res4.append([a, b])

                rows_by_category[Defect_ID - 1] = res2
                cols_by_category[Defect_ID - 1] = res3
                pixels_by_category[Defect_ID - 1] = res4
                Rowz.append(len(rows_by_category[Defect_ID - 1]))
                Colz.append(len(cols_by_category[Defect_ID - 1]))
                Pixels.append(len(pixels_by_category[Defect_ID - 1]))
                coo_shape_x.append(self._sig_shape[0])
                coo_shape_y.append(self._sig_shape[1])
        size = int(coo_shape_x[Defect_ID - 1])
        coords = np.zeros((int(self._sig_shape[0]), int(self._sig_shape[1])))

        for i1 in rows_by_category:
            if (len(rows_by_category[i1]) != 0):
                if i1 == Defect_ID - 1:

                    for i2 in rows_by_category[i1]:
                        if (len(i2) == 2):  # if its a list of rows in a [from,to] form
                            start_in = int(i2[0])
                            end_in = (int(i2[1]) + 1)
                            coords[start_in:end_in] = 1

                        if (len(i2) == 1):  # if its just a single row

                            tmp = [int(i2[0])]
                            coords[tmp] = 1
                    break

        for i1 in cols_by_category:
            if (len(cols_by_category[i1]) != 0):

                if i1 == Defect_ID - 1:
                    for i2 in cols_by_category[i1]:

                        if (len(i2) == 2):  # if its a list of cols in a [from,to] form
                            start_in = int(i2[0])
                            end_in = (int(i2[1]) + 1)

                            coords[:, start_in:end_in] = 1

                        if (len(i2) == 1):  # if its just a single col

                            tmp = [int(i2[0])]
                            coords[:, tmp] = 1
                    break

        for i1 in cols_by_category:
            if (len(pixels_by_category[i1]) != 0):

                if i1 == Defect_ID - 1:
                    for i2 in pixels_by_category[i1]:
                        coords[int(i2[0]), int(i2[1])] = 1

                    break
        return sparse.COO(coords)


    def get_correction_data(self):
        exclu=np.zeros((self._sig_shape[0],self._sig_shape[1]))
        for i in range (0, self._sig_shape[0]):
            exclu[775][i] = 1
            exclu[776][i] = 1


        return CorrectionSet(
            dark=self._dark,
            gain=self._gain,
            excluded_pixels=self._excluded_pixels,
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
    def _do_detect_params(cls, path):
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
    def detect_params(cls, path, executor):
        try:
            return executor.run_function(cls._do_detect_params, path)
        except Exception as e:
            raise DataSetException(repr(e)) from e

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