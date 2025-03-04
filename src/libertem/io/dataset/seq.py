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
from typing import Optional

import defusedxml.ElementTree as ET
import numpy as np
import sparse
from ncempy.io.mrc import mrcReader

from libertem.common.math import prod, make_2D_square
from libertem.common import Shape
from libertem.common.messageconverter import MessageConverter
from libertem.io.dataset.base import (
    FileSet, DataSet, BasePartition, DataSetMeta, DataSetException,
    File, IOBackend,
)
from libertem.io.corrections import CorrectionSet

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

HEADER_SIZE = sum(
    struct.Struct('<' + field[1]).size
    for field in HEADER_FIELDS
)


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


def xml_map_sizes(bad_pixel_maps):
    """
    returns the sizes and binning values of bad_pixel_maps

    Parameters
    ----------
    bad_pixel_maps: node
        all of the bad pixel map nodes from the root
    """
    map_sizes = []

    for size_map in bad_pixel_maps:
        if "Binning" in size_map.attrib:
            map_sizes.append((int(size_map.attrib['Columns']), int(size_map.attrib['Rows']),
                              int(size_map.attrib["Binning"])))
        else:
            map_sizes.append((int(size_map.attrib['Columns']), int(size_map.attrib['Rows']),
                              int(size_map.attrib.get("Binning", 1))))
    map_rearrange = zip(*map_sizes)
    xy_map_sizes = list(map_rearrange)
    return xy_map_sizes, map_sizes


def xml_unbinned_map_maker(xy_map_sizes):
    """
    returns two list of unbinned sizes(rows and cols) if the size was binned
    return 0 in the list in its place
    Parameters
    ----------
    xy_map_sizes: list of tuple
        bad pixel maps sizes separated by row and col
    """
    unbinned_x = []
    unbinned_y = []
    for map_ind in range(0, len(xy_map_sizes[0])):
        if xy_map_sizes[2][map_ind] < 2:
            unbinned_y.append(xy_map_sizes[0][map_ind])
            unbinned_x.append(xy_map_sizes[1][map_ind])
        else:
            unbinned_y.append(0)
            unbinned_x.append(0)
    return unbinned_x, unbinned_y


def xml_binned_map_maker(xy_map_sizes):
    """
    returns two list of binned sizes(rows and cols) if the size was not binned
    return 0 in the list in its place
    Parameters
    ----------
    xy_map_sizes: list of tuple
        bad pixel maps sizes separated by row and col
    """
    unbinned_x = []
    unbinned_y = []
    for map_ind in range(0, len(xy_map_sizes[0])):
        if xy_map_sizes[2][map_ind] > 1:
            unbinned_y.append(xy_map_sizes[0][map_ind])
            unbinned_x.append(xy_map_sizes[1][map_ind])
        else:
            unbinned_y.append(0)
            unbinned_x.append(0)
    return unbinned_x, unbinned_y


def xml_map_index_selector(used_y):
    """
    returns the bad pixel maps  index with the largest column count

    Parameters
    ----------
    used_y: list
        list of either binned or unbinned y coordinates
    """
    max_y = max(used_y)

    map_index = used_y.index(max_y)
    return map_index


def xml_defect_coord_extractor(bad_pixel_map, map_index, map_sizes):
    """
    returns the chosen bad pixel map's defects

    Parameters
    ----------
    bad_pixel_map: node
        the xml file's root node
    map_index: int
        the index of the correct bad pixel map
    map_sizes: list of tuples
    """
    excluded_rows = []
    excluded_cols = []
    excluded_pixels = []
    for defect in bad_pixel_map.findall('Defect'):
        if len(defect.attrib) == 1:
            defect_attrib_key = defect.attrib.keys()
            if "Rows" in defect_attrib_key:
                split = defect.attrib["Rows"].split('-')
                excluded_rows.append(split)
            if "Row" in defect_attrib_key:
                excluded_rows.append([defect.attrib["Row"]])

            if "Columns" in defect_attrib_key:
                split = defect.attrib["Columns"].split('-')
                excluded_cols.append(split)
            if "Column" in defect_attrib_key:
                excluded_cols.append([defect.attrib["Column"]])
        else:
            excluded_pixels.append([defect.attrib["Column"], defect.attrib["Row"]])

    return {"rows": excluded_rows,
            "cols": excluded_cols,
            "pixels": excluded_pixels,
            "size": (map_sizes[map_index][0], map_sizes[map_index][1])
            }


def xml_defect_data_extractor(root, metadata):
    """
    Parameters
    ----------
    root: node
        the xml file's root node
    metadata: dictionary
        the content of the metadata file as a dictionary
    """
    bad_pixel_maps = root.findall('.//BadPixelMap')
    xy_size, map_sizes = xml_map_sizes(bad_pixel_maps)
    if (metadata['HardwareBinning']) < 2:
        used_x, used_y = xml_unbinned_map_maker(xy_map_sizes=xy_size)
    else:
        used_x, used_y = xml_binned_map_maker(xy_map_sizes=xy_size)
    map_index = xml_map_index_selector(used_y)
    defect_dict = xml_defect_coord_extractor(bad_pixel_maps[map_index], map_index, map_sizes)
    return defect_dict


def bin_array2d(a, binning):
    """
    Parameters
    ----------
    a: np array
    binning: int
        the times we want to bin the array
    """
    sx, sy = a.shape
    sxc = sx // binning * binning
    syc = sy // binning * binning
    # crop:
    ac = a[:sxc, :syc]
    return ac.reshape(ac.shape[0] // binning, binning,
                      ac.shape[1] // binning, binning).sum(3).sum(1)


def array_cropping(arr, start_size, req_size, offsets):
    """
    returns a crop from the original array
    Parameters
    ----------
    arr: np array
        the array we will make the changes on
    start_size: tuple
        the size of the original array
    req_size: tuple
        the size we want to crop to
    offsets: tuple
        the top left coord of the crop
    """
    ac = arr
    if offsets[0] + req_size[0] <= start_size[0] and offsets[1] + req_size[1] <= start_size[1]:
        req_y = int(req_size[0]) // 2
        req_x = int(req_size[1]) // 2
        a = int(offsets[0]) + req_y
        b = int(offsets[1]) + req_x
        ac = ac[(a - req_y):(a + req_y), (b - req_x):(b + req_x)]
    return ac


def xml_generate_map_size(exc_rows, exc_cols, exc_pix, size, metadata):
    """
    This function will be responsible for generating new size based on the parameters.
    returns an np array with the excluded pixels as True

    Parameters
    ----------
    exc_rows: list of lists
        list with the excluded rows as lists which length is 1 if its a single row
        and two if its an interval of rows
    exc_cols: list of lists
    exc_pix: list of lists
    size: tuple
        selected bad pixel maps size
    metadata: dictionary
        the metadata where we specify the image's parameters
    """
    required_size = (metadata["UnbinnedFrameSizeY"], metadata["UnbinnedFrameSizeX"])
    offsets = (metadata["OffsetY"], metadata["OffsetX"])
    bin_value = metadata["HardwareBinning"]
    if bin_value > 1:
        required_size = (required_size[0]//2, required_size[1]//2)
        offsets = (offsets[0]//2, offsets[1]//2)
    dummy_m = np.zeros(size, dtype=bool)
    for row in exc_rows:

        if len(row) == 1:
            dummy_m[int(row[0])] = 1
        else:
            dummy_m[int(row[0]):int(row[1]) + 1] = 1

    for col in exc_cols:

        if len(col) == 1:
            dummy_m[:, int(col[0])] = 1
        else:
            dummy_m[:, int(col[0]):int(col[1]) + 1] = 1

    for pix in exc_pix:
        dummy_m[int(pix[1]), int(pix[0])] = 1

    cropped = array_cropping(dummy_m, start_size=size, req_size=required_size, offsets=offsets)
    return np.array(cropped, dtype=bool)


def xml_processing(tree, metadata_dict):
    data_dict = xml_defect_data_extractor(tree, metadata_dict)
    coord = xml_generate_map_size(data_dict["rows"], data_dict["cols"], data_dict["pixels"],
                                  data_dict["size"], metadata_dict)
    return sparse.COO(coord)


def _load_xml_from_string(xml, metadata):
    tree = ET.fromstring(xml)
    return xml_processing(tree, metadata)


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
            "io_backend": {
                "enum": IOBackend.get_supported(),
            }
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
        If positive, number of frames to skip from start.
        If negative, number of blank frames to insert at start.
    num_partitions: int, optional
        Override the number of partitions. This is useful if the
        default number of partitions, chosen based on common workloads,
        creates partitions which are too large (or small) for the UDFs
        being run on this dataset.

    Note
    ----

    Dark and gain reference are loaded from MRC files with the same root as the
    SEQ file and the extensions :code:`.dark.mrc` and :code:`.gain.mrc`, i.e.
    :code:`/path/to/file.dark.mrc` and :code:`/path/to/file.gain.mrc` if they are present.

    .. versionadded:: 0.8.0

    Dead pixels are read from an XML file with the same root as the
    SEQ file and the extension :code:`.Config.Metadata.xml`, i.e.
    :code:`/path/to/file.Config.Metadata.xml` in the above example if both this file and
    :code:`/path/to/file.metadata` are present.

    See :ref:`corrections` for more information on how to change or disable corrections.

    FIXME find public documentation of the XML format and dark/gain maps.
    """

    def __init__(
        self,
        path: str,
        scan_size: Optional[tuple[int, ...]] = None,
        nav_shape: Optional[tuple[int, ...]] = None,
        sig_shape: Optional[tuple[int, ...]] = None,
        sync_offset: int = 0,
        io_backend=None,
        num_partitions=None,
    ):
        super().__init__(
            io_backend=io_backend,
            num_partitions=num_partitions,
        )
        self._path = path
        # There might be '.seq.seq' and '.seq' in the wild
        # See https://github.com/LiberTEM/LiberTEM/issues/1120
        # We first try if '.seq.seq' matches, then '.seq'
        name, ext = os.path.splitext(path)
        name2, ext2 = os.path.splitext(name)
        if ext.lower() == '.seq' and ext2.lower() == '.seq':
            self._basename = name2
        elif ext.lower() == '.seq':
            self._basename = name
        else:
            self._basename = path
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
        self._excluded_pixels = None

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
        elif int(prod(self._sig_shape)) != (header['height'] * header['width']):
            raise DataSetException(
                "sig_shape must be of size: %s" % (header['height'] * header['width'])
            )

        self._nav_shape_product = int(prod(self._nav_shape))
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

    def _load_xml_from_file(self):
        xml_path = self._basename + ".seq.Config.Metadata.xml"
        meta_path = self._basename + ".seq.metadata"
        if not os.path.exists(xml_path):
            return None
        if not os.path.exists(meta_path):
            return None
        else:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            with open(meta_path, mode="rb") as file:
                met = file.read()
            metdata_keys = ['DEMetadataSize', 'DEMetadataVersion', 'UnbinnedFrameSizeX',
                            'UnbinnedFrameSizeY', 'OffsetX', 'OffsetY', 'HardwareBinning',
                            'Bitmode', 'FrameRate', 'RotationMode',
                            'FlipMode', 'OkraMode']
            metadata = dict(zip(metdata_keys, struct.unpack_from('iiiiiiiiiii?', met, 282)))
            return xml_processing(root, metadata)

    def _maybe_load_dark_gain(self):
        self._dark = self._maybe_load_mrc(self._basename + ".seq.dark.mrc")
        self._gain = self._maybe_load_mrc(self._basename + ".seq.gain.mrc")
        self._excluded_pixels = self._load_xml_from_file()

    def get_correction_data(self):
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
                    "nav_shape": make_2D_square((image_count,)),
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
        return {"seq"}

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    def _get_fileset(self):
        return SEQFileSet(files=[
            File(
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
        return f"<SEQDataSet of {self.dtype} shape={self.shape}>"
