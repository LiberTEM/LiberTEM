# -*- encoding: utf-8 -*-
import os
import re
import csv
import glob
import logging
import warnings
import configparser

import scipy.io as sio
import numpy as np

from libertem.common import Shape
from libertem.common.buffers import zeros_aligned
from libertem.web.messages import MessageConverter
from .base import (
    DataSet, DataSetException, DataSetMeta,
    File3D, FileSet3D, Partition3D
)

log = logging.getLogger(__name__)
READOUT_MODE_PAT = re.compile(
    r'^"bin:\s*(?P<bin>\d+),\s*windowing:\s*(?P<win_i>\d+)\s*x\s*(?P<win_j>\d+)\s*"$'
)

# file header: 1024 bytes, at the beginning of each file
file_header_dtype = [
    ('header_size', '<u2'),        # fixed value: 1024
    ('frame_header_size', '<u2'),  # fixed value: 64
    ('padding_1', (bytes, 3)),
    ('version', '<u1'),            # fixed value: 6
    ('comment_1', (bytes, 80)),
    ('width', '<u2'),              # -> columns
    ('height', '<u2'),             # -> rows
    ('comment_2', (bytes, 928)),
    ('num_frames', '<u4'),
]

# frame header: 64 bytes, before each frame
frame_header_dtype = [
    ('padding_1', (bytes, 4)),
    ('timestamp_s', '<u4'),
    ('timestamp_us', '<u4'),
    ('frame_number', '<u4'),
    ('padding_2', (bytes, 12)),
    ('comment', (bytes, 36)),
]


class FRMS6DatasetParams(MessageConverter):
    SCHEMA = {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "$id": "http://libertem.org/FRMS6DatasetParams.schema.json",
      "title": "FRMS6DatasetParams",
      "type": "object",
      "properties": {
          "type": {"const": "frms6"},
          "path": {"type": "string"},
      },
      "required": ["type", "path"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path"]
        }
        return data


class GainMapCSVDialect(csv.excel):
    delimiter = ';'


def _unbin(tile_data, factor):
    """
    tile_data should have shape (num_frames, y, x)
    """
    s = tile_data.shape

    # insert a binning dimension:
    tile_data = tile_data.reshape((s[0], s[1], 1, s[2]))
    unbinned = tile_data.repeat(factor, axis=2)
    # FIXME: should we scale the data by the binning factor?
    return unbinned.reshape((s[0], factor * s[1], s[2]))


def _get_base_filename(_path):
    path, ext = os.path.splitext(_path)
    if ext == ".hdr":
        base = path
    elif ext == ".frms6":
        base = re.sub(r'_[0-9]+$', '', path)
    else:
        raise DataSetException("unknown extension: %s" % ext)
    return base


def _read_hdr(fname):
    config = configparser.ConfigParser()
    config.read(fname)
    parsed = {}
    sections = config.sections()
    if 'measurementInfo' not in sections:
        raise DataSetException(
            "measurementInfo missing from .hdr file %s, have: %s" % (
                fname,
                repr(sections),
            )
        )
    msm_info = config['measurementInfo']
    int_fields = {'darkframes', 'dwelltimemicroseconds', 'gain', 'signalframes'}
    for key in msm_info:
        value = msm_info[key]
        if key in int_fields:
            parsed[key] = int(value)
        else:
            parsed[key] = value
    # FIXME: are the dimensions the right way aroud? is there a sample file with a non-square
    # scan region?
    parsed['stemimagesize'] = tuple(int(p) for p in parsed['stemimagesize'].split('x'))
    match = READOUT_MODE_PAT.match(parsed['readoutmode'])
    if match is None:
        raise DataSetException("could not parse readout mode")
    readout_mode = match.groupdict()
    parsed['readoutmode'] = {k: int(v) for k, v in readout_mode.items()}
    return parsed


class FRMS6File(File3D):
    def __init__(self, path, start_idx=None, hdr_info=None):
        self._path = path
        self._header = None
        self._hdr_info = hdr_info
        self._start_idx = start_idx
        super().__init__()

    @property
    def dtype(self):
        return np.dtype("<u2")

    @property
    def header(self):
        if self._header is not None:
            return self._header
        header_raw = np.fromfile(self._path, dtype=file_header_dtype, count=1)
        header = {}
        for field, dtype in file_header_dtype:
            if type(dtype) != str:
                continue
            header[field] = header_raw[field][0]
            # to prevent overflows in following computations:
            if np.dtype(dtype).kind == "u":
                header[field] = int(header[field])
        self._header = header
        return header

    @property
    def global_header(self):
        return self._hdr_info

    def check_valid(self):
        if not (self.header['header_size'] == 1024
                and self.header['frame_header_size'] == 64
                and self.header['version'] == 6):
            return False
        # TODO: file size sanity check?
        return True

    @property
    def num_frames(self):
        if self.header['num_frames'] != 0:
            return self.header['num_frames']
        # older FRMS6 files don't contain the number of frames in the header,
        # so calculate from filesize:
        size = os.stat(self._path).st_size
        header = self.header
        w, h = header['width'], header['height']
        bytes_per_frame = w * h * 2
        num = (size - 1024)
        denum = (bytes_per_frame + 64)
        res = num // denum
        if num % denum != 0:
            raise DataSetException("could not determine number of frames")
        return res

    @property
    def start_idx(self):
        return self._start_idx

    def readinto(self, start, stop, out, crop_to=None):
        # ignore crop_top here, as we don't support it anyways
        # (and it would be very awkward, as the data is still folded here...)
        slice_ = (slice(start, stop),)
        out[:] = self.data[slice_]

    def _get_mmapped_array(self):
        raw_data = np.memmap(self._path, dtype=self.dtype)
        # cut off the file header:
        header_size_px = self.header['header_size'] // self.dtype.itemsize
        frames = raw_data[header_size_px:]

        # TODO: we just throw away the frame headers here
        # TODO: for future work, we may want to validate the stuff in there!
        w, h = self.header['width'], self.header['height']
        num_frames = self.num_frames
        frames_w_headers = frames.reshape((num_frames, w * h + 32))
        frames_wo_headers = frames_w_headers[:, 32:]
        frames_wo_headers = frames_wo_headers.reshape((num_frames, h, w))
        return frames_wo_headers

    @property
    def data(self):
        return self._get_mmapped_array()


class FRMS6FileSet(FileSet3D):
    def __init__(self, files, meta, dark_frame, gain_map):
        """
        Represents all files belonging to a measurement.

        Parameters
        ----------
        files : list of FRMS6File
            full paths of all files, without the file containing dark frames
        meta : DataSetMeta
            dataset metadata
        dark_frame : numpy.ndarray or None
            the raw dark frame (2D, not folded)
        gain_map : numpy.ndarray or None
            gain map (2D, folded)
        """
        self._meta = meta
        self._dark_frame = dark_frame
        self._gain_map = gain_map
        super().__init__(files)

    def read_images_multifile(self, start, stop, out, crop_to=None):
        """
        Read [`start`, `stop`) images from the dataset into `out`

        start, stop: dataset-global indices

        The frames will be pre-processed and the indices may cross file boundaries.

        Pre-processing steps:

        1) convert into float
        2) offset correction (dark frame subtraction)
        3) folding
        4) apply gain map
        5) un-binning
        """

        # 1) conversion to float: happens as we write to this buffer
        raw_buffer = zeros_aligned((out.shape[0],) + tuple(self._meta['raw_frame_size']),
                                   dtype=out.dtype)

        super().read_images_multifile(
            start=start,
            stop=stop,
            out=raw_buffer,
            crop_to=crop_to
        )

        # 2) offset correction:
        if self._dark_frame is not None:
            raw_buffer -= self._dark_frame

        # 3) folding: l(eft) p(art), r(ight) p(art)
        # the right part is folded to below the left part
        # (imagine the bottom-right corner as a hinge)
        half_width = out.shape[2]
        assert out.shape[1] % 2 == 0
        half_height = out.shape[1] // 2

        lp = raw_buffer[..., :half_width]
        rp = raw_buffer[..., half_width:]
        # negative strides to flip both x and y direction:
        rp = rp[:, ::-1, ::-1]

        # 4) apply gain map:
        if self._gain_map is not None:
            gain_half = self._gain_map.shape[0] // 2
            gain_lp = self._gain_map[:gain_half, ...]
            gain_rp = self._gain_map[gain_half:, ...]
            lp *= gain_lp
            rp *= gain_rp

        # 5) un-binning:
        bin_factor = self._files[0].global_header['readoutmode']['bin']
        if bin_factor > 1:
            lp = _unbin(lp, factor=bin_factor)
            rp = _unbin(rp, factor=bin_factor)
        out[..., :half_height, :] = lp
        out[..., half_height:, :] = rp

        # FIXME: to be implemented:
        assert crop_to is None or tuple(crop_to.shape.sig) == tuple(out.shape[1:])

        return out


class FRMS6DataSet(DataSet):
    r"""
    Read PNDetector FRMS6 files. FRMS6 data sets consist of multiple .frms6 files and
    a .hdr file. The first .frms6 file (matching \*_000.frms6) contains dark frames, which
    are subtracted if `enable_offset_correction` is true.

    Parameters
    ----------

    path : string
        Path to one of the files of the FRMS6 dataset (either .hdr or .frms6)

    enable_offset_correction : boolean
        Subtract dark frames when reading data

    gain_map_path : string
        Path to a gain map to apply (.mat format)
    """
    def __init__(self, path, enable_offset_correction=True, gain_map_path=None, dest_dtype=None):
        super().__init__()
        self._path = path
        self._gain_map_path = gain_map_path
        self._dark_frame = None
        self._enable_offset_correction = enable_offset_correction
        self._meta = None
        if dest_dtype is not None:
            warnings.warn(
                "dest_dtype is now handled per `get_tiles` call, and ignored here",
                DeprecationWarning
            )

    @property
    def shape(self):
        return self._meta.shape

    def initialize(self):
        first_file = next(self._get_signal_files())
        header = first_file.header
        raw_frame_size = header['height'], header['width']
        # frms6 frames are folded in a specific way, this is the shape after unfolding:
        frame_size = 2 * header['height'], header['width'] // 2
        assert header['width'] % 2 == 0
        hdr = self._get_hdr_info()
        bin_factor = hdr['readoutmode']['bin']
        if bin_factor > 1:
            frame_size = (frame_size[0] * bin_factor, frame_size[1])

        sig_dims = 2  # FIXME: is there a different cameraMode that doesn't output 2D signals?
        self._meta = DataSetMeta(
            raw_dtype=np.dtype("u2"),
            metadata={'raw_frame_size': raw_frame_size},
            shape=Shape(tuple(hdr['stemimagesize']) + frame_size, sig_dims=sig_dims),
            iocaps={"FULL_FRAMES"},
        )
        self._dark_frame = self._get_dark_frame()
        self._gain_map = self._get_gain_map()
        self._total_filesize = sum(
            os.stat(path).st_size
            for path in self._files()
        )
        return self

    @classmethod
    def detect_params(cls, path):
        hdr_filename = "%s.hdr" % _get_base_filename(path)
        try:
            _read_hdr(hdr_filename)
        except Exception:
            return False
        return {"path": path}

    @classmethod
    def get_msg_converter(cls):
        return FRMS6DatasetParams

    def _get_hdr_info(self):
        hdr_filename = "%s.hdr" % _get_base_filename(self._path)
        return _read_hdr(hdr_filename)

    def _get_dark_frame(self):
        if not self._enable_offset_correction:
            return None
        dark_file = self._get_dark_file()
        # FIXME: currently doing two passes here: dtype conversion and summation
        return (
            dark_file.data.astype("float32").sum(axis=0) / dark_file.data.shape[0]
        )

    def _get_dark_file(self):
        """
        the file ending with "_000.frms6" contains the dark frames,
        which should be the first in our sorting order
        """
        # FIXME: dark frame acquisition may be disabled, we then need to either
        # 1) disable offset correction
        # 2) load the dark frames from a separate, user-given file
        return FRMS6File(path=self._files()[0])

    def _get_gain_map(self):
        if self._gain_map_path is None:
            return None
        _, ext = os.path.splitext(self._gain_map_path)
        if ext.lower() == '.mat':
            gain_mat = sio.loadmat(self._gain_map_path)
            return gain_mat['GainMap']
        elif ext.lower() == '.csv':
            with open(self._gain_map_path) as csv_f:
                csv_reader = csv.reader(csv_f, dialect=GainMapCSVDialect)
                csv_gain_data = list(csv_reader)
                csv_gain_nums = [[float(x) for x in row if x != ''] for row in csv_gain_data]
                return np.array(csv_gain_nums).T

    def _get_signal_files(self):
        start_idx = 0
        files = self._files()
        if len(files) < 2:
            raise DataSetException("did not find signal files")
        for path in files[1:]:
            f = FRMS6File(path=path, start_idx=start_idx, hdr_info=self._get_hdr_info())
            start_idx += f.num_frames
            yield f

    def _get_fileset(self):
        return FRMS6FileSet(
            files=list(self._get_signal_files()),
            meta=self._meta,
            dark_frame=self._dark_frame,
            gain_map=self._gain_map,
        )

    def _pattern(self):
        path, ext = os.path.splitext(self._path)
        if ext == ".hdr":
            pattern = "%s_*.frms6" % path
        elif ext == ".frms6":
            pattern = "%s*.frms6" % (
                re.sub(r'[0-9]+$', '', path)
            )
        else:
            raise DataSetException("unknown extension: %s" % ext)
        return pattern

    def _files(self):
        pattern = self._pattern()
        files = glob.glob(pattern)
        return list(sorted(files))

    def check_valid(self):
        try:
            for fn in self._files():
                f = FRMS6File(fn)
                if not f.check_valid():
                    raise DataSetException("error while checking validity of %s" % fn)
            return True
        except (IOError, OSError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def raw_dtype(self):
        return self._meta.raw_dtype

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 512MB (converted float data) per partition
        partition_size = 512 * 1024 * 1024 / 4
        partition_size /= np.dtype("u2").itemsize
        res = max(self._cores, self._total_filesize // int(partition_size))
        return res

    def get_partitions(self):
        for part_slice, start, stop in Partition3D.make_slices(
                shape=self.shape,
                num_partitions=self._get_num_partitions()):
            yield Partition3D(
                meta=self._meta,
                partition_slice=part_slice,
                fileset=self._get_fileset(),
                start_frame=start,
                num_frames=stop - start,
            )
