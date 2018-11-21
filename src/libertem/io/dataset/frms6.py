# -*- encoding: utf-8 -*-
import os
import re
import glob

import numpy as np

from libertem.common.slice import Slice
from .base import DataSet, Partition, DataTile, DataSetException


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


class FRMS6File(object):
    def __init__(self, path):
        self._path = path
        self._header = None

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

    def check_valid(self):
        if not (self.header['header_size'] == 1024
                and self.header['frame_header_size'] == 64
                and self.header['version'] == 6):
            return False
        # TODO: file size sanity check?
        return True

    def _num_frames(self):
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
        assert num % denum == 0
        return res

    def _get_mmapped_array(self):
        raw_data = np.memmap(self._path, dtype=self.dtype)
        # cut off the file header:
        header_size_px = self.header['header_size'] // self.dtype.itemsize
        frames = raw_data[header_size_px:]

        # TODO: we just throw away the frame headers here
        # TODO: for future work, we may want to validate the stuff in there!
        w, h = self.header['width'], self.header['height']
        num_frames = self._num_frames()
        frames_w_headers = frames.reshape((num_frames, w * h + 32))
        frames_wo_headers = frames_w_headers[:, 32:]
        frames_wo_headers = frames_wo_headers.reshape((num_frames, h, w))
        return frames_wo_headers

    def get_data(self):
        return self._get_mmapped_array()


class FRMS6DataSet(DataSet):
    def __init__(self, path, scan_size):
        self._path = path
        self._scan_size = scan_size
        # TODO: extract info from the .hdr file that should exist for each acquisition

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

    def _get_mmapped_arrays(self):
        return [
            FRMS6File(fn).get_data()
            for fn in self._files()
        ]

    def check_valid(self):
        try:
            for fn in self._files:
                f = FRMS6File(fn)
                if not f.check_valid():
                    raise DataSetException("error while checking validity of %s" % fn)
            return True
        except (IOError, OSError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    @property
    def dtype(self):
        return np.dtype("<u2")

    def get_partitions(self):
        raise NotImplementedError()


class FRMS6Partition(Partition):
    def __init__(self, tileshape, *args, **kwargs):
        raise NotImplementedError()

    def get_tiles(self, crop_to=None):
        raise NotImplementedError()

    def get_locations(self):
        return "127.0.1.1"  # FIXME
