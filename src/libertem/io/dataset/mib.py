import re
import io
import os
import glob
import logging

import numpy as np

from libertem.common import Shape
from libertem.io.partitioner import Partitioner3D
from .base import (
    DataSet, DataSetException, DataSetMeta,
    Partition3D, File3D, FileSet3D,
)

log = logging.getLogger(__name__)


class MIBFile(File3D):
    def __init__(self, path, fields=None):
        self.path = path
        if fields is None:
            self._fields = {}
        else:
            self._fields = fields
        super().__init__()

    def _get_np_dtype(self, dtype):
        dtype = dtype.lower()
        assert dtype[0] == "u"
        num_bytes = int(dtype[1:]) // 8
        return ">u%d" % num_bytes

    def read_header(self):
        with io.open(file=self.path, mode="r", encoding="ascii") as f:
            header = f.read(100)
            filesize = os.fstat(f.fileno()).st_size
        parts = header.split(",")
        image_size = (int(parts[5]), int(parts[4]))
        header_size_bytes = int(parts[2])
        bytes_per_pixel = int(parts[6][1:]) // 8
        num_images = filesize // (
            image_size[0] * image_size[1] * bytes_per_pixel + header_size_bytes
        )
        self._fields = {
            'header_size_bytes': header_size_bytes,
            'dtype': self._get_np_dtype(parts[6]),
            'bytes_per_pixel': bytes_per_pixel,
            'image_size': image_size,
            'sequence_first_image': int(parts[1]),
            'filesize': filesize,
            'num_images': num_images,
        }
        return self._fields

    @property
    def num_frames(self):
        return self.fields['num_images']

    @property
    def start_idx(self):
        return self.fields['sequence_first_image'] - 1

    @property
    def fields(self):
        if not self._fields:
            self.read_header()
        return self._fields

    def open(self):
        self._fh = open(self.path, "rb")

    def close(self):
        self._fh.close()

    def _frames_mmap(self, start, num):
        """
        read frames as views into the memmapped file

        Parameters
        ----------

        num : int
            number of frames to read
        start : int
            index of first frame to read (number of frames to skip)
        """
        bpp = self.fields['bytes_per_pixel']
        hsize = self.fields['header_size_bytes']
        assert hsize % bpp == 0
        size_px = self.fields['image_size'][0] * self.fields['image_size'][1]
        size = size_px * bpp  # bytes
        imagesize_incl_header = size + hsize  # bytes
        mapped = np.memmap(self.path, dtype=self.fields['dtype'], mode='r',
                           offset=start * imagesize_incl_header)

        # limit to number of frames to read
        mapped = mapped[:num * (size_px + hsize // bpp)]
        # reshape (num_frames, pixels) incl. header
        mapped = mapped.reshape((num, size_px + hsize // bpp))
        # cut off headers
        mapped = mapped[:, (hsize // bpp):]
        # reshape to (num_frames, pixels_y, pixels_x)
        return mapped.reshape((num, self.fields['image_size'][0], self.fields['image_size'][1]))

    def _frames_read(self, start, num):
        bpp = self.fields['bytes_per_pixel']
        hsize = self.fields['header_size_bytes']
        hsize_px = hsize // bpp
        assert hsize % bpp == 0
        size_px = self.fields['image_size'][0] * self.fields['image_size'][1]
        size = size_px * bpp  # bytes
        imagesize_incl_header = size + hsize  # bytes
        readsize = imagesize_incl_header * num
        buf = self.get_buffer("_frame_read", readsize)

        self._fh.seek(start * imagesize_incl_header)
        bytes_read = self._fh.readinto(buf)
        assert bytes_read == readsize
        arr = np.frombuffer(buf, dtype=self.fields['dtype'])

        # limit to number of frames to read
        arr = arr[:num * (size_px + hsize_px)]
        # reshape (num_frames, pixels) incl. header
        arr = arr.reshape((num, size_px + hsize_px))
        # cut off headers
        arr = arr[:, hsize_px:]
        # reshape to (num_frames, pixels_y, pixels_x)
        return arr.reshape((num, self.fields['image_size'][0], self.fields['image_size'][1]))

    def readinto(self, start, stop, out, crop_to=None):
        """
        Read a number of frames into an existing buffer, skipping the headers

        Parameters
        ----------

        stop : int
            end index
        start : int
            index of first frame to read
        out : buffer
            output buffer that should fit `stop - start` frames
        crop_to : Slice
            crop to the signal part of this Slice
        """
        num = stop - start
        frames = self._frames_read(num=num, start=start)
        if crop_to is not None:
            frames = frames[(...,) + crop_to.get(sig_only=True)]
        out[:] = frames
        return out


class MIBFileSet(FileSet3D):
    pass


class MIBDataSet(DataSet):
    def __init__(self, path, tileshape, scan_size, dest_dtype="float32"):
        self._sig_dims = 2
        self._path = path
        self._tileshape = Shape(tileshape, sig_dims=self._sig_dims)
        self._dest_dtype = np.dtype(dest_dtype)
        self._scan_size = tuple(scan_size)
        self._filename_cache = None
        self._files_sorted = None
        # ._preread_headers() calls ._files() which passes the cached headers down to MIBFile,
        # if they exist. So we need to make sure to initialize self._headers
        # before calling _preread_headers!
        self._headers = {}
        self._meta = None
        self._total_filesize = None

    def initialize(self):
        self._headers = self._preread_headers()
        self._files_sorted = list(sorted(self._files(),
                                         key=lambda f: f.fields['sequence_first_image']))

        try:
            first_file = self._files_sorted[0]
        except IndexError:
            raise DataSetException("no files found")
        shape = Shape(
            self._scan_size + first_file.fields['image_size'],
            sig_dims=self._sig_dims
        )
        raw_shape = shape.flatten_nav()
        dtype = first_file.fields['dtype']
        meta = DataSetMeta(shape=shape, raw_shape=raw_shape,
                           raw_dtype=dtype, dtype=self._dest_dtype)
        self._meta = meta
        self._total_filesize = sum(
            os.stat(path).st_size
            for path in self._filenames()
        )
        return self

    def get_diagnostics(self):
        return [
            {"name": "Data type",
             "value": str(self._meta.raw_dtype)},
        ]

    @classmethod
    def detect_params(cls, path):
        if path.endswith(".mib"):
            return {
                "path": path,
                "tileshape": (1, 8, 256, 256),
            }
        return False

    def _preread_headers(self):
        res = {}
        for f in self._files():
            res[f.path] = f.fields
        return res

    def _filenames(self):
        if self._filename_cache is not None:
            return self._filename_cache
        path, ext = os.path.splitext(self._path)
        ext = ext.lower()
        if ext == '.mib':
            pattern = "%s*.mib" % (
                re.sub(r'[0-9]+$', '', path)
            )
        elif ext == '.hdr':
            pattern = "%s*.mib" % path
        else:
            raise DataSetException("unknown extension")
        fns = glob.glob(pattern)
        self._filename_cache = fns
        return fns

    def _files(self):
        for path in self._filenames():
            f = MIBFile(path, fields=self._headers.get(path))
            yield f

    def _num_images(self):
        return sum(f.fields['num_images'] for f in self._files())

    @property
    def dtype(self):
        return self._meta.dtype

    @property
    def shape(self):
        """
        the 4D shape imprinted by number of images and scan_size
        """
        return self._meta.shape

    @property
    def raw_shape(self):
        """
        the original 3D shape
        """
        return self._meta.raw_shape

    def check_valid(self):
        try:
            s = self._scan_size
            num_images = self._num_images()
            # FIXME: read hdr file and check if num images matches the number there
            if s[0] * s[1] != num_images:
                raise DataSetException(
                    "scan_size (%r) does not match number of images (%d)" % (
                        s, num_images
                    )
                )
            if self._tileshape.sig != self.raw_shape.sig:
                raise DataSetException(
                    "MIB only supports tileshapes that match whole frames, %r != %r" % (
                        self._tileshape.sig, self.raw_shape.sig
                    )
                )
            if self._tileshape[0] != 1:
                raise DataSetException(
                    "MIB only supports tileshapes that don't cross rows"
                )
        except (IOError, OSError, KeyError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def _get_fileset(self):
        return MIBFileSet(files=self._files_sorted)

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 512MB per partition
        res = max(1, self._total_filesize // (512*1024*1024))
        return res

    def get_partitions(self):
        partitioner = Partitioner3D()
        fileset = self._get_fileset()
        for part_slice, start, stop in partitioner.get_slices(
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

    def __repr__(self):
        return "<MIBDataSet of %s shape=%s>" % (self.dtype, self.raw_shape)
