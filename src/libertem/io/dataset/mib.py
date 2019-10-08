import re
import io
import os
import glob
import logging

import numpy as np

from libertem.common import Shape
from libertem.web.messages import MessageConverter
from .base import (
    DataSet, DataSetException, DataSetMeta,
    Partition3D, File3D, FileSet3D
)

log = logging.getLogger(__name__)


class MIBDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/MIBDatasetParams.schema.json",
        "title": "MIBDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "mib"},
            "path": {"type": "string"},
            "scan_size": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
            "tileshape": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
        },
        "required": ["type", "path"],
    }

    def convert_to_python(self, raw_data):
        data = {
            "path": raw_data["path"],
        }
        if "scan_size" in raw_data:
            data["scan_size"] = tuple(raw_data["scan_size"])
        if "tileshape" in raw_data:
            data["tileshape"] = tuple(raw_data["tileshape"])
        return data


def read_hdr_file(path):
    result = {}
    with open(path, "r", encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("HDR") or line.startswith("End\t"):
                continue
            k, v = line.split("\t", 1)
            k = k.rstrip(':')
            v = v.rstrip("\n")
            result[k] = v
    return result


def is_valid_hdr(path):
    with open(path, "r", encoding='utf-8', errors='ignore') as f:
        line = next(f)
        return line.startswith("HDR")


def scan_size_from_hdr(hdr):
    num_frames, scan_x = (
        int(hdr['Frames in Acquisition (Number)']),
        int(hdr['Frames per Trigger (Number)'])
    )
    scan_size = (num_frames // scan_x, scan_x)
    return scan_size


class MIBFile(File3D):
    def __init__(self, path, fields=None, sequence_start=None):
        self.path = path
        if fields is None:
            self._fields = {}
        else:
            self._fields = fields
        self._sequence_start = sequence_start
        super().__init__()

    def _get_np_dtype(self, dtype):
        dtype = dtype.lower()
        num_bits = int(dtype[1:])
        if dtype[0] == "u":
            num_bytes = num_bits // 8
            return np.dtype(">u%d" % num_bytes)
        elif dtype[0] == "r":
            assert num_bits == 64
            return np.dtype("uint8")  # the dtype after np.unpackbits

    def read_header(self):
        with io.open(file=self.path, mode="r", encoding="ascii", errors='ignore') as f:
            header = f.read(100)
            filesize = os.fstat(f.fileno()).st_size
        parts = header.split(",")
        dtype = parts[6].lower()
        mib_kind = dtype[0]
        image_size = (int(parts[5]), int(parts[4]))
        header_size_bytes = int(parts[2])
        if mib_kind == "u":
            bytes_per_pixel = int(parts[6][1:]) // 8
            image_size_bytes = image_size[0] * image_size[1] * bytes_per_pixel
        elif mib_kind == "r":
            bytes_per_pixel = 1  # after np.unpackbits
            image_size_bytes = image_size[0] * image_size[1] // 8
        else:
            raise ValueError("unknown kind: %s" % mib_kind)

        num_images = filesize // (
            image_size_bytes + header_size_bytes
        )
        self._fields = {
            'header_size_bytes': header_size_bytes,
            'dtype': self._get_np_dtype(parts[6]),
            'mib_dtype': dtype,
            'mib_kind': mib_kind,
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
        return self.fields['sequence_first_image'] - self._sequence_start

    @property
    def fields(self):
        if not self._fields:
            self.read_header()
        return self._fields

    def open(self):
        self._fh = open(self.path, "rb")

    def close(self):
        self._fh.close()
        self._fh = None

    def _frames_read_int(self, start, num, method):
        bpp = self.fields['bytes_per_pixel']
        hsize = self.fields['header_size_bytes']
        hsize_px = hsize // bpp
        assert hsize % bpp == 0
        size_px = self.fields['image_size'][0] * self.fields['image_size'][1]
        size = size_px * bpp  # bytes
        imagesize_incl_header = size + hsize  # bytes

        if method == "read":
            readsize = imagesize_incl_header * num
            buf = self.get_buffer("_frames_read_int", readsize)

            self._fh.seek(start * imagesize_incl_header)
            bytes_read = self._fh.readinto(buf)
            assert bytes_read == readsize
            arr = np.frombuffer(buf, dtype=self.fields['dtype'])
        elif method == "mmap":
            arr = np.memmap(self.path, dtype=self.fields['dtype'], mode='r',
                            offset=start * imagesize_incl_header)
            # limit to number of frames to read
            arr = arr[:num * (size_px + hsize_px)]
        else:
            raise ValueError("unknown method: %d" % method)

        # reshape (num_frames, pixels) incl. header
        arr = arr.reshape((num, size_px + hsize_px))
        # cut off headers
        arr = arr[:, hsize_px:]
        return arr

    def _frames_read_bits(self, start, num):
        """
        read frames for type r64, that is, pixels are bit-packed into big-endian
        64bit integers
        """
        hsize_px = 8 * self.fields['header_size_bytes']  # unpacked header size
        size_px = self.fields['image_size'][0] * self.fields['image_size'][1]
        size = size_px // 8
        hsize = self.fields['header_size_bytes']
        imagesize_incl_header = size + hsize
        readsize = imagesize_incl_header * num
        buf = self.get_buffer("_frames_read_bits", readsize)

        self._fh.seek(start * imagesize_incl_header)
        bytes_read = self._fh.readinto(buf)
        assert bytes_read == readsize
        raw_data = np.frombuffer(buf, dtype="u1")
        unpacked = np.unpackbits(raw_data)

        # reshape (num_frames, pixels) incl. header
        unpacked = unpacked.reshape((num, size_px + hsize_px))
        # cut off headers
        unpacked = unpacked[:, hsize_px:]

        arr = unpacked.reshape((num * 4 * 256, 64))[:, ::-1].reshape((num, 256, 256))
        return arr

    def _frames_read(self, start, num, method="read"):
        mib_kind = self.fields['mib_kind']

        if mib_kind == "r":
            arr = self._frames_read_bits(start, num)
        elif mib_kind == "u":
            arr = self._frames_read_int(start, num, method)

        # reshape to (num_frames, pixels_y, pixels_x)
        return arr.reshape((num, self.fields['image_size'][0], self.fields['image_size'][1]))

    def readinto(self, start, stop, out, crop_to=None):
        """
        Read a number of frames into an existing buffer, skipping the headers.

        Note: this method is not thread safe!

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
        frames = self._frames_read(num=num, start=start, method="read")
        if crop_to is not None:
            frames = frames[(...,) + crop_to.get(sig_only=True)]
        out[:] = frames
        return out


class MIBFileSet(FileSet3D):
    pass


class MIBDataSet(DataSet):
    # FIXME include sample file for doctest, see Issue #86
    """
    MIB data sets consist of one or more `.mib` files, and optionally
    a `.hdr` file. The HDR file is used to automatically set the
    `scan_size` parameter from the fields "Frames per Trigger" and "Frames
    in Acquisition." When loading a MIB data set, you can either specify
    the path to the HDR file, or choose one of the MIB files. The MIB files
    are assumed to follow a naming pattern of some non-numerical prefix,
    and a sequential numerical suffix.

    Note that, as of the current version, no gain correction or hot/cold pixel
    removal is done yet: processing is done on the RAW data, though you can do
    pre-processing in your own UDF.

    Examples
    --------

    >>> # both examples look for files matching /path/to/default*.mib:
    >>> ds1 = ctx.load("mib", path="/path/to/default.hdr")  # doctest: +SKIP
    >>> ds2 = ctx.load("mib", path="/path/to/default64.mib")  # doctest: +SKIP

    Parameters
    ----------
    path: str
        Path to either the .hdr file or one of the .mib files

    tileshape: tuple of int, optional
        Tuning parameter, specifying the size of the smallest data unit
        we are reading and working on. Will be automatically determined
        if left None.

    scan_size: tuple of int, optional
        A tuple (y, x) that specifies the size of the scanned region. It is
        automatically read from the .hdr file if you specify one as `path`.
    """
    def __init__(self, path, tileshape=None, scan_size=None):
        super().__init__()
        self._sig_dims = 2
        self._path = path
        if tileshape is None:
            tileshape = (1, 3, 256, 256)
        tileshape = Shape(tileshape, sig_dims=self._sig_dims)
        self._tileshape = tileshape
        if scan_size is not None:
            scan_size = tuple(scan_size)
        else:
            if not path.lower().endswith(".hdr"):
                raise ValueError(
                    "either scan_size needs to be passed, or path needs to point to a .hdr file"
                )
        self._scan_size = scan_size
        self._filename_cache = None
        self._files_sorted = None
        # ._preread_headers() calls ._files() which passes the cached headers down to MIBFile,
        # if they exist. So we need to make sure to initialize self._headers
        # before calling _preread_headers!
        self._headers = {}
        self._meta = None
        self._total_filesize = None
        self._sequence_start = None

    def initialize(self):
        self._headers = self._preread_headers()
        self._files_sorted = list(sorted(self._files(),
                                         key=lambda f: f.fields['sequence_first_image']))

        try:
            first_file = self._files_sorted[0]
        except IndexError:
            raise DataSetException("no files found")
        if self._scan_size is None:
            hdr = read_hdr_file(self._path)
            self._scan_size = scan_size_from_hdr(hdr)
        shape = Shape(
            self._scan_size + first_file.fields['image_size'],
            sig_dims=self._sig_dims
        )
        dtype = first_file.fields['dtype']
        meta = DataSetMeta(shape=shape, raw_dtype=dtype, iocaps={
            "FRAME_CROPS", "MMAP", "FULL_FRAMES"
        })
        if first_file.fields['mib_dtype'] == "r64":
            meta.iocaps.remove("MMAP")
        self._meta = meta
        self._total_filesize = sum(
            os.stat(path).st_size
            for path in self._filenames()
        )
        self._sequence_start = first_file.fields['sequence_first_image']
        self._files_sorted = list(sorted(self._files(),
                                         key=lambda f: f.fields['sequence_first_image']))
        return self

    def get_diagnostics(self):
        first_file = self._files_sorted[0]
        return [
            {"name": "Data type",
             "value": str(first_file.fields['mib_dtype'])},
        ]

    @classmethod
    def get_msg_converter(cls):
        return MIBDatasetParams

    @classmethod
    def detect_params(cls, path):
        pathlow = path.lower()
        if pathlow.endswith(".mib"):
            return {
                "path": path,
                "tileshape": (1, 3, 256, 256),
            }
        elif pathlow.endswith(".hdr") and is_valid_hdr(path):
            hdr = read_hdr_file(path)
            scan_size = scan_size_from_hdr(hdr)
            return {
                "path": path,
                "tileshape": (1, 3, 256, 256),
                "scan_size": scan_size,
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
            f = MIBFile(path, fields=self._headers.get(path),
                        sequence_start=self._sequence_start)
            yield f

    def _num_images(self):
        return sum(f.fields['num_images'] for f in self._files())

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        """
        the 4D shape imprinted by number of images and scan_size
        """
        return self._meta.shape

    def check_valid(self):
        try:
            s = self._scan_size
            num_images = self._num_images()
            # FIXME: read hdr file and check if num images matches the number there
            if s[0] * s[1] > num_images:
                raise DataSetException(
                    "scan_size (%r) does not match number of images (%d)" % (
                        s, num_images
                    )
                )
            if self._tileshape.sig != self.shape.sig:
                raise DataSetException(
                    "MIB only supports tileshapes that match whole frames, %r != %r" % (
                        self._tileshape.sig, self.shape.sig
                    )
                )
            if self._tileshape[0] != 1:
                raise DataSetException(
                    "MIB only supports tileshapes that don't cross rows"
                )
        except (IOError, OSError, KeyError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def _get_fileset(self):
        assert self._sequence_start is not None
        return MIBFileSet(files=self._files_sorted)

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 512MB (converted float data) per partition
        partition_size = 512 * 1024 * 1024 / 4
        first_file = self._files_sorted[0]
        if first_file.fields['mib_kind'] == "r":
            partition_size /= 4
        else:
            bpp = first_file.fields['bytes_per_pixel']
            partition_size /= bpp
        res = max(self._cores, self._total_filesize // int(partition_size))
        return res

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in Partition3D.make_slices(
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
        return "<MIBDataSet of %s shape=%s>" % (self.dtype, self.shape)
