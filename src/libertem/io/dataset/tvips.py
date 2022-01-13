import os
import re
from typing import TYPE_CHECKING, NamedTuple, List, Optional, Tuple
import numpy as np
from glob import glob, escape

from libertem.common.math import prod
from libertem.common import Shape
from libertem.executor.base import JobExecutor
from libertem.web.messages import MessageConverter
from .base import (
    DataSet, DataSetException, DataSetMeta,
    BasePartition, File, FileSet, IOBackend,
)

if TYPE_CHECKING:
    from numpy import typing as nt


class TVIPSDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/TVIPSDatasetParams.schema.json",
        "title": "TVIPSDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "TVIPS"},
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
        "required": ["type", "path"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path"]
        }
        for k in ["nav_shape", "sig_shape", "sync_offset"]:
            if k in raw_data:
                data[k] = raw_data[k]
        return data


series_header_dtype = [
    ('ISize', 'i4'),            # The size of the series header (always 256)
    ('IVersion', 'i4'),         # The version of the file (1 or 2)
    ('IXDim', 'i4'),            # The x dimension of all images (width)
    ('IYDim', 'i4'),            # The y dimension of all images (height)
    ('IBPP', 'i4'),             # The number of bits per pixel (8 or 16)
    ('IXOff', 'i4'),            # The camera X offset of the image
    ('IYOff', 'i4'),            # The camera Y offset of the image
    ('IXBin', 'i4'),            # The camera X binning
    ('IYBin', 'i4'),            # The camera Y binning
    ('IPixelSize', 'i4'),       # The pixelsize in nm
    ('IHT', 'i4'),              # The high tension in kV
    ('IMagTotal', 'i4'),        # The total magnification including MagPost and MagCor factors
    ('IImgHeaderBytes', 'i4'),  # The size in bytes of the image headers (version 2 only)
    # 204 unused bytes follow
]


class SeriesHeader(NamedTuple):
    version: int
    xdim: int
    ydim: int
    xbin: int
    ybin: int
    bpp: int
    pixel_size_nm: int
    high_tension_kv: int
    mag_total: int
    frame_header_bytes: int


def read_series_header(path: str) -> SeriesHeader:
    with open(path, 'rb') as f:
        arr = np.fromfile(f, dtype=series_header_dtype, count=1)
    version = int(arr['IVersion'])
    if version not in [1, 2]:
        raise DataSetException(f"Unknown TVIPS header version: {version}")
    size = int(arr['ISize'])
    if size != 256:
        raise DataSetException(
            f"Invalid header size {size}, should be 256. Maybe not a TVIPS file?"
        )
    bpp = int(arr['IBPP'])
    if bpp not in [8, 16]:
        raise DataSetException(
            f"unknown bpp value: {bpp} (should be either 8 or 16)"
        )
    img_header_bytes = int(arr['IImgHeaderBytes'])
    if version == 1:
        img_header_bytes = 12
    return SeriesHeader(
        version=int(arr['IVersion']),
        xdim=int(arr['IXDim']),
        ydim=int(arr['IYDim']),
        xbin=int(arr['IXBin']),
        ybin=int(arr['IYBin']),
        bpp=bpp,
        pixel_size_nm=int(arr['IPixelSize']),
        high_tension_kv=int(arr['IHT']),
        mag_total=int(arr['IMagTotal']),
        frame_header_bytes=img_header_bytes,
    )


def frames_in_file(path: str, series_header: SeriesHeader) -> int:
    filesize = os.stat(path).st_size
    file_header = 0
    if _get_suffix(path) == 0:
        file_header = 256
    filesize -= file_header
    total_size_per_frame = series_header.frame_header_bytes + (
        series_header.bpp // 8 * series_header.xdim * series_header.ydim
    )
    rest = filesize % total_size_per_frame
    assert rest == 0, f"found a rest of {rest}, corrupted file?"
    return filesize // total_size_per_frame


def _pattern(path: str) -> str:
    path, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == '.tvips':
        pattern = "%s*.tvips" % (
            re.sub(r'[0-9]+$', '', escape(path))
        )
    else:
        raise DataSetException("unknown extension")
    return pattern


def get_image_count_and_sig_shape(path: str) -> Tuple[int, Tuple[int, int]]:
    fns = get_filenames(path)
    count = 0
    series_header = read_series_header(fns[0])
    for path in fns:
        count += frames_in_file(path, series_header)
    sig_shape = (series_header.ydim, series_header.xdim)
    return count, sig_shape


def _get_suffix(path: str) -> int:
    path, ext = os.path.splitext(path)
    # according to the docs, the suffix is always an underscore followed
    # by a three digit number with leading zeros:
    return int(path[-3:])


def get_filenames(path: str) -> List[str]:
    return list(sorted(glob(_pattern(path)), key=_get_suffix))


class TVIPSDataSet(DataSet):
    """
    Read data from one or more .tvips files. You can specify the path to any
    file that is part of a set - the whole data set will be loaded. The `nav_shape`
    needs to be specified if you are loading ND data.

    Examples
    --------

    >>> ds = ctx.load(
    ...     "tvips",
    ...     path="./path/to/file_000.tvips",
    ...     nav_shape=(16, 16)
    ... )  # doctest: +SKIP

    Parameters
    ----------

    path: str
        Path to the file

    nav_shape: tuple of int
        A n-tuple that specifies the size of the navigation region ((y, x), but
        can also be of length 1 for example for a line scan, or length 3 for
        a data cube, for example)

    sig_shape: tuple of int
        Common case: (height, width); but can be any dimensionality

    sync_offset: int, optional
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start
    """
    def __init__(
        self,
        path,
        nav_shape: Optional[Tuple[int, ...]] = None,
        sig_shape: Optional[Tuple[int, ...]] = None,
        sync_offset: int = 0,
        io_backend: Optional[IOBackend] = None,
    ):
        super().__init__(io_backend=io_backend)
        self._nav_shape = tuple(nav_shape) if nav_shape else nav_shape
        self._sig_shape = tuple(sig_shape) if sig_shape else sig_shape
        self._sync_offset = sync_offset
        self._path = path
        self._filesize = None
        self._series_header: Optional[SeriesHeader] = None

    def initialize(self, executor: JobExecutor):
        self._filesize = executor.run_function(self._get_filesize)
        files = executor.run_function(get_filenames, self._path)

        # The series header is contained in the first file:
        self._series_header = executor.run_function(read_series_header, files[0])

        # The total number of frames is not contained in a header, so we need
        # to calculate it from the file sizes:
        image_count = sum(
            executor.run_function(frames_in_file, fname, self._series_header)
            for fname in files
        )

        raw_dtype: "nt.DTypeLike"
        if self._series_header.bpp == 8:
            raw_dtype = np.uint8
        elif self._series_header.bpp == 16:
            raw_dtype = np.uint16

        nav_shape: Tuple[int, ...]
        if self._nav_shape is None:
            nav_shape = (image_count,)
        else:
            nav_shape = self._nav_shape

        self._image_count = image_count
        self._nav_shape_product = prod(nav_shape)
        image_size = (self._series_header.ydim, self._series_header.xdim)

        sig_shape: Tuple[int, ...]
        if self._sig_shape is None:
            sig_shape = image_size
        elif prod(self._sig_shape) != prod(image_size):
            raise DataSetException(
                "sig_shape must be of size: %s" % prod(image_size)
            )
        else:
            sig_shape = self._sig_shape

        # FIXME: reshaping self._sig_shape, self._nav_shape
        shape = Shape(
            nav_shape + sig_shape,
            sig_dims=2,
        )

        self._sync_offset_info = self.get_sync_offset_info()
        self._meta = DataSetMeta(
            shape=shape,
            raw_dtype=raw_dtype,
            sync_offset=self._sync_offset,
            image_count=image_count,
        )
        return self

    def _get_filesize(self):
        files = get_filenames(self._path)
        return sum(
            os.stat(fname).st_size
            for fname in files
        )

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    @classmethod
    def get_msg_converter(cls):
        return TVIPSDatasetParams

    @classmethod
    def get_supported_extensions(cls):
        return {"tvips"}

    @classmethod
    def detect_params(cls, path, executor):
        pathlow = path.lower()
        if pathlow.endswith(".tvips"):
            image_count, sig_shape = executor.run_function(get_image_count_and_sig_shape, path)
            nav_shape = tuple((image_count,))
        else:
            return False
        return {
            "parameters": {
                "path": path,
                "nav_shape": nav_shape,
                "sig_shape": sig_shape
            },
            "info": {
                "image_count": image_count,
                "native_sig_shape": sig_shape,
            }
        }

    def get_diagnostics(self):
        header = self._series_header
        return [
            {"name": "Bits per pixel",
             "value": str(header.bpp)},
            {"name": "High tension (kV)",
             "value": str(header.high_tension_kv)},
            {"name": "Pixel size (nm)",
             "value": str(header.pixel_size_nm)},
            {"name": "Binning (x)",
             "value": str(header.xbin)},
            {"name": "Binning (y)",
             "value": str(header.ybin)},
            {"name": "File Format Version",
             "value": str(header.version)},
        ]

    def _get_fileset(self):
        filenames = get_filenames(self._path)
        series_header = self._series_header
        start_idx = 0
        files = []
        for fname in filenames:
            num_frames = frames_in_file(fname, series_header)
            files.append(
                File(
                    path=fname,
                    file_header=256 if _get_suffix(fname) == 0 else 0,
                    start_idx=start_idx,
                    end_idx=start_idx + num_frames,
                    sig_shape=self.shape.sig,
                    native_dtype=self._meta.raw_dtype,
                    frame_header=series_header.frame_header_bytes,
                )
            )
            start_idx += num_frames
        return FileSet(files, frame_header_bytes=series_header.frame_header_bytes)

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
            "shape": tuple(self.shape),
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
        return f"<TVIPSDataSet shape={self.shape}>"
