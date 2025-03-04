from io import SEEK_SET
import math
import os
import re
from typing import IO, TYPE_CHECKING, NamedTuple, Optional
import numpy as np
from glob import glob, escape

from libertem.common.math import prod, make_2D_square
from libertem.common import Shape
from libertem.common.executor import JobExecutor
from libertem.common.messageconverter import MessageConverter
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


SERIES_HEADER_SIZE = 256

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


image_header_v2_dtype = [
    ('ICounter', 'u4'),                 # image counter, continues through all files
    ('ITime', 'u4'),                    # unix time stamp
    ('IMS', 'u4'),                      # timestamp milliseconds
    ('LUT_Index', 'u4'),                # LUT index (?)
    ('Faraday', 'float32'),             # faraday cup value (unit?)
    ('TEM_Mag', 'u4'),                  # magnification (unit?)
    ('TEM_Mag_mode', 'u4'),             # magnification mode (1=imaging, 2=diffraction)
    ('TEM_Stage_x', 'float32'),         # stage X in nm
    ('TEM_Stage_y', 'float32'),         # stage Y in nm
    ('TEM_Stage_z', 'float32'),         # stage Z in nm
    ('TEM_Stage_alpha', 'float32'),     # in degree
    ('TEM_Stage_beta', 'float32'),      # in degree
    ('Index_of_rotator', 'u4'),         # ?
    ('DENS_T_measure', 'float32'),
    ('DENS_T_setpoint', 'float32'),
    ('DENS_Power', 'float32'),
    ('TEM_Obj_current', 'float32'),     # unit?
    ('Scan_x', 'float32'),
    ('Scan_y', 'float32'),
    ('DENS_Bias_U_setpoint', 'float32'),
    ('DENS_Bias_U_value', 'float32'),
    ('DENS_Bias_I_setpoint', 'float32'),
    ('DENS_Bias_I_value', 'float32'),
    ('DENS_Bias_E_setpoint', 'float32'),
    ('DENS_Bias_R', 'float32'),
    ('DENS_Bias_limit_U', 'float32'),  # compliance limit
    ('DENS_Bias_limit_I', 'float32'),  # compliance limit
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
    version = int(arr['IVersion'][0])
    if version not in [1, 2]:
        raise DataSetException(f"Unknown TVIPS header version: {version}")
    size = int(arr['ISize'][0])
    if size != SERIES_HEADER_SIZE:
        raise DataSetException(
            f"Invalid header size {size}, should be 256. Maybe not a TVIPS file?"
        )
    bpp = int(arr['IBPP'][0])
    if bpp not in [8, 16]:
        raise DataSetException(
            f"unknown bpp value: {bpp} (should be either 8 or 16)"
        )
    img_header_bytes = int(arr['IImgHeaderBytes'][0])
    if version == 1:
        img_header_bytes = 12
    return SeriesHeader(
        version=int(arr['IVersion'][0]),
        xdim=int(arr['IXDim'][0]),
        ydim=int(arr['IYDim'][0]),
        xbin=int(arr['IXBin'][0]),
        ybin=int(arr['IYBin'][0]),
        bpp=bpp,
        pixel_size_nm=int(arr['IPixelSize'][0]),
        high_tension_kv=int(arr['IHT'][0]),
        mag_total=int(arr['IMagTotal'][0]),
        frame_header_bytes=img_header_bytes,
    )


def frames_in_file(path: str, series_header: SeriesHeader) -> int:
    filesize = os.stat(path).st_size
    file_header = 0
    if _get_suffix(path) == 0:
        file_header = SERIES_HEADER_SIZE
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


def get_image_count_and_sig_shape(path: str) -> tuple[int, tuple[int, int]]:
    fns = get_filenames(path)
    count = 0
    series_header = read_series_header(fns[0])
    for path in fns:
        count += frames_in_file(path, series_header)
    sig_shape = (series_header.ydim, series_header.xdim)
    return count, sig_shape


MAX_SCAN_IDX = 4096  # we only check until this index for the beginning of the scan


def _image_header_for_idx(f: IO[bytes], series_header: SeriesHeader, idx: int) -> np.ndarray:
    image_size_bytes = series_header.bpp // 8 * series_header.xdim * series_header.ydim
    skip_size = series_header.frame_header_bytes + image_size_bytes
    offset = SERIES_HEADER_SIZE + idx * skip_size
    f.seek(offset, SEEK_SET)
    return np.fromfile(f, dtype=image_header_v2_dtype, count=1)  # type:ignore


def _scan_for_idx(f: IO[bytes], series_header: SeriesHeader, idx: int) -> tuple[int, int]:
    arr = _image_header_for_idx(f, series_header, idx)
    # this assumes integer scan coordinates:
    scan_y = int(arr['Scan_y'][0])
    scan_x = int(arr['Scan_x'][0])
    scan = (scan_y, scan_x)
    return scan


class DetectionError(Exception):
    pass


def detect_shape(path: str) -> tuple[int, tuple[int, ...]]:
    series_header = read_series_header(path)

    if series_header.version != 2:
        raise DetectionError(
            "unknown series header version, can only detect shape from v2"
        )

    count, _ = get_image_count_and_sig_shape(path)
    filenames = get_filenames(path)
    first_file = filenames[0]
    sync_offset = 0

    with open(first_file, "rb") as f:
        idx = 0
        last_was_zero = False
        found_offset = False
        while idx < MAX_SCAN_IDX and idx < count:
            scan = _scan_for_idx(f, series_header, idx)
            if last_was_zero and scan == (0, 1):
                sync_offset = idx - 1
                found_offset = True
                break
            if scan == (0, 0):
                last_was_zero = True
            idx += 1

        if not found_offset:
            raise DetectionError("Could not auto-detect sync_offset")

        # continue where we left off and search for max(scan_x):
        max_x = 0  # scan positions start at 0, so our shape is (y, max_x + 1)
        found_shape = False
        while idx < MAX_SCAN_IDX and idx < count:
            scan = _scan_for_idx(f, series_header, idx)
            # assume monotonously increasing values
            max_x = max(max_x, scan[1])
            if scan[1] < max_x:
                found_shape = True
                break
            idx += 1

    shape: tuple[int, ...]
    if found_shape:
        shape = (int(math.floor((count - sync_offset) / (max_x + 1))), max_x + 1)
    else:
        shape = (count,)

    return sync_offset, shape


def _get_suffix(path: str) -> int:
    path, ext = os.path.splitext(path)
    # according to the docs, the suffix is always an underscore followed
    # by a three digit number with leading zeros:
    return int(path[-3:])


def get_filenames(path: str) -> list[str]:
    return list(sorted(glob(_pattern(path)), key=_get_suffix))


class TVIPSDataSet(DataSet):
    """
    Read data from one or more .tvips files. You can specify the path to any
    file that is part of a set - the whole data set will be loaded. We will try
    to guess :code:`nav_shape` and :code:`sync_offset` from the image headers
    for 4D STEM data, but you may need to specify these parameters in case the
    guessing logic fails.

    .. versionadded:: 0.9.0

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
        If not given, we try to automatically determine the sync_offset from
        the scan metadata in the image headers.

    num_partitions: int, optional
        Override the number of partitions. This is useful if the
        default number of partitions, chosen based on common workloads,
        creates partitions which are too large (or small) for the UDFs
        being run on this dataset.
    """
    def __init__(
        self,
        path,
        nav_shape: Optional[tuple[int, ...]] = None,
        sig_shape: Optional[tuple[int, ...]] = None,
        sync_offset: Optional[int] = None,
        io_backend: Optional[IOBackend] = None,
        num_partitions: Optional[int] = None,
    ):
        super().__init__(
            io_backend=io_backend,
            num_partitions=num_partitions,
        )
        self._nav_shape = tuple(nav_shape) if nav_shape else nav_shape
        self._sig_shape = tuple(sig_shape) if sig_shape else sig_shape
        self._sync_offset = sync_offset
        self._path = path
        self._filesize = None
        self._files: Optional[list[str]] = None
        self._frame_counts: dict[str, int] = {}
        self._series_header: Optional[SeriesHeader] = None

    def initialize(self, executor: JobExecutor):
        self._filesize = executor.run_function(self._get_filesize)
        files = executor.run_function(get_filenames, self._path)

        # The series header is contained in the first file:
        self._series_header = executor.run_function(read_series_header, files[0])

        for fname in files:
            self._frame_counts[fname] = executor.run_function(
                frames_in_file,
                fname,
                self._series_header
            )

        self._files = files

        try:
            sync_offset_detected, nav_shape_detected = executor.run_function(
                detect_shape, self._path
            )
            if self._sync_offset is None:
                self._sync_offset = sync_offset_detected
        except DetectionError:
            sync_offset_detected = None
            nav_shape_detected = None
            if self._sync_offset is None:
                self._sync_offset = 0

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

        nav_shape: tuple[int, ...]
        if self._nav_shape is None and nav_shape_detected is not None:
            nav_shape = nav_shape_detected
        elif self._nav_shape is None and nav_shape_detected is None:
            nav_shape = (image_count,)
        elif self._nav_shape is not None:
            nav_shape = self._nav_shape
        else:
            raise RuntimeError("should not happen")  # logic and all that good stuff...

        self._image_count = image_count
        self._nav_shape_product = prod(nav_shape)
        image_size = (self._series_header.ydim, self._series_header.xdim)

        sig_shape: tuple[int, ...]
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
            try:
                sync_offset, nav_shape = executor.run_function(detect_shape, path)
            except DetectionError:
                sync_offset = 0
                nav_shape = make_2D_square((image_count,))
        else:
            return False
        return {
            "parameters": {
                "path": path,
                "nav_shape": nav_shape,
                "sig_shape": sig_shape,
                "sync_offset": sync_offset,
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
        filenames = self._files
        series_header = self._series_header
        start_idx = 0
        files = []
        for fname in filenames:
            num_frames = self._frame_counts[fname]
            files.append(
                File(
                    path=fname,
                    file_header=SERIES_HEADER_SIZE if _get_suffix(fname) == 0 else 0,
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
