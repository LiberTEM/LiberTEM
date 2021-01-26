import re
import io
import os
from glob import glob
import typing
import logging
import warnings

import numba
import numpy as np

from libertem.common import Shape
from libertem.web.messages import MessageConverter
from .base import (
    DataSet, DataSetException, DataSetMeta,
    BasePartition, FileSet, LocalFile, make_get_read_ranges,
    Decoder, TilingScheme, default_get_read_ranges,
    DtypeConversionDecoder,
)

log = logging.getLogger(__name__)


class MIBDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/MIBDatasetParams.schema.json",
        "title": "MIBDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "MIB"},
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
        "required": ["type", "path"],
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path"]
        }
        if "nav_shape" in raw_data:
            data["nav_shape"] = tuple(raw_data["nav_shape"])
        if "sig_shape" in raw_data:
            data["sig_shape"] = tuple(raw_data["sig_shape"])
        if "sync_offset" in raw_data:
            data["sync_offset"] = raw_data["sync_offset"]
        return data


def read_hdr_file(path):
    result = {}
    # FIXME: do this open via the io backend!
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
    # FIXME: do this open via the io backend!
    with open(path, "r", encoding='utf-8', errors='ignore') as f:
        line = next(f)
        return line.startswith("HDR")


def nav_shape_from_hdr(hdr):
    num_frames, scan_x = (
        int(hdr['Frames in Acquisition (Number)']),
        int(hdr['Frames per Trigger (Number)'])
    )
    nav_shape = (num_frames // scan_x, scan_x)
    return nav_shape


def get_filenames(path, disable_glob=False):
    if disable_glob:
        return [path]
    path, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == '.mib':
        pattern = "%s*.mib" % (
            re.sub(r'[0-9]+$', '', path)
        )
    elif ext == '.hdr':
        pattern = "%s*.mib" % path
    else:
        raise DataSetException("unknown extension")
    return glob(pattern)


def get_image_count_and_sig_shape(path):
    fns = get_filenames(path)
    count = 0
    files = []
    for path in fns:
        f = MIBHeaderReader(path)
        count += f.fields['num_images']
        files.append(f)
    try:
        first_file = list(sorted(files, key=lambda f: f.fields['sequence_first_image']))[0]
        sig_shape = first_file.fields['image_size']
        return count, sig_shape
    except IndexError:
        raise DataSetException("no files found")


@numba.njit(inline='always', cache=True)
def _mib_r_px_to_bytes(
    bpp, frame_in_file_idx, slice_sig_size, sig_size, sig_origin,
    frame_footer_bytes, frame_header_bytes,
    file_idx, read_ranges,
):
    # NOTE: bpp is not used, instead we divide by 8 because we have 8 pixels per byte!

    # we are reading a part of a single frame, so we first need to find
    # the offset caused by headers and footers:
    footer_offset = frame_footer_bytes * frame_in_file_idx
    header_offset = frame_header_bytes * (frame_in_file_idx + 1)
    byte_offset = footer_offset + header_offset

    # now let's figure in the current frame index:
    # (go down into the file by full frames; `sig_size`)
    offset = byte_offset + frame_in_file_idx * sig_size // 8

    # offset in px in the current frame:
    sig_origin_bytes = sig_origin // 8

    start = offset + sig_origin_bytes

    # size of the sig part of the slice:
    sig_size_bytes = slice_sig_size // 8

    stop = start + sig_size_bytes
    read_ranges.append((file_idx, start, stop))


@numba.njit(inline='always', cache=True)
def _mib_r24_px_to_bytes(
    bpp, frame_in_file_idx, slice_sig_size, sig_size, sig_origin,
    frame_footer_bytes, frame_header_bytes,
    file_idx, read_ranges,
):
    # we are reading a part of a single frame, so we first need to find
    # the offset caused by headers and footers:
    footer_offset = frame_footer_bytes * frame_in_file_idx
    header_offset = frame_header_bytes * (frame_in_file_idx + 1)
    byte_offset = footer_offset + header_offset

    # now let's figure in the current frame index:
    # (go down into the file by full frames; `sig_size`)
    offset = byte_offset + frame_in_file_idx * sig_size * bpp

    # offset in px in the current frame:
    sig_origin_bytes = sig_origin * bpp

    start = offset + sig_origin_bytes

    # size of the sig part of the slice:
    sig_size_bytes = slice_sig_size * bpp

    stop = start + sig_size_bytes
    read_ranges.append((file_idx, start, stop))

    # this is the addition for medipix 24bit raw:
    # we read the part from the "second 12bit frame"
    second_frame_offset = sig_size * bpp
    read_ranges.append((file_idx, start + second_frame_offset, stop + second_frame_offset))


mib_r_get_read_ranges = make_get_read_ranges(px_to_bytes=_mib_r_px_to_bytes)
mib_r24_get_read_ranges = make_get_read_ranges(px_to_bytes=_mib_r24_px_to_bytes)


@numba.jit(inline='always', cache=True)
def decode_r1_swap(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    """
    RAW 1bit format: each bit is actually saved as a single bit. 64 bits
    need to be unpacked together.
    """
    for stripe in range(inp.shape[0] // 8):
        for byte in range(8):
            inp_byte = inp[(stripe + 1) * 8 - (byte + 1)]
            for bitpos in range(8):
                out[idx, 64 * stripe + 8 * byte + bitpos] = (inp_byte >> bitpos) & 1


@numba.njit(inline='always', cache=True)
def decode_r6_swap(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    """
    RAW 6bit format: the pixels need to be re-ordered in groups of 8. `inp`
    should have dtype uint8.
    """
    for i in range(out.shape[1]):
        col = i % 8
        pos = i // 8
        out_pos = (pos + 1) * 8 - col - 1
        out[idx, out_pos] = inp[i]


@numba.njit(inline='always', cache=True)
def decode_r12_swap(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    """
    RAW 12bit format: the pixels need to be re-ordered in groups of 4. `inp`
    should be an uint8 view on big endian 12bit data (">u2")
    """
    for i in range(out.shape[1]):
        col = i % 4
        pos = i // 4
        out_pos = (pos + 1) * 4 - col - 1
        out[idx, out_pos] = (inp[i * 2] << 8) + (inp[i * 2 + 1] << 0)


@numba.njit(inline='always', cache=True)
def decode_r24_swap(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    """
    RAW 24bit format: a single 24bit consists of two frames that are encoded
    like the RAW 12bit format, the first contains the most significant bits.

    So after a frame header, there are (512, 256) >u2 values, which then
    need to be shuffled like in `decode_r12_swap`.

    This decoder function only works together with mib_r24_get_read_ranges
    which generates twice as many read ranges than normally.
    """
    for i in range(out.shape[1]):
        col = i % 4
        pos = i // 4
        out_pos = (pos + 1) * 4 - col - 1
        out_val = np.uint32((inp[i * 2] << 8) + (inp[i * 2 + 1] << 0))
        if idx % 2 == 0:  # from first frame: most significant bits
            out_val = out_val << 12
        out[idx // 2, out_pos] += out_val


class MIBDecoder(Decoder):
    def __init__(self, kind, dtype, bit_depth):
        self._kind = kind
        self._dtype = dtype
        self._bit_depth = bit_depth

    def do_clear(self):
        """
        In case of 24bit raw data, the output buffer needs to be cleared
        before writing, as we can't decode the whole frame in a single call of the decoder.
        The `decode_r24_swap` function needs to be able to add to the output
        buffer, so at the beginning, it needs to be cleared.
        """
        if self._kind == 'r' and self._bit_depth == 24:
            return True
        return False

    def _get_decode_r(self):
        # FIXME: R-mode files also mean we are constrained
        # to tile sizes that are a multiple of 64px in the fastest
        # dimension! need to take care of this in the negotiation
        # -> if we make sure full "x-lines" are taken, we are fine
        bit_depth = self._bit_depth
        if bit_depth == 1:
            return decode_r1_swap
        elif bit_depth == 6:
            return decode_r6_swap
        elif bit_depth == 12:
            return decode_r12_swap
        elif bit_depth == 24:
            return decode_r24_swap
        else:
            raise ValueError("unknown raw bitdepth")

    def get_decode(self, native_dtype, read_dtype):
        kind = self._kind

        if kind == "u":
            return DtypeConversionDecoder().get_decode(
                native_dtype=native_dtype,
                read_dtype=read_dtype,
            )
        elif kind == "r":
            # FIXME: on big endian systems, these need to be implemented without byteswapping
            return self._get_decode_r()
        else:
            raise RuntimeError("unknown type of MIB file")

    def get_native_dtype(self, inp_native_dtype, read_dtype):
        if self._kind == "u":
            # drop the byteswap from the dtype, if it is there
            return inp_native_dtype.newbyteorder('N')
        else:
            # decode byte-by-byte
            return np.dtype("u1")


class MIBHeaderReader:
    def __init__(self, path, fields=None, sequence_start=None):
        self.path = path
        if fields is None:
            self._fields = {}
        else:
            self._fields = fields
        self._sequence_start = sequence_start

    def __repr__(self):
        return "<MIBHeaderReader: %s>" % self.path

    def _get_np_dtype(self, dtype, bit_depth):
        dtype = dtype.lower()
        num_bits = int(dtype[1:])
        if dtype[0] == "u":
            num_bytes = num_bits // 8
            return np.dtype(">u%d" % num_bytes)
        elif dtype[0] == "r":
            if bit_depth == 1:
                return np.dtype("uint64")
            elif bit_depth == 6:
                return np.dtype("uint8")
            elif bit_depth in (12, 24):  # 24bit raw is two 12bit images after another
                return np.dtype("uint16")
            else:
                raise NotImplementedError("unknown bit depth: %s" % bit_depth)

    def read_header(self):
        # FIXME: do this read via the IO backend!
        with io.open(file=self.path, mode="r", encoding="ascii", errors='ignore') as f:
            header = f.read(1024)
            filesize = os.fstat(f.fileno()).st_size
        parts = header.split(",")
        header_size_bytes = int(parts[2])
        parts = [p
                for p in header[:header_size_bytes].split(",")
                if '\x00' not in p]
        self._header_parts = parts
        dtype = parts[6].lower()
        mib_kind = dtype[0]
        image_size = (int(parts[5]), int(parts[4]))
        # FIXME: There can either be threshold values for all chips, or maybe also
        # none. For now, we just make use of the fact that the bit depth is
        # supposed to be the last value.
        bits_per_pixel_raw = int(parts[-1])
        if mib_kind == "u":
            bytes_per_pixel = int(parts[6][1:]) // 8
            image_size_bytes = image_size[0] * image_size[1] * bytes_per_pixel
            num_images = filesize // (
                image_size_bytes + header_size_bytes
            )
        elif mib_kind == "r":
            size_factor = {
                1: 1/8,
                6: 1,
                12: 2,
                24: 4,
            }[bits_per_pixel_raw]
            if bits_per_pixel_raw == 24:
                image_size = (image_size[0], image_size[1] // 2)
            image_size_bytes = int(image_size[0] * image_size[1] * size_factor)
            num_images = filesize // (
                image_size_bytes + header_size_bytes
            )
        else:
            raise ValueError("unknown kind: %s" % mib_kind)

        self._fields = {
            'header_size_bytes': header_size_bytes,
            'dtype': self._get_np_dtype(parts[6], bits_per_pixel_raw),
            'mib_dtype': dtype,
            'mib_kind': mib_kind,
            'bits_per_pixel': bits_per_pixel_raw,
            'image_size': image_size,
            'image_size_bytes': image_size_bytes,
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


class MIBFile(LocalFile):
    def __init__(self, header, *args, **kwargs):
        self._header = header
        super().__init__(*args, **kwargs)

    def _mmap_to_array(self, raw_mmap, start, stop):
        res = np.frombuffer(raw_mmap, dtype="uint8")
        cutoff = self._header['num_images'] * (
            self._header['image_size_bytes'] + self._header['header_size_bytes']
        )
        res = res[:cutoff]
        return res.view(dtype=self._native_dtype).reshape(
            (self.num_frames, -1)
        )[:, start:stop]


class MIBFileSet(FileSet):
    def __init__(self, kind, dtype, bit_depth, *args, **kwargs):
        self._kind = kind
        self._mib_dtype = dtype
        self._bit_depth = bit_depth
        super().__init__(*args, **kwargs)

    def _clone(self, *args, **kwargs):
        return self.__class__(
            kind=self._kind, dtype=self._mib_dtype, bit_depth=self._bit_depth,
            *args, **kwargs
        )

    def get_read_ranges(
        self, start_at_frame: int, stop_before_frame: int,
        dtype, tiling_scheme: TilingScheme, sync_offset: int = 0,
        roi: typing.Union[np.ndarray, None] = None,
    ):
        fileset_arr = self.get_as_arr()
        bit_depth = self._bit_depth
        kwargs = dict(
            start_at_frame=start_at_frame,
            stop_before_frame=stop_before_frame,
            roi=roi,
            depth=tiling_scheme.depth,
            slices_arr=tiling_scheme.slices_array,
            fileset_arr=fileset_arr,
            sig_shape=tuple(tiling_scheme.dataset_shape.sig),
            sync_offset=sync_offset,
            bpp=np.dtype(dtype).itemsize,
            frame_header_bytes=self._frame_header_bytes,
            frame_footer_bytes=self._frame_footer_bytes,
        )
        if self._kind == "r" and bit_depth in (1,):
            return mib_r_get_read_ranges(**kwargs)
        elif self._kind == "r" and bit_depth in (24,):
            return mib_r24_get_read_ranges(**kwargs)
        else:
            return default_get_read_ranges(**kwargs)


class MIBDataSet(DataSet):
    # FIXME include sample file for doctest, see Issue #86
    """
    MIB data sets consist of one or more `.mib` files, and optionally
    a `.hdr` file. The HDR file is used to automatically set the
    `nav_shape` parameter from the fields "Frames per Trigger" and "Frames
    in Acquisition." When loading a MIB data set, you can either specify
    the path to the HDR file, or choose one of the MIB files. The MIB files
    are assumed to follow a naming pattern of some non-numerical prefix,
    and a sequential numerical suffix.

    Note that if you are using a per-pixel trigger setup, LiberTEM won't
    be able to deduce the x scanning dimension - in that case, you will
    need to specify the `nav_shape` yourself.

    Examples
    --------

    >>> # both examples look for files matching /path/to/default*.mib:
    >>> ds1 = ctx.load("mib", path="/path/to/default.hdr")  # doctest: +SKIP
    >>> ds2 = ctx.load("mib", path="/path/to/default64.mib")  # doctest: +SKIP

    Parameters
    ----------
    path: str
        Path to either the .hdr file or one of the .mib files

    nav_shape: tuple of int, optional
        A n-tuple that specifies the size of the navigation region ((y, x), but
        can also be of length 1 for example for a line scan, or length 3 for
        a data cube, for example)

    sig_shape: tuple of int, optional
        Common case: (height, width); but can be any dimensionality

    sync_offset: int, optional
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start
    """
    def __init__(self, path, tileshape=None, scan_size=None, disable_glob=False,
                 nav_shape=None, sig_shape=None, sync_offset=0, io_backend=None):
        super().__init__(io_backend=io_backend)
        self._sig_dims = 2
        self._path = path
        self._nav_shape = tuple(nav_shape) if nav_shape else nav_shape
        self._sig_shape = tuple(sig_shape) if sig_shape else sig_shape
        self._sync_offset = sync_offset
        # handle backwards-compatability:
        if tileshape is not None:
            warnings.warn(
                "tileshape argument is ignored and will be removed after 0.6.0",
                FutureWarning
            )
        if scan_size is not None:
            warnings.warn(
                "scan_size argument is deprecated. please specify nav_shape instead",
                FutureWarning
            )
            if nav_shape is not None:
                raise ValueError("cannot specify both scan_size and nav_shape")
            self._nav_shape = tuple(scan_size)
        if self._nav_shape is None and not path.lower().endswith(".hdr"):
            raise ValueError(
                    "either nav_shape needs to be passed, or path needs to point to a .hdr file"
                )
        self._filename_cache = None
        self._files_sorted = None
        # ._preread_headers() calls ._files() which passes the cached headers down to
        # MIBHeaderReader, if they exist. So we need to make sure to initialize self._headers
        # before calling _preread_headers!
        self._headers = {}
        self._meta = None
        self._total_filesize = None
        self._sequence_start = None
        self._disable_glob = disable_glob

    def _do_initialize(self):
        self._headers = self._preread_headers()
        self._files_sorted = list(sorted(self._files(),
                                         key=lambda f: f.fields['sequence_first_image']))

        try:
            first_file = self._files_sorted[0]
        except IndexError:
            raise DataSetException("no files found")
        if self._nav_shape is None:
            hdr = read_hdr_file(self._path)
            self._nav_shape = nav_shape_from_hdr(hdr)
        if self._sig_shape is None:
            self._sig_shape = first_file.fields['image_size']
        elif int(np.prod(self._sig_shape)) != int(np.prod(first_file.fields['image_size'])):
            raise DataSetException(
                "sig_shape must be of size: %s" % int(np.prod(first_file.fields['image_size']))
            )
        self._sig_dims = len(self._sig_shape)
        shape = Shape(self._nav_shape + self._sig_shape, sig_dims=self._sig_dims)
        dtype = first_file.fields['dtype']
        self._total_filesize = sum(
            os.stat(path).st_size
            for path in self._filenames()
        )
        self._sequence_start = first_file.fields['sequence_first_image']
        self._files_sorted = list(sorted(self._files(),
                                         key=lambda f: f.fields['sequence_first_image']))
        self._image_count = self._num_images()
        self._nav_shape_product = int(np.prod(self._nav_shape))
        self._sync_offset_info = self.get_sync_offset_info()
        self._meta = DataSetMeta(
            shape=shape,
            raw_dtype=dtype,
            sync_offset=self._sync_offset,
            image_count=self._image_count,
        )
        return self

    def initialize(self, executor):
        return executor.run_function(self._do_initialize)

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
    def get_supported_extensions(cls):
        return set(["mib", "hdr"])

    @classmethod
    def detect_params(cls, path, executor):
        pathlow = path.lower()
        if pathlow.endswith(".mib"):
            image_count, sig_shape = executor.run_function(get_image_count_and_sig_shape, path)
            nav_shape = tuple((image_count,))
        elif pathlow.endswith(".hdr") and executor.run_function(is_valid_hdr, path):
            hdr = executor.run_function(read_hdr_file, path)
            image_count, sig_shape = executor.run_function(get_image_count_and_sig_shape, path)
            nav_shape = nav_shape_from_hdr(hdr)
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

    def _preread_headers(self):
        res = {}
        for f in self._files():
            res[f.path] = f.fields
        return res

    def _filenames(self):
        if self._filename_cache is not None:
            return self._filename_cache
        fns = get_filenames(self._path)
        if len(fns) > 16384:
            warnings.warn(
                "Saving data in many small files (here: %d) is not efficient, please increase "
                "the \"Images Per File\" parameter when acquiring data",
                RuntimeWarning
            )
        self._filename_cache = fns
        return fns

    def _files(self):
        for path in self._filenames():
            f = MIBHeaderReader(path, fields=self._headers.get(path),
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
        the shape specified or imprinted by nav_shape from the HDR file
        """
        return self._meta.shape

    def check_valid(self):
        pass

    def get_cache_key(self):
        return {
            "path": self._path,
            # shape is included here because the structure will be invalid if you open
            # the same .mib file with a different nav_shape; should be no issue if you
            # open via the .hdr file
            "shape": tuple(self.shape),
            "sync_offset": self._sync_offset,
        }

    def _get_fileset(self):
        assert self._sequence_start is not None
        first_file = self._files_sorted[0]
        dtype = first_file.fields['dtype']
        kind = first_file.fields['mib_kind']
        bit_depth = first_file.fields['bits_per_pixel']
        header_size = first_file.fields['header_size_bytes']
        return MIBFileSet(files=[
            MIBFile(
                path=f.path,
                start_idx=f.start_idx,
                end_idx=f.start_idx + f.num_frames,
                native_dtype=f.fields['dtype'],
                sig_shape=self._meta.shape.sig,
                frame_header=f.fields['header_size_bytes'],
                file_header=0,
                header=f.fields,
            )
            for f in self._files_sorted
        ], dtype=dtype, kind=kind, bit_depth=bit_depth, frame_header_bytes=header_size)

    def get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 512MB (converted float data) per partition
        partition_size_px = 512 * 1024 * 1024 // 4
        total_size_px = np.prod(self.shape, dtype=np.int64)
        res = max(self._cores, total_size_px // partition_size_px)
        return res

    def get_partitions(self):
        first_file = self._files_sorted[0]
        fileset = self._get_fileset()
        kind = first_file.fields['mib_kind']
        for part_slice, start, stop in self.get_slices():
            yield MIBPartition(
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
                kind=kind,
                bit_depth=first_file.fields['bits_per_pixel'],
                io_backend=self.get_io_backend(),
            )

    def __repr__(self):
        return "<MIBDataSet of %s shape=%s>" % (self.dtype, self.shape)


class MIBPartition(BasePartition):
    def __init__(self, kind, bit_depth, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kind = kind
        self._bit_depth = bit_depth

    def _get_decoder(self):
        return MIBDecoder(
            kind=self._kind,
            dtype=self.meta.raw_dtype,
            bit_depth=self._bit_depth,
        )
