import re
import os
import platform
from glob import glob, escape
import logging
from typing import TYPE_CHECKING, Optional, Union
from collections.abc import Generator, Sequence
from typing_extensions import Literal, TypedDict
import warnings
from concurrent.futures import ThreadPoolExecutor

from numba.typed import List as NumbaList
import numba
import psutil
import numpy as np

from libertem.common.math import prod, make_2D_square, flat_nonzero
from libertem.common import Shape
from libertem.io.dataset.base.file import OffsetsSizes
from libertem.common.messageconverter import MessageConverter
from .base import (
    DataSet, DataSetException, DataSetMeta,
    BasePartition, FileSet, File, make_get_read_ranges,
    Decoder, TilingScheme, default_get_read_ranges,
    DtypeConversionDecoder, IOBackend,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy.typing as nt


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
            "io_backend": {
                "enum": IOBackend.get_supported(),
            },
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
    with open(path, encoding='utf-8', errors='ignore') as f:
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
    with open(path, encoding='utf-8', errors='ignore') as f:
        line = next(f)
        return line.startswith("HDR")


def nav_shape_from_hdr(hdr):
    if 'ScanX' in hdr and 'ScanY' in hdr:
        return (int(hdr['ScanY']), int(hdr['ScanX']))
    num_frames, scan_x = (
        int(hdr['Frames in Acquisition (Number)']),
        int(hdr['Frames per Trigger (Number)'])
    )
    nav_shape = (num_frames // scan_x, scan_x)
    return nav_shape


def _pattern(path: str) -> str:
    path, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == '.mib':
        pattern = "%s*.mib" % (
            re.sub(r'[0-9]+$', '', escape(path))
        )
    elif ext == '.hdr':
        pattern = "%s*.mib" % escape(path)
    else:
        raise DataSetException("unknown extension")
    return pattern


def get_filenames(path: str, disable_glob=False) -> list[str]:
    if disable_glob:
        return [path]
    else:
        return glob(_pattern(path))


def _get_sequence(f: "MIBHeaderReader"):
    return f.fields['sequence_first_image']


def get_image_count_and_sig_shape(
    path: str,
    disable_glob: bool = False,
) -> tuple[int, tuple[int, int]]:
    fns = get_filenames(path, disable_glob=disable_glob)
    headers = MIBDataSet._preread_headers(fns)
    count = 0
    files = []
    for path, fields in headers.items():
        f = MIBHeaderReader(path, fields=fields)
        count += f.fields['num_images']
        files.append(f)
    try:
        first_file = list(sorted(files, key=_get_sequence))[0]
        sig_shape = first_file.fields['image_size']
        return count, sig_shape
    except IndexError:
        raise DataSetException("no files found")


# These encoders takes 2D input/output data - this means we can use
# strides to do slicing and reversing. 2D input data means one output
# row (of bytes) corresponds to one input row (of pixels).


@numba.njit(cache=True)
def encode_u1(inp, out):
    for y in range(out.shape[0]):
        out[y] = inp[y]


@numba.njit(cache=True)
def encode_u2(inp, out):
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]
        for i in range(row_in.shape[0]):
            in_value = row_in[i]
            row_out[i * 2] = (0xFF00 & np.uint16(in_value)) >> 8
            row_out[i * 2 + 1] = 0xFF & np.uint16(in_value)


@numba.njit(cache=True)
def encode_r1(inp, out):
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]
        for stripe in range(row_out.shape[0] // 8):
            for byte in range(8):
                out_byte = 0
                for bitpos in range(8):
                    value = row_in[64 * stripe + 8 * byte + bitpos] & 1
                    out_byte |= (np.uint64(value) << bitpos)
                row_out[(stripe + 1) * 8 - (byte + 1)] = out_byte


@numba.njit(cache=True)
def encode_r6(inp, out):
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]
        for i in range(row_out.shape[0]):
            col = i % 8
            pos = i // 8
            in_pos = (pos + 1) * 8 - col - 1
            row_out[i] = row_in[in_pos]


@numba.njit(cache=True)
def encode_r12(inp, out):
    for y in range(out.shape[0]):
        row_out = out[y]
        row_in = inp[y]
        for i in range(row_in.shape[0]):
            col = i % 4
            pos = i // 4
            in_pos = (pos + 1) * 4 - col - 1
            in_value = row_in[in_pos]
            row_out[i * 2] = (0xFF00 & in_value) >> 8
            row_out[i * 2 + 1] = 0xFF & in_value


@numba.njit(inline='always', cache=True)
def _get_row_start_stop(offset_global, offset_local, stride, row_idx, row_length):
    start = offset_global + offset_local + row_idx * stride
    stop = start + row_length
    return start, stop


@numba.njit(inline='always', cache=True)
def _mib_r24_px_to_bytes(
    bpp, frame_in_file_idx, slice_sig_size, sig_size, sig_origin,
    frame_footer_bytes, frame_header_bytes, file_header_bytes,
    file_idx, read_ranges,
):
    # we are reading a part of a single frame, so we first need to find
    # the offset caused by headers and footers:
    footer_offset = frame_footer_bytes * frame_in_file_idx
    header_offset = frame_header_bytes * (frame_in_file_idx + 1)
    byte_offset = footer_offset + header_offset

    # now let's figure in the current frame index:
    # (go down into the file by full frames; `sig_size`)
    offset = file_header_bytes + byte_offset + frame_in_file_idx * sig_size * bpp // 8

    # offset in px in the current frame:
    sig_origin_bytes = sig_origin * bpp // 8

    start = offset + sig_origin_bytes

    # size of the sig part of the slice:
    sig_size_bytes = slice_sig_size * bpp // 8

    stop = start + sig_size_bytes
    read_ranges.append((file_idx, start, stop))

    # this is the addition for medipix 24bit raw:
    # we read the part from the "second 12bit frame"
    second_frame_offset = sig_size * bpp // 8
    read_ranges.append((file_idx, start + second_frame_offset, stop + second_frame_offset))


mib_r24_get_read_ranges = make_get_read_ranges(px_to_bytes=_mib_r24_px_to_bytes)


@numba.njit(inline="always", cache=True)
def _mib_2x2_tile_block(
    slices_arr, fileset_arr, slice_sig_sizes, sig_origins,
    inner_indices_start, inner_indices_stop, frame_indices, sig_size,
    px_to_bytes, bpp, frame_header_bytes, frame_footer_bytes, file_idxs,
    slice_offset, extra, sig_shape,
):
    """
    Generate read ranges for 2x2 Merlin Quad raw data.

    The arrangement means that reading a contiguous block of data from the file,
    we get data from all four quadrants. The arrangement, and thus resulting
    array, looks like this:

    _________
    | 1 | 2 |
    ---------
    | 3 | 4 |
    ---------

    with the original data layed out like this:

    [4 | 3 | 2 | 1]

    (note that quadrants 3 and 4 are also flipped in x and y direction in the
    resulting array, compared to the original data)

    So if we read one row of raw data, we first get the bottom-most rows from 4
    and 3 first, then the top-most rows from 2 and 1.

    This is similar to how FRMS6 works, and we generate the read ranges in a
    similar way here. In addition to the cut-and-flip from FRMS6, we also have
    the split in x direction in quadrants.
    """
    result = NumbaList()

    # positions in the signal dimensions:
    for slice_idx in range(slices_arr.shape[0]):
        # (offset, size) arrays defining what data to read (in pixels)
        # NOTE: assumes contiguous tiling scheme
        # (i.e. a shape like (1, 1, ..., 1, X1, ..., XN))
        # where X1 is <= the dataset shape at that index, and X2, ..., XN are
        # equal to the dataset shape at that index
        slice_origin = slices_arr[slice_idx][0]
        slice_shape = slices_arr[slice_idx][1]

        read_ranges = NumbaList()

        x_shape = slice_shape[1]
        x_size = x_shape * bpp // 8  # back in bytes
        x_size_half = x_size // 2
        stride = x_size_half * 4

        sig_size_bytes = sig_size * bpp // 8

        y_start = slice_origin[0]
        y_stop = slice_origin[0] + slice_shape[0]

        y_size_half = sig_shape[0] // 2
        y_size = sig_shape[0]

        # inner "depth" loop along the (flat) navigation axis of a tile:
        for i, inner_frame_idx in enumerate(range(inner_indices_start, inner_indices_stop)):
            inner_frame = frame_indices[inner_frame_idx]
            file_idx = file_idxs[i]
            f = fileset_arr[file_idx]
            frame_in_file_idx = inner_frame - f[0]
            file_header_bytes = f[3]

            # we are reading a part of a single frame, so we first need to find
            # the offset caused by headers:
            header_offset = file_header_bytes + frame_header_bytes * (frame_in_file_idx + 1)

            # now let's figure in the current frame index:
            # (go down into the file by full frames; `sig_size`)
            offset = header_offset + frame_in_file_idx * sig_size_bytes

            # in total, we generate depth * 2 * (y_stop - y_start) read ranges per tile
            for y in range(y_start, y_stop):
                if y < y_size_half:

                    # top: no y-flip, no x-flip
                    flip = 0

                    # quadrant 1, left part of the result: we have the three other blocks
                    # in the original data in front of us
                    start, stop = _get_row_start_stop(
                        offset, 3 * x_size_half, stride, y, x_size_half
                    )
                    read_ranges.append((
                        file_idx,
                        start,
                        stop,
                        flip,
                    ))
                    # quadrant 2, right part of the result: we have the two other blocks
                    # in the original data in front of us
                    start, stop = _get_row_start_stop(
                        offset, 2 * x_size_half, stride, y, x_size_half
                    )
                    read_ranges.append((
                        file_idx,
                        start,
                        stop,
                        flip,
                    ))
                else:
                    # bottom: both x and y flip
                    flip = 1
                    y = y_size - y - 1
                    # quadrant 3, left part of the result: we have the one other block
                    # in the original data in front of us
                    start, stop = _get_row_start_stop(
                        offset, 1 * x_size_half, stride, y, x_size_half
                    )
                    read_ranges.append((
                        file_idx,
                        start,
                        stop,
                        flip,
                    ))
                    # quadrant 4, right part of the result: we have the no other blocks
                    # in the original data in front of us
                    start, stop = _get_row_start_stop(offset, 0, stride, y, x_size_half)
                    read_ranges.append((
                        file_idx,
                        start,
                        stop,
                        flip,
                    ))

        # the indices are compressed to the selected frames
        compressed_slice = np.array([
            [slice_offset + inner_indices_start] + [i for i in slice_origin],
            [inner_indices_stop - inner_indices_start] + [i for i in slice_shape],
        ])
        result.append((slice_idx, compressed_slice, read_ranges))

    return result


@numba.njit(inline='always', cache=True, boundscheck=True)
def decode_r1_swap_2x2(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    """
    RAW 1bit format: each pixel is actually saved as a single bit. 64 bits
    need to be unpacked together. This is the quad variant.

    Parameters
    ==========
    inp : np.ndarray

    out : np.ndarray
        The output buffer, with the signal dimensions flattened

    idx : int
        The index in the read ranges array

    native_dtype : nt.DtypeLike
        The "native" dtype (format-specific)

    rr : np.ndarray
        A single entry from the read ranges array

    origin : np.ndarray
        The 3D origin of the tile, for example :code:`np.array([2, 0, 0])`

    shape : np.ndarray
        The 3D tileshape, for example :code:`np.array([2, 512, 512])`

    ds_shape : np.ndarray
        The complete ND dataset shape, for example :code:`np.array([32, 32, 512, 512])`
    """
    # in case of 2x2 quad, the index into `out` is not straight `idx`, but
    # we have `2 * shape[1]` read ranges generated for one depth.
    num_rows_tile = shape[1]

    out_3d = out.reshape(out.shape[0], -1, shape[-1])

    # each line in the output array generates two entries in the
    # read ranges. so `(idx // 2) % num_rows_tile` is the correct
    # out_y:
    out_y = (idx // 2) % num_rows_tile
    out_x_start = (idx % 2) * (shape[-1] // 2)

    depth = idx // (num_rows_tile * 2)
    flip = rr[3]

    if flip == 0:
        for stripe in range(inp.shape[0] // 8):
            for byte in range(8):
                inp_byte = inp[(stripe + 1) * 8 - (byte + 1)]
                for bitpos in range(8):
                    out_x = 64 * stripe + 8 * byte + bitpos
                    out_x += out_x_start
                    out_3d[depth, out_y, out_x] = (inp_byte >> bitpos) & 1
    else:
        # flip in x direction:
        x_shape = shape[2]
        x_shape_half = x_shape // 2
        for stripe in range(inp.shape[0] // 8):
            for byte in range(8):
                inp_byte = inp[(stripe + 1) * 8 - (byte + 1)]
                for bitpos in range(8):
                    out_x = out_x_start + x_shape_half - 1 - (64 * stripe + 8 * byte + bitpos)
                    out_3d[depth, out_y, out_x] = (inp_byte >> bitpos) & 1


@numba.njit(inline='always', cache=True, boundscheck=True)
def decode_r6_swap_2x2(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    """
    RAW 6bit format: the pixels need to be re-ordered in groups of 8. `inp`
    should have dtype uint8. This is the quad variant.

    Parameters
    ==========
    inp : np.ndarray

    out : np.ndarray
        The output buffer, with the signal dimensions flattened

    idx : int
        The index in the read ranges array

    native_dtype : nt.DtypeLike
        The "native" dtype (format-specific)

    rr : np.ndarray
        A single entry from the read ranges array

    origin : np.ndarray
        The 3D origin of the tile, for example :code:`np.array([2, 0, 0])`

    shape : np.ndarray
        The 3D tileshape, for example :code:`np.array([2, 512, 512])`

    ds_shape : np.ndarray
        The complete ND dataset shape, for example :code:`np.array([32, 32, 512, 512])`
    """
    # in case of 2x2 quad, the index into `out` is not straight `idx`, but
    # we have `2 * shape[1]` read ranges generated for one depth.
    num_rows_tile = shape[1]

    out_3d = out.reshape(out.shape[0], -1, shape[-1])

    # each line in the output array generates two entries in the
    # read ranges. so `(idx // 2) % num_rows_tile` is the correct
    # out_y:
    out_y = (idx // 2) % num_rows_tile
    out_x_start = (idx % 2) * (shape[-1] // 2)

    depth = idx // (num_rows_tile * 2)
    flip = rr[3]

    out_cut = out_3d[depth, out_y, out_x_start:out_x_start + out_3d.shape[2] // 2]

    if flip == 0:
        for i in range(out_cut.shape[0]):
            col = i % 8
            pos = i // 8
            out_pos = (pos + 1) * 8 - col - 1
            out_cut[out_pos] = inp[i]
    else:
        # flip in x direction:
        for i in range(out_cut.shape[0]):
            col = i % 8
            pos = i // 8
            out_pos = (pos + 1) * 8 - col - 1
            out_cut[out_cut.shape[0] - out_pos - 1] = inp[i]


@numba.njit(inline='always', cache=True, boundscheck=True)
def decode_r12_swap_2x2(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    """
    RAW 12bit format: the pixels need to be re-ordered in groups of 4. `inp`
    should be an uint8 view on padded big endian 12bit data (">u2").
    This is the quad variant.

    Parameters
    ==========
    inp : np.ndarray

    out : np.ndarray
        The output buffer, with the signal dimensions flattened

    idx : int
        The index in the read ranges array

    native_dtype : nt.DtypeLike
        The "native" dtype (format-specific)

    rr : np.ndarray
        A single entry from the read ranges array

    origin : np.ndarray
        The 3D origin of the tile, for example :code:`np.array([2, 0, 0])`

    shape : np.ndarray
        The 3D tileshape, for example :code:`np.array([2, 512, 512])`

    ds_shape : np.ndarray
        The complete ND dataset shape, for example :code:`np.array([32, 32, 512, 512])`
    """
    # in case of 2x2 quad, the index into `out` is not straight `idx`, but
    # we have `2 * shape[1]` read ranges generated for one depth.
    num_rows_tile = shape[1]

    out_3d = out.reshape(out.shape[0], -1, shape[-1])

    # each line in the output array generates two entries in the
    # read ranges. so `(idx // 2) % num_rows_tile` is the correct
    # out_y:
    out_y = (idx // 2) % num_rows_tile
    out_x_start = (idx % 2) * (shape[-1] // 2)

    depth = idx // (num_rows_tile * 2)
    flip = rr[3]

    out_cut = out_3d[depth, out_y, out_x_start:out_x_start + out_3d.shape[2] // 2]

    if flip == 0:
        for i in range(out_cut.shape[0]):
            col = i % 4
            pos = i // 4
            out_pos = (pos + 1) * 4 - col - 1
            out_cut[out_pos] = (np.uint16(inp[i * 2]) << 8) + (inp[i * 2 + 1] << 0)
    else:
        # flip in x direction:
        for i in range(out_cut.shape[0]):
            col = i % 4
            pos = i // 4
            out_pos = (pos + 1) * 4 - col - 1
            out_cut[out_cut.shape[0] - out_pos - 1] = (
                (np.uint16(inp[i * 2]) << 8) + (inp[i * 2 + 1] << 0)
            )

    # reference non-quad impl:
    # for i in range(out.shape[1]):
    #     col = i % 4
    #     pos = i // 4
    #     out_pos = (pos + 1) * 4 - col - 1
    #     out[idx, out_pos] = (inp[i * 2] << 8) + (inp[i * 2 + 1] << 0)


mib_2x2_get_read_ranges = make_get_read_ranges(
    read_ranges_tile_block=_mib_2x2_tile_block
)


@numba.njit(inline='always', cache=True)
def decode_r1_swap(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    """
    RAW 1bit format: each pixel is actually saved as a single bit. 64 bits
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
        out[idx, out_pos] = (np.uint16(inp[i * 2]) << 8) + (np.uint16(inp[i * 2 + 1]) << 0)


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
        out_val = np.uint32((np.uint32(inp[i * 2]) << 8) + (np.uint32(inp[i * 2 + 1]) << 0))
        if idx % 2 == 0:  # from first frame: most significant bits
            out_val = out_val << 12
        out[idx // 2, out_pos] += out_val


class MIBDecoder(Decoder):
    def __init__(self, header: "HeaderDict"):
        self._kind = header['mib_kind']
        self._dtype = header['dtype']
        self._bit_depth = header['bits_per_pixel']
        self._header = header

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
        bit_depth = self._bit_depth
        layout = self._header['sensor_layout']
        num_chips = self._header['num_chips']
        if layout == (2, 2) and num_chips == 4:
            if bit_depth == 1:
                return decode_r1_swap_2x2
            elif bit_depth == 6:
                return decode_r6_swap_2x2
            elif bit_depth == 12:
                return decode_r12_swap_2x2
            else:
                raise NotImplementedError(
                    f"bit depth {bit_depth} not implemented for layout {layout}"
                )
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


MIBKind = Literal['u', 'r']


class HeaderDict(TypedDict):
    header_size_bytes: int
    dtype: "nt.DTypeLike"
    mib_dtype: str
    mib_kind: MIBKind
    bits_per_pixel: int
    image_size: tuple[int, int]
    image_size_bytes: int
    sequence_first_image: int
    filesize: int
    num_images: int
    num_chips: int
    sensor_layout: tuple[int, int]


class MIBHeaderReader:
    def __init__(
        self,
        path: str,
        fields: Optional[HeaderDict] = None,
        sequence_start: int = 0,
    ):
        self.path = path
        if fields is None:
            self._fields = None
        else:
            self._fields = fields
        self._sequence_start = sequence_start

    def __repr__(self) -> str:
        return "<MIBHeaderReader: %s>" % self.path

    @staticmethod
    def _get_np_dtype(dtype: str, bit_depth: int) -> "nt.DTypeLike":
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
        else:
            raise NotImplementedError(f"unknown dtype: {dtype}")

    def read_header(self) -> HeaderDict:
        header, filesize = self._read_header_bytes(self.path)
        self._fields = self._parse_header_bytes(header, filesize)
        return self._fields

    @staticmethod
    def _read_header_bytes(path) -> tuple[bytes, int]:
        # FIXME: do this read via the IO backend!
        with open(file=path, mode='rb') as f:
            filesize = os.fstat(f.fileno()).st_size
            return f.read(1024), filesize

    @staticmethod
    def _parse_header_bytes(header: bytes, filesize: int) -> HeaderDict:
        header: str = header.decode(encoding='ascii', errors='ignore')
        parts = header.split(",")
        header_size_bytes = int(parts[2])
        parts = [p
                for p in header[:header_size_bytes].split(",")
                if '\x00' not in p]
        dtype = parts[6].lower()
        mib_kind: Literal['u', 'r']
        # To make mypy Literal check happy...
        if dtype[0] == "r":
            mib_kind = "r"
        elif dtype[0] == "u":
            mib_kind = "u"
        else:
            raise ValueError("unknown kind: %s" % dtype[0])
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
        num_chips = int(parts[3])

        # 2x2, Nx1, 2x2G, Nx1G -> strip off the 'G' suffix for now:
        # Assumption: layout is width x height

        # We currently only support 1x1 and 2x2 layouts, and support for raw
        # files with 2x2 layout is limited to 1, 6 or 12 bits. Support for 24bit TBD.
        # We don't yet support general Nx1 layouts, or arbitrary "chip select"
        # values (2x2 layout, but only a subset of them is enabled).
        sensor_layout_str = parts[7].replace('G', '').split('x')
        sensor_layout = (
            int(sensor_layout_str[0]),
            int(sensor_layout_str[1]),
        )

        if mib_kind == "r":
            # in raw data, the layout information is not decoded yet - we just get
            # rows from different sensors after one another. The image size is
            # accordingly also different.

            # NOTE: this is still a bit hacky - individual sensors in a layout
            # can be enabled or disabled; because we only support 1x1 and 2x2
            # layouts for now, we just do the "layout based reshaping" only
            # in the 2x2 case:
            if num_chips > 1:
                px_length = image_size[0]
                image_size_orig = image_size
                image_size = (
                    px_length * sensor_layout[1],
                    px_length * sensor_layout[0],
                )
                if prod(image_size_orig) != prod(image_size):
                    raise ValueError(
                        f"invalid sensor layout {sensor_layout} "
                        f"(original image size: {image_size_orig})"
                    )

        return {
            'header_size_bytes': header_size_bytes,
            'dtype': MIBHeaderReader._get_np_dtype(parts[6], bits_per_pixel_raw),
            'mib_dtype': dtype,
            'mib_kind': mib_kind,
            'bits_per_pixel': bits_per_pixel_raw,
            'image_size': image_size,
            'image_size_bytes': image_size_bytes,
            'sequence_first_image': int(parts[1]),
            'filesize': filesize,
            'num_images': num_images,
            'num_chips': num_chips,
            'sensor_layout': sensor_layout,
        }

    @property
    def num_frames(self) -> int:
        return self.fields['num_images']

    @property
    def start_idx(self) -> int:
        return self.fields['sequence_first_image'] - self.sequence_start

    @property
    def sequence_start(self) -> Optional[int]:
        return self._sequence_start

    @sequence_start.setter
    def sequence_start(self, val: int):
        self._sequence_start = val

    @property
    def fields(self) -> HeaderDict:
        if self._fields is None:
            self._fields = self.read_header()
        return self._fields


class MIBFile(File):
    def __init__(self, header, *args, **kwargs):
        self._header = header
        super().__init__(*args, **kwargs)

    def get_array_from_memview(self, mem: memoryview, slicing: OffsetsSizes):
        mem = mem[slicing.file_offset:-slicing.skip_end]
        res = np.frombuffer(mem, dtype="uint8")
        cutoff = self._header['num_images'] * (
            self._header['image_size_bytes'] + self._header['header_size_bytes']
        )
        res = res[:cutoff]
        return res.view(dtype=self._native_dtype).reshape(
            (self.num_frames, -1)
        )[:, slicing.frame_offset:slicing.frame_offset + slicing.frame_size]


class MIBFileSet(FileSet):
    def __init__(
        self,
        header: HeaderDict,
        *args, **kwargs
    ):
        self._header = header
        super().__init__(*args, **kwargs)

    def _clone(self, *args, **kwargs):
        return self.__class__(header=self._header, *args, **kwargs)

    def get_read_ranges(
        self, start_at_frame: int, stop_before_frame: int,
        dtype, tiling_scheme: TilingScheme, sync_offset: int = 0,
        roi: Union[np.ndarray, None] = None,
    ):
        fileset_arr = self.get_as_arr()
        bit_depth = self._header['bits_per_pixel']
        kind = self._header['mib_kind']
        layout = self._header['sensor_layout']
        if kind == "r" and bit_depth == 1:
            # special case for bit-packed 1-bit format:
            bpp = 1
        else:
            bpp = np.dtype(dtype).itemsize * 8
        roi_nonzero = None
        if roi is not None:
            roi_nonzero = flat_nonzero(roi).astype(np.int64)
        kwargs = dict(
            start_at_frame=start_at_frame,
            stop_before_frame=stop_before_frame,
            roi_nonzero=roi_nonzero,
            depth=tiling_scheme.depth,
            slices_arr=tiling_scheme.slices_array,
            fileset_arr=fileset_arr,
            sig_shape=tuple(tiling_scheme.dataset_shape.sig),
            sync_offset=sync_offset,
            bpp=bpp,
            frame_header_bytes=self._frame_header_bytes,
            frame_footer_bytes=self._frame_footer_bytes,
        )
        if layout == (1, 1) or self._header['num_chips'] == 1:
            if kind == "r" and bit_depth in (24,):
                return mib_r24_get_read_ranges(**kwargs)
            else:
                return default_get_read_ranges(**kwargs)
        elif layout == (2, 2):
            if kind == "r":
                return mib_2x2_get_read_ranges(**kwargs)
            else:
                return default_get_read_ranges(**kwargs)
        else:
            raise NotImplementedError(
                f"No support for layout {layout} yet - please contact us!"
            )


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

    Note that if you are using a per-pixel or per-scan trigger setup, LiberTEM
    won't be able to deduce the x scanning dimension - in that case, you will
    need to specify the `nav_shape` yourself.

    Currently, we support all integer formats, and most RAW formats. Especially,
    the following configurations are not yet supported for RAW files:

     * Non-2x2 layouts with more than one chip
     * 24bit with more than one chip

    .. versionadded:: 0.9.0
        Support for the raw quad format was added

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

    disable_glob : bool, default False
        Usually, MIB data sets are stored as a series of .mib files, and we can reliably
        guess the whole set from a single path. If you instead save your data set into
        a single .mib file, and have multiple of these in a single directory with the same prefix
        (for example, a.mib, a1.mib and a2.mib), loading a.mib would include a1.mib and a2.mib
        in the data set. Setting :code:`disable_glob` to :code:`True` will only load the single
        .mib file specified as :code:`path`.

    num_partitions: int, optional
        Override the number of partitions. This is useful if the
        default number of partitions, chosen based on common workloads,
        creates partitions which are too large (or small) for the UDFs
        being run on this dataset.
    """
    def __init__(
        self,
        path,
        tileshape=None,
        scan_size=None,
        disable_glob=False,
        nav_shape=None,
        sig_shape=None,
        sync_offset=0,
        io_backend=None,
        num_partitions=None,
    ):
        super().__init__(
            io_backend=io_backend,
            num_partitions=num_partitions,
        )
        self._sig_dims = 2
        self._path = str(path)
        self._nav_shape = tuple(nav_shape) if nav_shape else nav_shape
        self._sig_shape = tuple(sig_shape) if sig_shape else sig_shape
        self._sync_offset = sync_offset
        # handle backwards-compatibility:
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
        if self._nav_shape is None and not self._path.lower().endswith(".hdr"):
            raise ValueError(
                    "either nav_shape needs to be passed, or path needs to point to a .hdr file"
                )
        self._filename_cache = None
        self._files_sorted: Optional[Sequence[MIBHeaderReader]] = None
        # ._preread_headers() in _do_initialize() filles self._headers
        self._headers = {}
        self._meta = None
        self._total_filesize = None
        self._sequence_start = None
        self._disable_glob = disable_glob

    def _do_initialize(self):
        filenames = self._filenames()
        self._headers = self._preread_headers(filenames)
        self._files_sorted = list(sorted(self._files(),
                                         key=lambda f: f.fields['sequence_first_image']))

        try:
            first_file = self._files_sorted[0]
        except IndexError:
            raise DataSetException("no files found")

        self._sequence_start = first_file.fields['sequence_first_image']
        # _files_sorted is initially created with _sequence_start = None
        # now that we know the first file we can set the correct value
        for f in self._files_sorted:
            f.sequence_start = self._sequence_start

        if self._nav_shape is None:
            hdr = read_hdr_file(self._path)
            self._nav_shape = nav_shape_from_hdr(hdr)
        if self._sig_shape is None:
            self._sig_shape = first_file.fields['image_size']
        elif int(prod(self._sig_shape)) != int(prod(first_file.fields['image_size'])):
            raise DataSetException(
                "sig_shape must be of size: %s" % int(prod(first_file.fields['image_size']))
            )
        self._sig_dims = len(self._sig_shape)
        shape = Shape(self._nav_shape + self._sig_shape, sig_dims=self._sig_dims)
        dtype = first_file.fields['dtype']
        self._total_filesize = sum(
            f.fields['filesize']
            for f in self._files_sorted
        )
        self._image_count = self._num_images()
        self._nav_shape_product = int(prod(self._nav_shape))
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
        assert self._files_sorted is not None
        first_file = self._files_sorted[0]
        header = first_file.fields
        return [
            {"name": "Bits per pixel",
             "value": str(header['bits_per_pixel'])},
            {"name": "Data kind",
             "value": str(header['mib_kind'])},
            {"name": "Layout",
             "value": str(header['sensor_layout'])},
        ]

    @classmethod
    def get_msg_converter(cls):
        return MIBDatasetParams

    @classmethod
    def get_supported_extensions(cls):
        return {"mib", "hdr"}

    @classmethod
    def detect_params(cls, path, executor):
        pathlow = path.lower()
        if pathlow.endswith(".mib"):
            image_count, sig_shape = executor.run_function(get_image_count_and_sig_shape, path)
            nav_shape = make_2D_square((image_count,))
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

    @staticmethod
    def _preread_headers(filenames):
        # Avoid overhead of creating the Pool on low-file-count datasets
        if len(filenames) > 512:
            # Default ThreadPoolExecutor allocates 5 threads per CPU on the machine
            # In testing this was found to be far too many on Linux, and sometimes
            # a good number on Windows but sometimes too many
            max_workers = 2
            if platform.system() == "Windows":
                num_cores = psutil.cpu_count(logical=False)
                max_workers = max(max_workers, int(num_cores * 0.5))
            with ThreadPoolExecutor(max_workers=max_workers) as p:
                header_and_size = p.map(MIBHeaderReader._read_header_bytes, filenames)
        else:
            header_and_size = tuple(map(MIBHeaderReader._read_header_bytes, filenames))

        res = {}
        for path, (header, filesize) in zip(filenames, header_and_size):
            res[path] = MIBHeaderReader._parse_header_bytes(header, filesize)
        return res

    def _filenames(self):
        if self._filename_cache is not None:
            return self._filename_cache
        fns = get_filenames(self._path, disable_glob=self._disable_glob)
        num_files = len(fns)
        if num_files > 16384:
            warnings.warn(
                f"Saving data in many small files (here: {num_files}) is not efficient, "
                "please increase the \"Images Per File\" parameter when acquiring data. If "
                "this leads to \"Too many open files\" errors, on POSIX systems you can increase "
                "the maximum open files limit using the \"ulimit\" shell command.",
                RuntimeWarning
            )
        self._filename_cache = fns
        return fns

    def _files(self) -> Generator[MIBHeaderReader, None, None]:
        for path, fields in self._headers.items():
            f = MIBHeaderReader(path, fields=fields,
                        sequence_start=self._sequence_start)
            yield f

    def _num_images(self):
        return sum(f.fields['num_images'] for f in self._files_sorted)

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

    def get_decoder(self) -> Decoder:
        assert self._files_sorted is not None
        first_file = self._files_sorted[0]
        assert self.meta is not None
        return MIBDecoder(
            header=first_file.fields,
        )

    def _get_fileset(self):
        assert self._sequence_start is not None
        first_file = self._files_sorted[0]
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
        ], header=first_file.fields, frame_header_bytes=header_size)

    def get_base_shape(self, roi: Optional[np.ndarray]) -> tuple[int, ...]:
        # With R-mode files, we are constrained to tile sizes that are a
        # multiple of 64px in the fastest dimension!
        # If we make sure full "x-lines" are taken, we are fine (this is the
        # case by default)
        base_shape = super().get_base_shape(roi)
        assert self.meta is not None and base_shape[-1] == self.meta.shape[-1]
        return base_shape

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
                decoder=self.get_decoder(),
            )

    def __repr__(self):
        return f"<MIBDataSet of {self.dtype} shape={self.shape}>"


class MIBPartition(BasePartition):
    def __init__(self, kind, bit_depth, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kind = kind
        self._bit_depth = bit_depth
