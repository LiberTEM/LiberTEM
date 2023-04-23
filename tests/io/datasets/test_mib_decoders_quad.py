import pytest
import numpy as np
from numpy.testing import assert_allclose
from sparseconverter import NUMPY
from libertem.common.shape import Shape
from libertem.common.slice import Slice
from libertem.common.math import flat_nonzero
from libertem.io.dataset.base import TilingScheme
from libertem.io.dataset.base.backend_mmap import MMapBackendImpl, MMapFile

from libertem.io.dataset.mib import (
    encode_r1,
    encode_r6,
    encode_r12,
    HeaderDict,
    MIBDecoder,
    MIBFile,
    MIBFileSet,
    mib_2x2_get_read_ranges,
)


def encode_quad(encode, data, bits_per_pixel, with_headers=False):
    """
    Parameters
    ==========
    encode : Callable
        One of the `encode_r*` functions

    data : np.ndarray
        The array that should be encoded, with dtype int, shape (-1, 512, 512)

    bits_per_pixel : int
        One of 1, 8, 16 - the bits per pixels padded to byte boudaries

    with_headers : bool
        Will insert some random data between the frames, not real headers.
    """
    shape = data.shape
    num_frames = shape[0]
    # typically the header size for quad data, but doesn't really matter,
    # as we don't generate a "true" header, but just random padding at
    # the beginning of each frame.
    header_bytes = 768
    assert len(shape) == 3  # decoding multiple frames at once
    enc_bytes_per_frame = shape[1] * shape[2] // 8 * bits_per_pixel
    x_shape_px = 256 // 8 * bits_per_pixel

    encoded = np.zeros(
        data.size // 8 * bits_per_pixel + shape[0] * header_bytes,
        dtype=np.uint8
    )
    encoded = encoded.reshape((-1, enc_bytes_per_frame + header_bytes))

    # encoders only do one frame per call:
    for i in range(shape[0]):
        encoded[i, :header_bytes] = np.random.randint(0, 0x100, header_bytes)

        # reshape destination buffer to allow convenient row-based assignment:
        # dest = [4 | 3 | 2 | 1]
        dest = encoded[i, header_bytes:].reshape((256, -1))
        assert dest.shape == (256, 4 * x_shape_px)

        src = data[i]
        src_half = src.shape[0] // 2, src.shape[1] // 2

        q1 = src[:src_half[0], :src_half[1]]
        encode(inp=q1, out=dest[:, 3 * x_shape_px:])

        q2 = src[:src_half[0], src_half[1]:]
        encode(inp=q2, out=dest[:, 2 * x_shape_px:3 * x_shape_px])

        # q3/q4 flipped in y direction
        q3 = src[src_half[0]:, :src_half[1]][::-1, ::-1]
        encode(inp=q3, out=dest[:, 1 * x_shape_px:2 * x_shape_px])
        q4 = src[src_half[0]:, src_half[1]:][::-1, ::-1]
        encode(inp=q4, out=dest[:, 0 * x_shape_px:1 * x_shape_px])

    if with_headers:
        return encoded
    else:
        encoded_data = encoded.reshape((num_frames, -1,))[:, header_bytes:].reshape(
            (num_frames, data.shape[1], -1)
        )
        return encoded_data


class InMemoryFile(MMapFile):
    """
    For testing purposes, this `MMapFile` gets the mmap-like objects
    from the underlying `desc`.
    """
    def __init__(self, path, desc):
        super().__init__(path, desc)
        self._handle = None
        encoded_data = desc.data
        assert encoded_data.dtype == np.uint8
        self._mmap = encoded_data.reshape((-1,))
        self._array = self._mmap

    def open(self):
        return self

    def close(self):
        pass  # do nothing

    @property
    def handle(self):
        return None


class MMapBackendImplInMem(MMapBackendImpl):
    FILE_CLS = InMemoryFile


def encode_roundtrip_quad(
    encode, bits_per_pixel, input_data=None, dataset_shape=None, tileshape=None,
    start_at_frame=2, stop_before_frame=6
):
    if dataset_shape is None:
        # make some read ranges:
        dataset_shape = (6, 512, 512)
    dataset_shape = Shape(dataset_shape, sig_dims=2)
    if tileshape is None:
        tileshape = (2, 128, 512)
    tiling_scheme = TilingScheme.make_for_shape(
        dataset_shape=dataset_shape,
        tileshape=Shape(tileshape, sig_dims=2),
    )
    sync_offset = 0
    roi = None

    frame_header_bytes = 768

    image_size_bytes = dataset_shape.sig.size * bits_per_pixel // 8

    if bits_per_pixel in (1, 8):
        native_dtype = np.uint8
    elif bits_per_pixel == 16:
        native_dtype = np.uint16

    fields: HeaderDict = {
        'header_size_bytes': frame_header_bytes,
        'dtype': native_dtype,
        'mib_dtype': 'R64',
        'mib_kind': 'r',

        # remove padding from `bits_per_pixel`
        'bits_per_pixel': {1: 1, 8: 6, 16: 12}[bits_per_pixel],
        'image_size': (512, 512),
        'image_size_bytes': image_size_bytes,
        'sequence_first_image': 1,
        'filesize': dataset_shape.nav.size * (image_size_bytes + frame_header_bytes),
        'num_images': dataset_shape.nav.size,
        'num_chips': 4,
        'sensor_layout': (2, 2),
    }

    file = MIBFile(
        path="",
        start_idx=0,
        end_idx=dataset_shape.nav.size,
        native_dtype=native_dtype,
        sig_shape=dataset_shape.sig,
        frame_header=frame_header_bytes,
        file_header=0,
        header=fields,
    )

    fileset = MIBFileSet(files=[file], header=fields, frame_header_bytes=frame_header_bytes)

    backend = MMapBackendImplInMem()

    max_value = (1 << bits_per_pixel) - 1
    if input_data is None:
        data_full = np.random.randint(0, max_value + 1, tuple(dataset_shape.flatten_nav()))
        # make sure min/max values are indeed hit:
        data_full.reshape((-1,))[0] = max_value
        data_full.reshape((-1,))[-1] = 0
        assert np.max(data_full) == max_value
        assert np.min(data_full) == 0
    else:
        data_full = input_data.reshape(dataset_shape.flatten_nav())
    data = data_full[start_at_frame:stop_before_frame]

    # we need headers in-between, in contrast to the frame-by-frame decoding, the decoder
    # expects contiguous input data and we can't slice them away beforehand:
    encoded_data = encode_quad(encode, data_full, bits_per_pixel, with_headers=True)
    decoded = np.zeros_like(data)

    # that's the "interface" we made up for the in-mem mmap file above:
    file.data = encoded_data

    # wrapping the numba decoder function:
    decoder = MIBDecoder(header=fields)

    outer_slice = Slice(
        origin=(start_at_frame, 0, 0),
        shape=dataset_shape.flatten_nav(),
    )

    read_ranges = fileset.get_read_ranges(
        start_at_frame=start_at_frame,
        stop_before_frame=stop_before_frame,
        dtype=native_dtype,
        tiling_scheme=tiling_scheme,
        sync_offset=sync_offset,
        roi=roi,
    )

    for tile in backend.get_tiles(
        tiling_scheme=tiling_scheme,
        fileset=fileset,
        read_ranges=read_ranges,
        roi=roi,
        native_dtype=np.uint8,
        read_dtype=np.float32,
        decoder=decoder,
        sync_offset=0,
        corrections=None,
        array_backend=NUMPY,
    ):
        slice_shifted = tile.tile_slice.shift(outer_slice)
        decoded[slice_shifted.get()] = tile.data.reshape(tile.tile_slice.shape)

    assert_allclose(data, decoded)
    return data, decoded


@pytest.mark.with_numba
@pytest.mark.parametrize(
    'encode,bits_per_pixel', [
        (encode_r1, 1),
        (encode_r6, 8),
        (encode_r12, 16),
    ],
)
@pytest.mark.slow
def test_encode_roundtrip_quad(encode, bits_per_pixel):
    data, decoded = encode_roundtrip_quad(
        encode, bits_per_pixel, dataset_shape=(1, 512, 512), tileshape=(1, 512, 512),
        start_at_frame=0, stop_before_frame=1,
    )
    assert_allclose(data, decoded)


@pytest.mark.slow
@pytest.mark.parametrize(
    'encode,bits_per_pixel', [
        (encode_r1, 1),
        (encode_r6, 8),
        (encode_r12, 16),
    ],
)
def test_encode_roundtrip_quad_slow(encode, bits_per_pixel):
    """
    This version is built to be a bit more thorough by taking in larger amounts
    of data and making sure it is still decoded properly. This is not marked `with_numba`
    as the non-numba code runs really slow.
    """
    data, decoded = encode_roundtrip_quad(encode, bits_per_pixel)
    assert_allclose(data, decoded)


@pytest.mark.parametrize('bits_per_pixel', (1, 8, 16))
def test_readranges_quad(bits_per_pixel):
    # make some read ranges:
    dataset_shape = Shape((4, 4, 512, 512), sig_dims=2)
    tiling_scheme = TilingScheme.make_for_shape(
        dataset_shape=dataset_shape,
        tileshape=Shape((2, 32, 512), sig_dims=2),
    )
    sync_offset = 0
    start_at_frame = 2
    stop_before_frame = 6
    roi = None
    # fileset_arr with one "file":
    fileset_arr = np.zeros((1, 4), dtype=np.int64)

    frame_header_bytes = 768
    frame_footer_bytes = 0

    # This is for formats like DM where the offset in each file can be different.
    # In this case, we only have the per-frame header:
    file_header_bytes = 0

    # (start_idx, end_idx, file_idx, file_header_bytes)
    fileset_arr[0] = (0, dataset_shape.nav.size, 0, file_header_bytes)

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
        bpp=bits_per_pixel,
        frame_header_bytes=frame_header_bytes,
        frame_footer_bytes=frame_footer_bytes,
    )
    read_ranges = mib_2x2_get_read_ranges(**kwargs)

    # a 3-tuple is returned from get_read_ranges functions
    assert len(read_ranges) == 3

    rr_slices, rr_ranges, rr_scheme_indices = read_ranges

    # - we read four frames, [2, 6)
    # - each frame is divided into 512/32 = 16 tiles in sig dimensions
    # - each tile has depth=2, so we have 4/2*16 = 32 tiles in total
    assert rr_slices.shape == (
        32,  # number of tiles
        2,   # origin and shape
        3,   # indices in flattened dimensions
    )
    # first and last tile slices:
    assert tuple(rr_slices[0, 0]) == (2, 0, 0)
    assert tuple(rr_slices[0, 1]) == (2, 32, 512)
    assert tuple(rr_slices[-1, 0]) == (4, 480, 0)
    assert tuple(rr_slices[-1, 1]) == (2, 32, 512)

    assert rr_ranges.shape == (
        32,   # 32 tiles
        128,  # 128 reads per file per tile (ugh...)
        4     # (file_idx, start, stop, flip)
    )
    # each rr corresponds to a single row:
    row_size = 256 * bits_per_pixel // 8  # row size for a single quadrant, in bytes
    assert set((rr_ranges[:, :, 2] - rr_ranges[:, :, 1]).reshape((-1,))) == {row_size}

    # the offset for the first frame we are reading.
    # we have 3 headers and two full frames ahead of us:
    sig_size_bytes = 512 * 512 * bits_per_pixel // 8
    frame_2_start = 3 * frame_header_bytes + 2 * sig_size_bytes
    q1_offset = 3 * row_size
    q2_offset = 2 * row_size
    q3_offset = 1 * row_size
    q4_offset = 0 * row_size
    stride = 4 * row_size

    # input layout is: [4 | 3 | 2 | 1]
    #
    # output layout is:
    # _________
    # | 1 | 2 |
    # ---------
    # | 3 | 4 |
    # ---------

    # first tile reads only from Q1/Q2:
    # 0, 0 is rr for the first row of Q1
    assert tuple(rr_ranges[0, 0]) == (
        0,  # file index
        frame_2_start + q1_offset + 0 * stride,
        frame_2_start + q1_offset + 0 * stride + row_size,
        0,   # don't flip
    )
    # 0, 1 is rr for the first row of Q2
    assert tuple(rr_ranges[0, 1]) == (
        0,  # file index
        frame_2_start + q2_offset + 0 * stride,
        frame_2_start + q2_offset + 0 * stride + row_size,
        0,   # don't flip
    )
    # 0, 2 is rr for the second row of Q1
    assert tuple(rr_ranges[0, 2]) == (
        0,  # file index
        frame_2_start + q1_offset + 1 * stride,
        frame_2_start + q1_offset + 1 * stride + row_size,
        0,   # don't flip
    )
    # 0, 3 is rr for the second row of Q2
    assert tuple(rr_ranges[0, 3]) == (
        0,  # file index
        frame_2_start + q2_offset + 1 * stride,
        frame_2_start + q2_offset + 1 * stride + row_size,
        0,   # don't flip
    )

    # flip is set in half of the rr's
    assert set(rr_ranges[:8, :, -1].reshape((-1,))) == {0}
    assert set(rr_ranges[8:16, :, -1].reshape((-1,))) == {1}
    # and so on...

    # Let's check out another tile that actually reads from the flipped Q3/Q4
    # (in this case, let's have a look at the last row of the *output*, but still
    # in the first "tile block", so depth starting at frame 2):
    last_tile_idx = 15
    assert tuple(rr_slices[last_tile_idx, 0]) == (2, 480, 0)
    assert tuple(rr_slices[last_tile_idx, 1]) == (2, 32, 512)

    # we have in total a tile depth of 2, which translates to 128
    # read ranges (depth 2, 32 rows per tile, two input rows per output row)
    assert rr_ranges[last_tile_idx].shape == (128, 4)

    # so this means the first input row of Q3 appears at the end of the
    # first half of the read ranges for the full tile.

    # 15, 62 is rr for the first *input* row of Q3, in depth=0 of the tile
    assert tuple(rr_ranges[last_tile_idx, 62]) == (
        0,  # file index
        frame_2_start + q3_offset + 0 * stride,
        frame_2_start + q3_offset + 0 * stride + row_size,
        1,   # flip
    )
    # 15, 63 is rr for the first *input* row of Q4, in depth=0 of the tile
    assert tuple(rr_ranges[last_tile_idx, 63]) == (
        0,  # file index
        frame_2_start + q4_offset + 0 * stride,
        frame_2_start + q4_offset + 0 * stride + row_size,
        1,   # flip
    )

    # for each tile, this is an index into the tiling scheme and indirectly
    # gives us the sig part of the tile slice:
    assert rr_scheme_indices.shape == (32,)
    assert set(rr_scheme_indices) == set(range(16))
