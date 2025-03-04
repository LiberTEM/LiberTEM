import os
import re
import glob
import math
import typing
import logging
import itertools

import numpy as np
import numba
from numba.typed import List
from ncempy.io import dm

from libertem.common.math import prod, make_2D_square, flat_nonzero
from libertem.common.buffers import zeros_aligned
from libertem.common import Shape
from libertem.common.messageconverter import MessageConverter
from .base import (
    DataSet, BasePartition, DataSetException, DataSetMeta,
    FileSet, File, Decoder, make_get_read_ranges,
    TilingScheme, IOBackend,
)
from .base.file import OffsetsSizes

log = logging.getLogger(__name__)


HEADER_SIZE = 40
BLOCK_SIZE = 0x5758
DATA_SIZE = BLOCK_SIZE - HEADER_SIZE
BLOCK_SHAPE = (930, 16)
BLOCKS_PER_SECTOR_PER_FRAME = 32
FIRST_FRAME_SCAN_SIZE = 400
NUM_SECTORS = 8
SECTOR_SIZE = (2 * 930, 256)

SHUTTER_ACTIVE_MASK = 0x1


class K2ISDatasetParams(MessageConverter):
    SCHEMA = {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "$id": "http://libertem.org/K2ISDatasetParams.schema.json",
      "title": "K2ISDatasetParams",
      "type": "object",
      "properties": {
          "type": {"const": "K2IS"},
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
        if "nav_shape" in raw_data:
            data["nav_shape"] = tuple(raw_data["nav_shape"])
        if "sig_shape" in raw_data:
            data["sig_shape"] = tuple(raw_data["sig_shape"])
        if "sync_offset" in raw_data:
            data["sync_offset"] = raw_data["sync_offset"]
        return data


@numba.njit(inline='always')
def decode_uint12_le(inp, out):
    """
    Decode bytes from bytestring ``inp`` as 12 bit into ``out``

    Based partially on https://stackoverflow.com/a/45070947/540644
    """
    # assert np.mod(len(inp), 3) == 0
    # assert len(out) >= len(inp) * 2 / 3

    for i in range(len(inp) // 3):
        fst_uint8 = np.uint16(inp[i * 3])
        mid_uint8 = np.uint16(inp[i * 3 + 1])
        lst_uint8 = np.uint16(inp[i * 3 + 2])

        a = fst_uint8 | (mid_uint8 & 0x0F) << 8
        b = (mid_uint8 & 0xF0) >> 4 | lst_uint8 << 4
        out[i * 2] = a
        out[i * 2 + 1] = b


# @numba.njit(inline='always', boundscheck=True)
@numba.njit(inline='always')
def decode_k2is(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    """
    Decode a single block, from a single read range, into a tile that may
    contain multiple blocks in the signal dimensions. This function is called
    multiple times for a single tile, for all read ranges that are part of this
    tile.
    """
    # blocks per tile (only in signal dimensions)
    blocks_per_tile = out.shape[1] // (BLOCK_SHAPE[0] * BLOCK_SHAPE[1])

    n_blocks_y, n_blocks_x, block_y_i, block_x_i = rr[3:]

    tile_idx = idx // blocks_per_tile

    # we take three bytes of the input to decode two numbers
    # -> as the blocks are 16 numbers wide, the two numbers
    # will always be part of the same row, so the y-stride is the same

    # the offset between two rows in the output (in indices, not bytes)
    stride_y = shape[2]

    # shortcut, in case we don't need to handle multiple
    # blocks in x direction:
    if stride_y == 16:
        block_out = out[tile_idx, 16 * 930 * block_y_i:16 * 930 * (block_y_i + 1)]
        return decode_uint12_le(inp=inp, out=block_out)

    # starting offset of the current block:
    # 930 * block_y_i:930 * (block_y_i + 1),
    # 16 * block_x_i:16 * (block_x_i + 1),
    block_offset_y = block_y_i * BLOCK_SHAPE[0] * n_blocks_x * BLOCK_SHAPE[1]
    block_offset = block_offset_y + block_x_i * BLOCK_SHAPE[1]

    out_z = out[tile_idx]

    # decode_uint12_le(inp=inp, out=blockbuf) inlined here:
    # inp is uint8, so the outer loop needs to jump 24 bytes each time.
    for i in range(len(inp) // 3 // 8):
        # i is the output row index of a single block,
        # so the beginning of the row in output coordinates:
        out_pos = block_offset + i * stride_y

        in_row_offset = i * 3 * 8

        # loop for a single row:
        # for each j, we process bytes of input into two output numbers
        # -> we consume 8*3 = 24 bytes and generate 8*2=16 numbers
        for j in range(8):
            triplet_offset = in_row_offset + j * 3
            fst_uint8 = np.uint16(inp[triplet_offset])
            mid_uint8 = np.uint16(inp[triplet_offset + 1])
            lst_uint8 = np.uint16(inp[triplet_offset + 2])

            a = fst_uint8 | (mid_uint8 & 0x0F) << 8
            b = (mid_uint8 & 0xF0) >> 4 | lst_uint8 << 4

            out_z[out_pos] = a
            out_z[out_pos + 1] = b

            out_pos += 2


class K2ISDecoder(Decoder):
    def get_decode(self, native_dtype, read_dtype):
        return decode_k2is


@numba.njit(inline='always')
def _k2is_read_ranges_tile_block(
    slices_arr, fileset_arr, slice_sig_sizes, sig_origins,
    inner_indices_start, inner_indices_stop, frame_indices, sig_size,
    px_to_bytes, bpp, frame_header_bytes, frame_footer_bytes, file_idxs,
    slice_offset, extra, sig_shape,
):
    result = List()

    # positions in the signal dimensions:
    for slice_idx in range(slices_arr.shape[0]):
        # (offset, size) arrays defining what data to read (in pixels)
        slice_origin = slices_arr[slice_idx][0]
        slice_shape = slices_arr[slice_idx][1]

        read_ranges = List()

        n_blocks_y = slice_shape[0] // 930
        n_blocks_x = slice_shape[1] // 16

        origin_block_y = slice_origin[0] // 930
        origin_block_x = slice_origin[1] // 16

        # inner "depth" loop along the (flat) navigation axis of a tile:
        for i, inner_frame_idx in enumerate(range(inner_indices_start, inner_indices_stop)):
            inner_frame = frame_indices[inner_frame_idx]
            frame_in_file_idx = inner_frame  # in k2is all files contain data from all frames

            for block_y_i in range(n_blocks_y):
                sector_index_y = origin_block_y + block_y_i
                for block_x_i in range(n_blocks_x):
                    block_index_x = origin_block_x + block_x_i
                    sector_id = block_index_x // 16
                    sector_index_x = block_index_x % 16
                    f = fileset_arr[sector_id]
                    file_header_bytes = f[3]

                    # "linear" block index per sector:
                    blockidx = (15 - sector_index_x) + sector_index_y * 16
                    offset = (
                        file_header_bytes
                        + frame_in_file_idx * BLOCK_SIZE * BLOCKS_PER_SECTOR_PER_FRAME
                        + blockidx * BLOCK_SIZE
                    )
                    start = offset + HEADER_SIZE
                    stop = offset + HEADER_SIZE + DATA_SIZE
                    read_ranges.append(
                        (sector_id, start, stop,
                         n_blocks_y, n_blocks_x,
                         block_y_i, block_x_i)
                    )

        # the indices are compressed to the selected frames
        compressed_slice = np.array([
            [slice_offset + inner_indices_start] + [i for i in slice_origin],
            [inner_indices_stop - inner_indices_start] + [i for i in slice_shape],
        ])

        result.append((slice_idx, compressed_slice, read_ranges))
    return result


k2is_get_read_ranges = make_get_read_ranges(
    read_ranges_tile_block=_k2is_read_ranges_tile_block
)


def _pattern(path):
    path, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".gtg":
        pattern = "%s*.bin" % glob.escape(path)
    elif ext == ".bin":
        pattern = "%s*.bin" % (
            glob.escape(re.sub(r'[0-9]+$', '', path))
        )
    else:
        raise DataSetException("unknown extension: %s" % ext)
    return pattern


def get_filenames(path, disable_glob=False):
    if disable_glob:
        return [path]
    else:
        return glob.glob(_pattern(path))


def _get_gtg_path(full_path):
    path, ext = os.path.splitext(full_path)
    ext = ext.lower()
    if ext == ".gtg":
        return full_path
    elif ext == ".bin":
        return "%s.gtg" % (
            re.sub(r'[0-9]+$', '', path)
        )


def _get_nav_shape(path):
    with dm.fileDM(_get_gtg_path(path), on_memory=True) as dm_file:
        nav_shape_y = dm_file.allTags.get('.SI Dimensions.Size Y')
        nav_shape_x = dm_file.allTags.get('.SI Dimensions.Size X')
        if nav_shape_y is not None and nav_shape_x is not None:
            return (int(nav_shape_y), int(nav_shape_x))
        return None


def _nav_shape_without_flyback(path):
    with dm.fileDM(_get_gtg_path(path), on_memory=True) as dm_file:
        nav_shape_y = dm_file.allTags.get(
            '.SI Image Tags.SI.Acquisition.Spatial Sampling.Height (pixels)'
        )
        nav_shape_x = dm_file.allTags.get(
            '.SI Image Tags.SI.Acquisition.Spatial Sampling.Width (pixels)'
        )
        if nav_shape_y is not None and nav_shape_x is not None:
            return (int(nav_shape_y), int(nav_shape_x))
        return None


def _get_num_frames_w_shutter_active_flag_set(syncer):
    syncer.sync()
    with syncer.sectors[0] as s:
        num_frames = (
            s.last_block_offset - s.first_block_offset + BLOCK_SIZE
        ) // BLOCK_SIZE // BLOCKS_PER_SECTOR_PER_FRAME
    return num_frames


def _get_num_frames(syncer):
    syncer.sync_sectors()
    with syncer.sectors[0] as s:
        num_frames = (
            s.last_block_offset - s.first_block_offset + BLOCK_SIZE
        ) // BLOCK_SIZE // BLOCKS_PER_SECTOR_PER_FRAME
    return num_frames


def _get_syncer_for_detect_params(files):
    return K2Syncer(files)


class K2Syncer:
    """
    Sync the 8 sectors of a K2IS data set. First, find the first complete frame and the
    last complete frame. Next, sync to the first frame with the shutter_active flag set.
    Finally, validate the first and last frames.

    Parameters
    ----------
    paths: list of str
        List of paths of the 8 .bin files

    start_offsets: list of int, optional
        List of first block offsets of the 8 sectors

    last_offsets: list of int, optional
        List of last block offsets of the 8 sectors
    """

    def __init__(self, paths, start_offsets=None, last_offsets=None):
        self.paths = paths
        if start_offsets is None:
            start_offsets = NUM_SECTORS * [0]
            last_offsets = NUM_SECTORS * [None]
        self.sectors = [
            Sector(fname, idx, initial_offset=start_offset, end_offset=last_offset)
            for (
                (idx, fname), start_offset, last_offset
            ) in zip(enumerate(paths), start_offsets, last_offsets)
        ]

    def sync_sectors(self):
        for b in self.first_blocks():
            assert b.is_valid, "first block is not valid!"
        # sync up all sectors to start with the same `block_count`
        block_with_max_idx = sorted(self.first_blocks(), key=lambda b: b.header['block_count'])[-1]
        start_blocks = [
            s.first_block_with(
                lambda b: b.header['block_count'] == block_with_max_idx.header['block_count']
            )
            for s in self.sectors
        ]
        for b in start_blocks:
            assert b.is_valid
            b.sector.set_first_block_offset(b.offset)
        log.debug("first_block_offsets #1: %r", [s.first_block_offset for s in self.sectors])
        # skip incomplete frames:
        # if the next 32 blocks of a sector don't have all the same frame id,
        # the frame is incomplete
        have_overlap = (
            len({
                b.header['frame_id']
                for b in itertools.islice(s.get_blocks(), BLOCKS_PER_SECTOR_PER_FRAME)
            }) > 1
            for s in self.sectors
        )
        if any(have_overlap):
            log.debug("have_overlap, finding next frame")
            frame_id = self.first_blocks()[0].header['frame_id']
            for s in self.sectors:
                offset = s.first_block_with(lambda b: b.header['frame_id'] != frame_id).offset
                s.set_first_block_offset(offset)
        log.debug("first_block_offsets #2: %r", [s.first_block_offset for s in self.sectors])
        for b in self.first_blocks():
            assert b.is_valid
        # find last_block_offsets
        for s in self.sectors:
            offset = s.first_block_offset
            while offset + BLOCK_SIZE <= s.last_block_offset:
                offset += BLOCK_SIZE
            s.set_last_block_offset(offset)
        log.debug("last_block_offsets #1: %r", [s.last_block_offset for s in self.sectors])
        for b in self.last_blocks():
            assert b.is_valid, "last block is not valid!"
        # end all sectors with the same `block_count`
        block_with_min_idx = sorted(self.last_blocks(), key=lambda b: b.header['block_count'])[0]
        end_blocks = [
            s.first_block_from_end_with(
                lambda b: b.header['block_count'] == block_with_min_idx.header['block_count']
            )
            for s in self.sectors
        ]
        for b in end_blocks:
            assert b.is_valid
            b.sector.set_last_block_offset(b.offset)
        log.debug("last_block_offsets #2: %r", [s.last_block_offset for s in self.sectors])
        # skip incomplete frames:
        # if the next 32 blocks from the end of a sector don't have all the same frame id,
        # the frame is incomplete
        have_overlap_at_end = (
            len({
                b.header['frame_id']
                for b in itertools.islice(s.get_blocks_from_end(), BLOCKS_PER_SECTOR_PER_FRAME)
            }) > 1
            for s in self.sectors
        )
        if any(have_overlap_at_end):
            log.debug("have_overlap, finding previous frame")
            frame_id = self.last_blocks()[0].header['frame_id']
            for s in self.sectors:
                offset = s.first_block_from_end_with(
                    lambda b: b.header['frame_id'] != frame_id
                ).offset
                s.set_last_block_offset(offset)
        log.debug("last_block_offsets #3: %r", [s.last_block_offset for s in self.sectors])
        for b in self.last_blocks():
            assert b.is_valid

    def sync_to_first_frame(self):
        log.debug("synchronizing to shutter_active flag...")
        for s in self.sectors:
            offset = s.first_block_with_search(lambda b: b.shutter_active).offset
            s.set_first_block_offset(offset)
        log.debug("first_block_offsets #3: %r", [s.first_block_offset for s in self.sectors])

    def validate_sync(self):
        # first blocks should be valid:
        first_blocks = self.first_blocks()
        frame_id = first_blocks[0].header['frame_id']
        for b in first_blocks:
            assert b.is_valid
            assert b.header['frame_id'] == frame_id

        # in each sector, a whole frame should follow, and frame ids should match:
        for s in self.sectors:
            blocks = itertools.islice(s.get_blocks(), BLOCKS_PER_SECTOR_PER_FRAME)
            assert all(b.header['frame_id'] == frame_id
                       for b in blocks)

        # last blocks should be valid:
        last_blocks = self.last_blocks()
        frame_id = last_blocks[0].header['frame_id']
        for b in last_blocks:
            assert b.is_valid
            assert b.header['frame_id'] == frame_id

        # each sector should end with a whole frame, and frame ids should match:
        for s in self.sectors:
            blocks = itertools.islice(s.get_blocks_from_end(), BLOCKS_PER_SECTOR_PER_FRAME)
            assert all(b.header['frame_id'] == frame_id
                       for b in blocks)

    def sync(self):
        self.sync_sectors()
        self.sync_to_first_frame()
        self.validate_sync()

    def first_blocks(self):
        return [next(s.get_blocks()) for s in self.sectors]

    def last_blocks(self):
        return [next(s.get_blocks_from_end()) for s in self.sectors]

    def close(self):
        for s in self.sectors:
            s.close()


class Sector:
    def __init__(self, fname, idx, initial_offset=0, end_offset=None):
        self.fname = fname
        self.idx = idx
        self.filesize = os.stat(fname).st_size
        self.first_block_offset = initial_offset
        if end_offset is None:
            self.last_block_offset = self.filesize + self.first_block_offset - BLOCK_SIZE
        else:
            self.last_block_offset = end_offset
        # FIXME: hardcoded sig_dims
        self.sig_dims = 2

    def open(self):
        self.f = open(self.fname, "rb")

    def close(self):
        self.f.close()
        self.f = None

    def seek(self, pos):
        self.f.seek(pos)

    def read(self, size):
        return self.f.read(size)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def get_block_by_index(self, idx):
        offset = self.first_block_offset + idx * BLOCK_SIZE
        return DataBlock(offset=offset, sector=self)

    def get_blocks(self):
        offset = self.first_block_offset
        while offset <= self.last_block_offset:
            yield DataBlock(offset=offset, sector=self)
            offset += BLOCK_SIZE

    def get_blocks_from_end(self):
        offset = self.last_block_offset
        while offset >= self.first_block_offset:
            yield DataBlock(offset=offset, sector=self)
            offset -= BLOCK_SIZE

    def get_block_by_offset(self, offset):
        return DataBlock(offset=offset, sector=self)

    def get_blocks_from_offset(self, offset):
        while offset <= self.last_block_offset:
            yield DataBlock(offset=offset, sector=self)
            offset += BLOCK_SIZE

    def set_first_block_offset(self, offset):
        self.first_block_offset = offset

    def set_last_block_offset(self, offset):
        self.last_block_offset = offset

    def first_block(self):
        return next(self.get_blocks())

    def last_block(self):
        return next(self.get_blocks_from_end())

    def first_block_with(self, predicate=lambda b: True):
        return next(b for b in self.get_blocks()
                    if predicate(b))

    def first_block_from_end_with(self, predicate=lambda b: True):
        return next(b for b in self.get_blocks_from_end()
                    if predicate(b))

    def first_block_with_search(self, predicate=lambda b: True, step=32 * 8 * 50):
        """
        Binary search variant of `first_block_with`, assuming that predicate is true
        from some index on, and stays true (at least for 10 * step).
        """
        # find a rough upper bound:
        upper = None
        for idx in range(0, step * 10, step):
            block = self.get_block_by_index(idx)
            if predicate(block):
                upper = idx
                break

        # the block is somewhere in [0, upper]
        def _rec(current_lower, current_upper):
            if current_upper == current_lower:
                # we found our match, return block
                assert predicate(self.get_block_by_index(current_upper))
                return self.get_block_by_index(current_upper)
            mid = math.floor((current_lower + current_upper) / 2)
            if predicate(self.get_block_by_index(mid)):
                # mid block has predicate=True, so result must be mid or lower:
                return _rec(current_lower, mid)
            else:
                # mid block has predicate=False, so result must be mid + 1 or higher:
                return _rec(mid + 1, current_upper)

        return _rec(0, upper)

    def __repr__(self):
        return "<Sector %d>" % self.idx


class DataBlock:
    header_dtype = [
        ('sync', '>u4'),
        ('padding1', (bytes, 4)),
        ('version', '>u1'),
        ('flags', '>u1'),
        ('padding2', (bytes, 6)),
        ('block_count', '>u4'),
        ('width', '>u2'),
        ('height', '>u2'),
        ('frame_id', '>u4'),
        ('pixel_x_start', '>u2'),  # first pixel x coordinate within sector
        ('pixel_y_start', '>u2'),  # first pixel y coordinate within sector
        ('pixel_x_end', '>u2'),  # last pixel x coordinate within sector
        ('pixel_y_end', '>u2'),  # last pixel y coordinate within sector
        ('block_size', '>u4'),  # should be fixed 0x5758
    ]

    def __init__(self, offset, sector):
        self.offset = offset
        self.sector = sector
        self._header_raw = None
        self._header = None
        self._data_raw = None

    @property
    def is_valid(self):
        return (
            self.sector.filesize >= self.offset + BLOCK_SIZE
            and self.header['width'] == 256
            and self.header['height'] == 1860
            and self.header['sync'] == 0xFFFF0055
        )

    @property
    def header(self):
        if self._header_raw is None:
            with self.sector:
                self.sector.seek(self.offset)
                self._header_raw = np.fromfile(self.sector.f, dtype=self.header_dtype, count=1)
        if self._header is not None:
            return self._header
        header = {}
        for field, dtype in self.header_dtype:
            if type(dtype) is not str:
                continue
            header[field] = self._header_raw[field][0]
        self._header = header
        return header

    @property
    def shutter_active(self):
        return self.header['flags'] & SHUTTER_ACTIVE_MASK == 1

    @property
    def pixel_data_raw(self):
        if self._data_raw is None:
            with self.sector:
                self.sector.seek(self.offset + HEADER_SIZE)
                self._data_raw = self.sector.read(BLOCK_SIZE - HEADER_SIZE)
        return self._data_raw

    def readinto(self, out):
        out = out.reshape(930 * 16)
        decode_uint12_le(inp=self.pixel_data_raw, out=out)
        return out.reshape(930, 16)

    @property
    def pixel_data(self):
        if not self.is_valid:
            raise ValueError("invalid block: %r" % self)
        arr = zeros_aligned((930 * 16), dtype="uint16")
        self.readinto(arr)
        return arr.reshape(930, 16)

    def copy_to_frame(self, frame):
        sector_width = 256
        x_offset = self.sector.idx * sector_width
        h = self.header
        self.readinto(frame[
            h['pixel_y_start']:h['pixel_y_end'] + 1,
            h['pixel_x_start'] + x_offset:h['pixel_x_end'] + 1 + x_offset,
        ])

    def __repr__(self):
        h = self.header
        return "<DataBlock for frame=%d x=%d:%d y=%d:%d @ %d>" % (
            h['frame_id'],
            h['pixel_x_start'], h['pixel_x_end'],
            h['pixel_y_start'], h['pixel_y_end'],
            self.offset,
        )


class K2FileSet(FileSet):
    def get_for_range(self, start, stop):
        """
        K2 files have a different mapping than other file formats, so we need
        to override this to not filter in navigation axis
        """
        return self

    def get_read_ranges(
        self, start_at_frame: int, stop_before_frame: int,
        dtype, tiling_scheme: TilingScheme, sync_offset: int = 0,
        roi: typing.Union[np.ndarray, None] = None,
    ):
        fileset_arr = self.get_as_arr()
        roi_nonzero = None
        if roi is not None:
            roi_nonzero = flat_nonzero(roi)
        kwargs = dict(
            start_at_frame=start_at_frame,
            stop_before_frame=stop_before_frame,
            roi_nonzero=roi_nonzero,
            depth=tiling_scheme.depth,
            slices_arr=tiling_scheme.slices_array,
            fileset_arr=fileset_arr,
            sig_shape=tuple(tiling_scheme.dataset_shape.sig),
            sync_offset=sync_offset,
            bpp=np.dtype(dtype).itemsize * 8,
            frame_header_bytes=self._frame_header_bytes,
            frame_footer_bytes=self._frame_footer_bytes,
        )
        return k2is_get_read_ranges(**kwargs)


class K2ISFile(File):
    def get_offsets_sizes(self, size: int) -> OffsetsSizes:
        """
        The simple "offset/size/header/footer" method doesn't
        apply for the K2IS format, so we stub it out here:
        """
        return OffsetsSizes(
            file_offset=self._file_header,
            skip_end=0,
            frame_offset=0,
            frame_size=0,
        )

    def get_array_from_memview(self, mem: memoryview, slicing: OffsetsSizes) -> np.ndarray:
        mem = mem[slicing.file_offset:]
        return np.frombuffer(mem, dtype=self._native_dtype)


class K2ISDataSet(DataSet):
    """
    Read raw K2IS data sets. They consist of 8 .bin files and one .gtg file.
    Currently, data acquired using the STEMx unit is supported, metadata
    about the nav_shape is read from the .gtg file.

    Parameters
    ----------
    path: str
        Path to one of the files of the data set (either one of the .bin files or the .gtg file)

    nav_shape: tuple of int, optional
        A n-tuple that specifies the size of the navigation region ((y, x), but
        can also be of length 1 for example for a line scan, or length 3 for
        a data cube, for example)

    sig_shape: tuple of int, optional
        Signal/detector size (height, width)

    sync_offset: int, optional
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start

    num_partitions: int, optional
        Override the number of partitions. This is useful if the
        default number of partitions, chosen based on common workloads,
        creates partitions which are too large (or small) for the UDFs
        being run on this dataset.

    Examples
    --------

    >>> ds = ctx.load("k2is", path='./path_to_file.bin', ...)  # doctest: +SKIP
    """

    def __init__(
        self,
        path,
        nav_shape=None,
        sig_shape=None,
        sync_offset=None,
        io_backend=None,
        num_partitions=None,
    ):
        super().__init__(
            io_backend=io_backend,
            num_partitions=num_partitions,
        )
        self._path = path
        self._start_offsets = None
        self._last_offsets = None
        self._files = None
        self._nav_shape = tuple(nav_shape) if nav_shape else nav_shape
        self._sig_shape = tuple(sig_shape) if sig_shape else sig_shape
        self._is_time_series = None
        self._sync_offset = None
        self._native_sync_offset = 0
        self._user_sync_offset = sync_offset
        self._cached_user_sync_offset = None

    def _do_initialize(self):
        self._files = self._get_files()
        self._set_skip_frames_and_nav_shape()
        if self._sig_shape is None:
            self._sig_shape = (SECTOR_SIZE[0], NUM_SECTORS * SECTOR_SIZE[1])
        elif int(prod(self._sig_shape)) != int(prod(
                    (SECTOR_SIZE[0], NUM_SECTORS * SECTOR_SIZE[1])
                )):
            raise DataSetException(
                "sig_shape must be of size: %s" % int(prod(
                    (SECTOR_SIZE[0], NUM_SECTORS * SECTOR_SIZE[1])
                ))
            )
        self._image_count = _get_num_frames(self._get_syncer(do_sync=False))
        self._set_sync_offset()
        self._get_syncer(do_sync=True)
        self._meta = DataSetMeta(
            shape=Shape(self._nav_shape + self._sig_shape, sig_dims=len(self._sig_shape)),
            raw_dtype=np.dtype("uint16"),
            sync_offset=self._sync_offset,
            image_count=self._image_count,
        )
        return self

    def initialize(self, executor):
        return executor.run_function(self._do_initialize)

    def _set_skip_frames_and_nav_shape(self):
        nav_shape = _get_nav_shape(self._path)
        if nav_shape is not None:
            # the sync flag appears to be set one frame too late, so
            # we compensate here by setting a negative _skip_frames value.
            # skip_frames is applied after synchronization.
            self._is_time_series = False
            self._skip_frames = -1
            if self._nav_shape is None:
                self._nav_shape = nav_shape
        else:
            # data set is time series and does not have any frames with
            # SHUTTER_ACTIVE_MASK not set. Hence skip_frames is not needed.
            self._is_time_series = True
            self._skip_frames = 0
            if self._nav_shape is None:
                self._nav_shape = (
                    _get_num_frames_w_shutter_active_flag_set(self._get_syncer(do_sync=False)),
                )

    def _set_sync_offset(self):
        self._num_frames_w_shutter_active_flag_set = _get_num_frames_w_shutter_active_flag_set(
            self._get_syncer(do_sync=False)
        )
        self._native_sync_offset = self._image_count - self._num_frames_w_shutter_active_flag_set
        if self._user_sync_offset is None:
            self._user_sync_offset = self._native_sync_offset
        self._nav_shape_product = int(prod(self._nav_shape))
        self._sync_offset_info = self._get_sync_offset_info()
        if not self._is_time_series:
            if self._user_sync_offset == self._native_sync_offset:
                self._sync_offset = 0
            elif self._user_sync_offset > self._native_sync_offset:
                self._sync_offset = self._user_sync_offset - self._native_sync_offset
            else:
                if self._user_sync_offset > 0:
                    self._skip_frames = self._user_sync_offset - self._native_sync_offset - 1
                    self._sync_offset = 0
                else:
                    self._skip_frames = -1 * self._native_sync_offset
                    self._sync_offset = self._user_sync_offset - 1
        else:
            self._sync_offset = self._user_sync_offset

    def _get_sync_offset_info(self):
        """
        Check sync_offset specified and returns number of frames skipped and inserted
        """
        if not -1*self._image_count < self._user_sync_offset < self._image_count:
            raise DataSetException(
                "sync_offset should be in (%s, %s), which is (-image_count, image_count)"
                % (-1*self._image_count, self._image_count)
            )
        return {
            "frames_skipped_start": max(0, self._user_sync_offset),
            "frames_ignored_end": max(
                0, self._image_count - self._nav_shape_product - self._user_sync_offset
            ),
            "frames_inserted_start": abs(min(0, self._user_sync_offset)),
            "frames_inserted_end": max(
                0, self._nav_shape_product - self._image_count + self._user_sync_offset
            )
        }

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    @classmethod
    def get_msg_converter(cls):
        return K2ISDatasetParams

    @classmethod
    def get_supported_extensions(cls):
        return {"gtg", "bin"}

    @classmethod
    def detect_params(cls, path, executor):
        """
        detect if path points to a file that is part of a k2is dataset.
        extension needs to be bin or gtg, and number of files of the dataset needs to match
        the number of sectors. no further checking is done yet. in certain cases there may be false
        positives, for example if users name their binary files .bin and have 8 of them
        in a directory.
        """
        try:
            files = executor.run_function(get_filenames, path)
            if len(files) != NUM_SECTORS:
                return False
        except DataSetException:
            return False
        s = executor.run_function(_get_syncer_for_detect_params, files)
        num_frames = executor.run_function(_get_num_frames, s)
        num_frames_w_shutter_active_flag_set = executor.run_function(
            _get_num_frames_w_shutter_active_flag_set, s
        )
        sync_offset = num_frames - num_frames_w_shutter_active_flag_set
        nav_shape = executor.run_function(_get_nav_shape, path)
        if nav_shape is None:
            nav_shape = make_2D_square((num_frames_w_shutter_active_flag_set,))
        return {
            "parameters": {
                "path": path,
                "nav_shape": nav_shape,
                "sig_shape": (SECTOR_SIZE[0], NUM_SECTORS * SECTOR_SIZE[1]),
                "sync_offset": sync_offset,
            },
            "info": {
                "image_count": num_frames,
                "native_sig_shape": (SECTOR_SIZE[0], NUM_SECTORS * SECTOR_SIZE[1]),
            }
        }

    def check_valid(self):
        try:
            syncer = self._get_syncer()
            syncer.sync()
        except Exception as e:
            raise DataSetException("failed to load dataset: %s" % e) from e
        return True

    def get_cache_key(self):
        gtg = _get_gtg_path(self._path)
        return {
            "gtg_path": gtg,
            "shape": tuple(self.shape),
            "sync_offset": self._sync_offset,
        }

    def get_diagnostics(self):
        with self._get_syncer().sectors[0] as sector:
            first_block = sector.first_block()
            last_block = sector.last_block()
        fs_nosync = self._get_syncer(do_sync=False)
        sector_nosync = fs_nosync.sectors[0]
        first_block_nosync = next(sector_nosync.get_blocks())
        last_block_nosync = next(sector_nosync.get_blocks_from_end())

        return [
            {"name": "first block offsets for all sectors",
             "value": ", ".join([str(s) for s in self._start_offsets])},

            {"name": "last block offsets for all sectors",
             "value": ", ".join([str(s) for s in self._last_offsets])},

            {"name": "number of frames before sync (from first sector)",
             "value": str(self._image_count)},

            {"name": "number of frames after sync (from first sector)",
             "value": str(self._num_frames_w_shutter_active_flag_set)},

            {"name": "first frame id before sync (from first sector)",
             "value": str(first_block_nosync.header['frame_id'])},

            {"name": "first frame id after sync (from first sector)",
             "value": str(first_block.header['frame_id'])},

            {"name": "last frame id before sync (from first sector)",
             "value": str(last_block_nosync.header['frame_id'])},

            {"name": "last frame id after sync (from first sector)",
             "value": str(last_block.header['frame_id'])},
        ]

    def _get_files(self):
        files = get_filenames(self._path)
        if len(files) != NUM_SECTORS:
            raise DataSetException("expected %d files at %s, found %d" % (
                NUM_SECTORS,
                _pattern(self._path),
                len(files)
            ))
        return list(sorted(files))

    def _cache_first_block_offsets(self, fs):
        self._start_offsets = [
            s.first_block_offset
            for s in fs.sectors
        ]
        # apply skip_frames value to the start_offsets
        self._start_offsets = [o + BLOCK_SIZE * self._skip_frames * BLOCKS_PER_SECTOR_PER_FRAME
                               for o in self._start_offsets]

    def _cache_last_block_offsets(self, fs):
        self._last_offsets = [
            s.last_block_offset
            for s in fs.sectors
        ]

    def _cache_user_sync_offset(self, user_sync_offset):
        self._cached_user_sync_offset = user_sync_offset

    def _get_syncer(self, do_sync=True):
        if not do_sync:
            return K2Syncer(self._files)
        if self._start_offsets is None or self._user_sync_offset != self._cached_user_sync_offset:
            sy = K2Syncer(self._files)
            sy.sync()
            self._cache_first_block_offsets(sy)
            self._cache_last_block_offsets(sy)
            self._cache_user_sync_offset(self._user_sync_offset)
        else:
            sy = K2Syncer(
                self._files, start_offsets=self._start_offsets, last_offsets=self._last_offsets
            )
        return sy

    def get_decoder(self) -> Decoder:
        return K2ISDecoder()

    def get_base_shape(self, roi):
        return (1, 930, 16)

    def _get_fileset(self):
        files = [
            K2ISFile(
                path=path,
                start_idx=0,
                end_idx=self._image_count,
                native_dtype=np.uint8,
                sig_shape=self._sig_shape,
                file_header=offset,
            )
            for path, offset in zip(self._files, self._start_offsets)
        ]
        return K2FileSet(files=files)

    def get_partitions(self):
        io_backend = self.get_io_backend()
        fileset = self._get_fileset()
        for part_slice, start, stop in self.get_slices():
            yield K2ISPartition(
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
                io_backend=io_backend,
                decoder=self.get_decoder(),
            )

    def __repr__(self):
        return "<K2ISDataSet for pattern={} nav_shape={}>".format(
            _pattern(self._path), self._nav_shape
        )


class K2ISPartition(BasePartition):
    def validate_tiling_scheme(self, tiling_scheme):
        a = len(tiling_scheme.shape) == 3
        b = tiling_scheme.shape[1] % 930 == 0
        c = tiling_scheme.shape[2] % 16 == 0
        if not (a and b and c):
            raise ValueError(
                "Invalid tiling scheme: needs to be aligned to blocksize (930, 16)"
            )
