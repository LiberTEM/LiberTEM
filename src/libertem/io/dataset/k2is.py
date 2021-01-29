# -*- encoding: utf-8 -*-
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

from libertem.common.buffers import zeros_aligned
from libertem.common import Shape
from libertem.web.messages import MessageConverter
from .base import (
    DataSet, BasePartition, DataSetException, DataSetMeta,
    FileSet, LocalFile, Decoder, make_get_read_ranges,
    TilingScheme,
)

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
      },
      "required": ["type", "path"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path"]
        }
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
                    # f = fileset_arr[sector_id]

                    # "linear" block index per sector:
                    blockidx = (15 - sector_index_x) + sector_index_y * 16
                    offset = (
                        frame_in_file_idx * BLOCK_SIZE * BLOCKS_PER_SECTOR_PER_FRAME
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
        pattern = "%s*.bin" % path
    elif ext == ".bin":
        pattern = "%s*.bin" % (
            re.sub(r'[0-9]+$', '', path)
        )
    else:
        raise DataSetException("unknown extension: %s" % ext)
    return pattern


def _get_gtg_path(full_path):
    path, ext = os.path.splitext(full_path)
    ext = ext.lower()
    if ext == ".gtg":
        return full_path
    elif ext == ".bin":
        return "%s.gtg" % (
            re.sub(r'[0-9]+$', '', path)
        )


class K2Syncer:
    def __init__(self, paths, start_offsets=None):
        self.paths = paths
        if start_offsets is None:
            start_offsets = NUM_SECTORS * [0]
        self.sectors = [Sector(fname, idx, initial_offset=start_offset)
                        for ((idx, fname), start_offset) in zip(enumerate(paths), start_offsets)]

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

    def sync(self):
        self.sync_sectors()
        self.sync_to_first_frame()
        self.validate_sync()

    def first_blocks(self):
        return [next(s.get_blocks()) for s in self.sectors]

    def close(self):
        for s in self.sectors:
            s.close()


class Sector:
    def __init__(self, fname, idx, initial_offset=0):
        self.fname = fname
        self.idx = idx
        self.filesize = os.stat(fname).st_size
        self.first_block_offset = initial_offset
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
        while offset + BLOCK_SIZE < self.filesize:
            yield DataBlock(offset=offset, sector=self)
            offset += BLOCK_SIZE

    def set_first_block_offset(self, offset):
        self.first_block_offset = offset

    def first_block(self):
        return next(self.get_blocks())

    def first_block_with(self, predicate=lambda b: True):
        return next(b for b in self.get_blocks()
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
            if type(dtype) != str:
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
        return k2is_get_read_ranges(**kwargs)


class K2ISFile(LocalFile):
    def _mmap_to_array(self, raw_mmap, start, stop):
        return np.frombuffer(raw_mmap, dtype=self._native_dtype)


class K2ISDataSet(DataSet):
    """
    Read raw K2IS data sets. They consist of 8 .bin files and one .gtg file.
    Currently, data acquired using the STEMx unit is supported, metadata
    about the scan size is read from the .gtg file.

    Parameters
    ----------
    path: str
        Path to one of the files of the data set (either one of the .bin files or the .gtg file)
    """

    def __init__(self, path, io_backend=None):
        super().__init__(io_backend=io_backend)
        self._path = path
        self._start_offsets = None
        # NOTE: the sync flag appears to be set one frame too late, so
        # we compensate here by setting a negative _skip_frames value.
        # skip_frames is applied after synchronization.
        self._skip_frames = -1
        self._files = None
        self._sync_offset = 0

    def _do_initialize(self):
        self._files = self._get_files()
        self._get_syncer(do_sync=True)
        self._scan_size = self._get_scansize()
        self._image_count = int(np.prod(self._scan_size))
        self._nav_shape_product = self._image_count
        self._sync_offset_info = self.get_sync_offset_info()
        self._meta = DataSetMeta(
            shape=Shape(self._scan_size + (SECTOR_SIZE[0], NUM_SECTORS * SECTOR_SIZE[1]),
                     sig_dims=2),
            raw_dtype=np.dtype("uint16"),
            sync_offset=self._sync_offset,
            image_count=self._image_count,
        )
        return self

    def initialize(self, executor):
        return executor.run_function(self._do_initialize)

    def _get_scansize(self):
        with dm.fileDM(_get_gtg_path(self._path), on_memory=True) as dm_file:
            return (int(dm_file.allTags['.SI Dimensions.Size Y']),
                    int(dm_file.allTags['.SI Dimensions.Size X']))

    def _scansize_without_flyback(self):
        with dm.fileDM(_get_gtg_path(self._path), on_memory=True) as dm_file:
            ss = (
                dm_file.allTags['.SI Image Tags.SI.Acquisition.Spatial Sampling.Height (pixels)'],
                dm_file.allTags['.SI Image Tags.SI.Acquisition.Spatial Sampling.Width (pixels)']
            )
            return tuple(int(i) for i in ss)

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
        return set(["gtg", "bin"])

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
            pattern = _pattern(path)
            files = executor.run_function(glob.glob, pattern)
            if len(files) != NUM_SECTORS:
                return False
        except DataSetException:
            return False
        return {
            "parameters": {
                "path": path,
            },
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
        }

    def get_diagnostics(self):
        with self._get_syncer().sectors[0] as sector:
            est_num_frames = sector.filesize // BLOCK_SIZE // BLOCKS_PER_SECTOR_PER_FRAME
            first_block = next(sector.get_blocks())
        fs_nosync = self._get_syncer(do_sync=False)
        sector_nosync = fs_nosync.sectors[0]
        first_block_nosync = next(sector_nosync.get_blocks())

        return [
            {"name": "first block offsets for all sectors",
             "value": ", ".join([str(s) for s in self._start_offsets])},

            {"name": "est. number of frames (from first sector)",
             "value": str(est_num_frames)},

            {"name": "first frame id after sync, (from first sector)",
             "value": str(first_block.header['frame_id'])},

            {"name": "first frame id before sync (from first sector)",
             "value": str(first_block_nosync.header['frame_id'])},
        ]

    def _get_files(self):
        pattern = _pattern(self._path)
        files = glob.glob(pattern)
        if len(files) != NUM_SECTORS:
            raise DataSetException("expected %d files at %s, found %d" % (
                NUM_SECTORS,
                pattern,
                len(files)
            ))
        return list(sorted(files))

    def _cache_first_block_offsets(self, fs):
        self._start_offsets = [
            s.first_block_offset
            for s in fs.sectors
        ]
        # apply skip_frames value to the start_offsets
        self._start_offsets = [o + BLOCK_SIZE*self._skip_frames*32
                               for o in self._start_offsets]

    def _get_syncer(self, do_sync=True):
        if not do_sync:
            return K2Syncer(self._files)
        if self._start_offsets is None:
            sy = K2Syncer(self._files)
            sy.sync()
            self._cache_first_block_offsets(sy)
        else:
            sy = K2Syncer(self._files, start_offsets=self._start_offsets)
        return sy

    def _get_fileset(self, with_sync=True):
        sig_shape = (SECTOR_SIZE[0], NUM_SECTORS * SECTOR_SIZE[1])
        num_frames = int(np.prod(self._scan_size))
        files = [
            K2ISFile(
                path=path,
                start_idx=0,
                end_idx=num_frames,
                native_dtype=np.uint8,
                sig_shape=sig_shape,
                file_header=offset,
            )
            for path, offset in zip(self._files, self._start_offsets)
        ]
        return K2FileSet(files=files)

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 1024MB (converted float data) per partition
        partition_size_px = 1024 * 1024 * 1024 // 4
        total_size_px = np.prod(self.shape, dtype=np.int64)
        res = max(self._cores, total_size_px // partition_size_px)
        return res

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in K2ISPartition.make_slices(
                shape=self.shape,
                num_partitions=self._get_num_partitions()):
            yield K2ISPartition(
                meta=self._meta,
                fileset=fileset.get_for_range(start, stop),
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
                io_backend=self.get_io_backend(),
            )

    def __repr__(self):
        return "<K2ISDataSet for pattern=%s scan_size=%s>" % (
            _pattern(self._path), self._scan_size
        )


class K2ISPartition(BasePartition):
    def _get_decoder(self):
        return K2ISDecoder()

    def validate_tiling_scheme(self, tiling_scheme):
        a = len(tiling_scheme.shape) == 3
        b = tiling_scheme.shape[1] % 930 == 0
        c = tiling_scheme.shape[2] % 16 == 0
        if not (a and b and c):
            raise ValueError(
                "Invalid tiling scheme: needs to be aligned to blocksize (930, 16)"
            )

    def get_base_shape(self):
        return (1, 930, 16)
