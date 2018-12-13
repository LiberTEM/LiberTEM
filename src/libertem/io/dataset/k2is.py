# -*- encoding: utf-8 -*-
import os
import re
import glob
import math
import mmap
import logging
import itertools
from contextlib import contextmanager

import numpy as np
import numba

from libertem.common import Slice, Shape
from .base import DataSet, Partition, DataTile, DataSetException

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


@numba.njit
def decode_uint12_le(inp, out):
    """
    decode bytes from bytestring ``inp`` as 12 bit into ``out``

    based partially on https://stackoverflow.com/a/45070947/540644
    """
    assert np.mod(len(inp), 3) == 0

    for i in range(len(inp) // 3):
        fst_uint8 = np.uint16(inp[i * 3])
        mid_uint8 = np.uint16(inp[i * 3 + 1])
        lst_uint8 = np.uint16(inp[i * 3 + 2])

        a = fst_uint8 | (mid_uint8 & 0x0F) << 8
        b = (mid_uint8 & 0xF0) >> 4 | lst_uint8 << 4
        out[i * 2] = a
        out[i * 2 + 1] = b


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


class K2FileSet:
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

    def get_slice(self, start, stop):
        """
        start, stop: frame indices
        """
        return Slice(
            origin=(
                start, 0, SECTOR_SIZE[1] * self.idx
            ),
            shape=Shape(((stop - start),) + SECTOR_SIZE, sig_dims=self.sig_dims)
        )

    def get_block_by_index(self, idx):
        offset = self.first_block_offset + idx * BLOCK_SIZE
        return DataBlock(offset=offset, sector=self)

    def get_blocks(self):
        offset = self.first_block_offset
        while offset + BLOCK_SIZE < self.filesize:
            yield DataBlock(offset=offset, sector=self)
            offset += BLOCK_SIZE

    def read_stacked(self, start_at_frame, num_frames, stackheight=16,
                     dtype="float32", crop_to=None):
        """
        Reads `stackheight` blocks into a single buffer.
        The blocks are read from consecutive frames, always
        from the same coordinates inside the sector of the frame.

        yields DataTiles of the shape (stackheight, 930, 16)
        (different tiles at the borders may be yielded if the stackheight doesn't evenly divide
        the total number of frames to read)
        """
        tileshape = (
            stackheight,
        ) + BLOCK_SHAPE
        raw_data = mmap.mmap(
            fileno=self.f.fileno(),
            length=0,   # whole file
            access=mmap.ACCESS_READ,
        )

        tile_buf_full = np.zeros(tileshape, dtype=dtype)
        assert DATA_SIZE % 3 == 0
        log.debug("starting read_stacked with start_at_frame=%d, num_frames=%d, stackheight=%d",
                  start_at_frame, num_frames, stackheight)
        for outer_frame in range(start_at_frame, start_at_frame + num_frames, stackheight):
            # log.debug("outer_frame=%d", outer_frame)
            # end of the selected frame range, calculate rest of stack:
            if start_at_frame + num_frames - outer_frame < stackheight:
                end_frame = start_at_frame + num_frames
                current_stackheight = end_frame - outer_frame
                current_tileshape = (
                    current_stackheight,
                ) + BLOCK_SHAPE
                tile_buf = np.zeros(current_tileshape, dtype=dtype)
            else:
                current_stackheight = stackheight
                current_tileshape = tileshape
                tile_buf = tile_buf_full
            for blockidx in range(BLOCKS_PER_SECTOR_PER_FRAME):
                start_x = (self.idx + 1) * 256 - (16 * (blockidx % 16 + 1))
                start_y = 930 * (blockidx // 16)
                tile_slice = Slice(
                    origin=(
                        outer_frame,
                        start_y,
                        start_x,
                    ),
                    shape=Shape(current_tileshape, sig_dims=self.sig_dims),
                )
                if crop_to is not None:
                    intersection = tile_slice.intersection_with(crop_to)
                    if intersection.is_null():
                        continue
                offset = (
                    self.first_block_offset
                    + outer_frame * BLOCK_SIZE * BLOCKS_PER_SECTOR_PER_FRAME
                    + blockidx * BLOCK_SIZE
                )
                for frame in range(current_stackheight):
                    block_offset = (
                        offset + frame * BLOCK_SIZE * BLOCKS_PER_SECTOR_PER_FRAME
                    )
                    input_start = block_offset + HEADER_SIZE
                    input_end = block_offset + HEADER_SIZE + DATA_SIZE
                    out = tile_buf[frame].reshape((-1,))
                    decode_uint12_le(
                        inp=raw_data[input_start:input_end],
                        out=out,
                    )
                yield DataTile(
                    data=tile_buf,
                    tile_slice=tile_slice
                )
        raw_data.close()

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
        arr = np.zeros((930 * 16), dtype="uint16")
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


class K2ISDataSet(DataSet):
    def __init__(self, path, scan_size):
        self._path = path
        self._scan_size = tuple(scan_size)
        self._sig_dims = 2
        self._start_offsets = None
        # NOTE: the sync flag appears to be set one frame too late, so
        # we compensate here by setting a negative _skip_frames value.
        # skip_frames is applied after synchronization.
        self._skip_frames = -1
        self._files = None
        self._fileset = None

    def initialize(self):
        self._files = self._get_files()
        self._fileset = self._get_fileset()
        return self

    @property
    def dtype(self):
        return np.dtype("uint16")

    @property
    def shape(self):
        return Shape(self._scan_size + (SECTOR_SIZE[0], NUM_SECTORS * SECTOR_SIZE[1]),
                     sig_dims=self._sig_dims)

    @property
    def raw_shape(self):
        # FIXME: the number of frames should come from the dataset, not from the user (scan_size)
        ss = self._scan_size
        return Shape((ss[0] * ss[1], SECTOR_SIZE[0], NUM_SECTORS * SECTOR_SIZE[1]),
                     sig_dims=self._sig_dims)

    @classmethod
    def detect_params(cls, path):
        """
        detect if path points to a file that is part of a k2is dataset.
        extension needs to be bin or gtg, and number of files of the dataset needs to match
        the number of sectors. no further checking is done yet. in certain cases there may be false
        positives, for example if users name their binary files .bin and have 8 of them
        in a directory.
        """
        try:
            pattern = _pattern(path)
            files = glob.glob(pattern)
            if len(files) != NUM_SECTORS:
                return False
        except DataSetException:
            return False
        params = {
            "path": path,
        }
        try:
            # FIXME: hyperspy is GPL, so this would infect our K2 reader:
            """
            from hyperspy.io_plugins.digital_micrograph import DigitalMicrographReader
            with open(_get_gtg_path(path), "rb") as f:
                reader = DigitalMicrographReader(f)
                reader.parse_file()
            size = reader.tags_dict['SI Dimensions']
            params['scan_size'] = (size['Size Y'], size['Size X'])
            """
        except Exception:
            pass
        return params

    def check_valid(self):
        try:
            fs = self._get_fileset()
            fs.sync()
        except Exception as e:
            raise DataSetException("failed to load dataset: %s" % e) from e
        return True

    def get_diagnostics(self):
        p = next(self.get_partitions())
        with p._get_sector() as sector:
            est_num_frames = sector.filesize // BLOCK_SIZE // BLOCKS_PER_SECTOR_PER_FRAME
            first_block = next(sector.get_blocks())
        fs_nosync = self._get_fileset(with_sync=False)
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

    def _get_fileset(self, with_sync=True):
        if not with_sync:
            return K2FileSet(self._files)
        if self._start_offsets is None:
            fs = K2FileSet(self._files)
            fs.sync()
            self._cache_first_block_offsets(fs)
        else:
            fs = K2FileSet(self._files, start_offsets=self._start_offsets)
        return fs

    def _get_partitions_per_file(self):
        """
        returns the number of partitions each file (sector) should be split into
        """
        sector0 = self._fileset.sectors[0]
        size = sector0.filesize
        # let's try to aim for 512MB per partition
        res = max(1, size // (512*1024*1024))
        return res

    def get_partitions(self):
        fs = self._fileset
        num_frames = self.shape.nav.size
        f_per_part = num_frames // self._get_partitions_per_file()

        for sector in fs.sectors:
            c0 = itertools.count(start=0, step=f_per_part)
            c1 = itertools.count(start=f_per_part, step=f_per_part)
            for (start, stop) in zip(c0, c1):
                if start >= num_frames:
                    break
                stop = min(stop, num_frames)
                yield K2ISPartition(
                    dataset=self,
                    dtype=self.dtype,
                    partition_slice=sector.get_slice(
                        start=start, stop=stop
                    ),
                    path=sector.fname,
                    index=sector.idx,
                    offset=sector.first_block_offset,
                    start_frame=start,
                    num_frames=stop - start,
                )

    def __repr__(self):
        return "<K2ISDataSet for pattern=%s scan_size=%s>" % (
            _pattern(self._path), self._scan_size
        )


class K2ISPartition(Partition):
    def __init__(self, path, index, offset, start_frame, num_frames,
                 strategy='READ_STACKED', *args, **kwargs):
        self._path = path
        self._index = index
        self._offset = offset
        self._strategy = strategy
        self._start_frame = start_frame
        self._num_frames = num_frames
        super().__init__(*args, **kwargs)

    @contextmanager
    def _get_sector(self):
        with Sector(
            fname=self._path,
            idx=self._index,
            initial_offset=self._offset,
        ) as sector:
            yield sector

    def get_tiles(self, crop_to=None, strat=None):
        if strat is None:
            strat = self._strategy
        if strat == 'BLOCK_BY_BLOCK':
            yield from self._get_tiles_tile_per_datablock()
        elif strat == 'READ_STACKED':
            yield from self._read_stacked(crop_to=crop_to)
        else:
            raise DataSetException("unknown strategy")

    def _read_stacked(self, crop_to=None):
        with self._get_sector() as s:
            yield from s.read_stacked(
                start_at_frame=self._start_frame,
                num_frames=self._num_frames,
                crop_to=crop_to
            )

    def _get_tiles_tile_per_datablock(self):
        """
        yield one tile per underlying data block
        """

        with self._get_sector() as s:
            all_blocks = s.get_blocks()
            blocks_to_read = (
                BLOCKS_PER_SECTOR_PER_FRAME * self.slice.shape.nav.size,
            )
            buf = np.zeros((1, 1) + BLOCK_SHAPE, dtype="float32")
            for block_idx, block in enumerate(itertools.islice(all_blocks, blocks_to_read)):
                frame_idx = block_idx // BLOCKS_PER_SECTOR_PER_FRAME
                h = block.header
                # FIXME: hardcoded sig_dims
                sector_offset = SECTOR_SIZE[1] * block.sector.idx
                tile_slice = Slice(
                    origin=(frame_idx,
                            h['pixel_y_start'],
                            sector_offset + h['pixel_x_start']),
                    shape=Shape((1,) + BLOCK_SHAPE, sig_dims=2),
                )
                block.readinto(buf)
                yield DataTile(
                    data=buf,
                    tile_slice=tile_slice
                )

    def __repr__(self):
        return "<K2ISPartition: sector %d, start_frame=%d, num_frames=%d>" % (
            self._index, self._start_frame, self._num_frames,
        )
