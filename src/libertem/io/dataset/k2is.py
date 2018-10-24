# -*- encoding: utf-8 -*-
import os
import re
import glob
import mmap
import logging
import itertools

import numpy as np
import numba

from libertem.common.slice import Slice
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
            offset = s.first_block_with(lambda b: b.shutter_active).offset
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
        self.f = open(fname, "rb")
        self.idx = idx
        self.filesize = os.fstat(self.f.fileno()).st_size
        self.first_block_offset = initial_offset

    def seek(self, pos):
        self.f.seek(pos)

    def read(self, size):
        return self.f.read(size)

    def get_slice(self, scan_size, start, stop):
        # we are working on full rows here:
        assert start % scan_size[1] == 0
        assert (stop - start) % scan_size[1] == 0
        return Slice(
            origin=(
                start // scan_size[1],
                0,
                0, SECTOR_SIZE[1] * self.idx
            ),
            shape=(
                (stop - start) // scan_size[1],
                scan_size[1],
            ) + SECTOR_SIZE
        )

    def get_blocks(self):
        offset = self.first_block_offset
        while offset + BLOCK_SIZE < self.filesize:
            yield DataBlock(offset=offset, sector=self)
            offset += BLOCK_SIZE

    def read_stacked(self, scan_width, total_frames, num_frames, start_at_frame=0,
                     stackheight=16,
                     dtype="float32", crop_to=None):
        """
        Reads `stackheight` blocks into a single buffer.
        The blocks are read from consecutive frames, always
        from the same coordinates inside the sector of the frame.

        yields DataTiles of the shape (1, stackheight, 930, 16)
        (different tiles at the borders may be yielded if the stackheight doesn't divide
        the scan_width)
        """
        assert stackheight <= scan_width
        assert start_at_frame % scan_width == 0
        assert num_frames % scan_width == 0, "read_stacked should be called for whole scan rows"
        tileshape = (
            1, stackheight
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
        for row in range(start_at_frame // scan_width, (start_at_frame + num_frames) // scan_width):
            for outer_frame in range(row * scan_width, (row + 1) * scan_width, stackheight):

                # log.debug("outer_frame=%d", outer_frame)
                for blockidx in range(BLOCKS_PER_SECTOR_PER_FRAME):
                    offset = (
                        self.first_block_offset
                        + outer_frame * BLOCK_SIZE * BLOCKS_PER_SECTOR_PER_FRAME
                        + blockidx * BLOCK_SIZE
                    )
                    # end of the row, calculate rest of stack:
                    if outer_frame % scan_width > (outer_frame + stackheight) % scan_width:
                        end_frame = (row + 1) * scan_width  # last frame of the row
                        current_stackheight = end_frame - outer_frame
                        current_tileshape = (
                            1, current_stackheight
                        ) + BLOCK_SHAPE
                        tile_buf = np.zeros(current_tileshape, dtype=dtype)
                    else:
                        current_stackheight = stackheight
                        current_tileshape = tileshape
                        tile_buf = tile_buf_full
                    start_x = (self.idx + 1) * 256 - (16 * (blockidx % 16 + 1))
                    start_y = 930 * (blockidx // 16)
                    tile_slice = Slice(
                        origin=(
                            outer_frame // scan_width,
                            outer_frame % scan_width,
                            start_y,
                            start_x,
                        ),
                        shape=current_tileshape,
                    )
                    if crop_to is not None:
                        intersection = tile_slice.intersection_with(crop_to)
                        if intersection.is_null():
                            continue
                    for frame in range(current_stackheight):
                        block_offset = (
                            offset + frame * BLOCK_SIZE * BLOCKS_PER_SECTOR_PER_FRAME
                        )
                        input_start = block_offset + HEADER_SIZE
                        input_end = block_offset + HEADER_SIZE + DATA_SIZE
                        decode_uint12_le(
                            inp=raw_data[input_start:input_end],
                            out=tile_buf[0, frame].ravel()
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

    def close(self):
        self.f.close()

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
    def __init__(self, path, scan_size, skip_frames=0):
        self._path = path
        self._scan_size = tuple(scan_size)
        self._start_offsets = None
        self._skip_frames = skip_frames

    @property
    def dtype(self):
        return np.dtype("uint16")

    @property
    def shape(self):
        return self._scan_size + (SECTOR_SIZE[0], NUM_SECTORS * SECTOR_SIZE[1])

    def check_valid(self):
        try:
            fs = self._get_fileset()
            fs.sync()
        except Exception as e:
            raise DataSetException("failed to load dataset: %s" % e)
        return True

    def get_diagnostics(self):
        p = next(self.get_partitions())
        sector = p._get_sector()
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

            {"name": "first frame id after sync, including skip_frames (from first sector)",
             "value": str(first_block.header['frame_id'])},

            {"name": "first frame id before sync (from first sector)",
             "value": str(first_block_nosync.header['frame_id'])},
        ]

    def _pattern(self):
        path, ext = os.path.splitext(self._path)
        if ext == ".gtg":
            pattern = "%s*.bin" % path
        elif ext == ".bin":
            pattern = "%s*.bin" % (
                re.sub(r'[0-9]+$', '', path)
            )
        else:
            raise DataSetException("unknown extension: %s" % ext)
        return pattern

    def _files(self):
        pattern = self._pattern()
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
        # FIXME
        self._start_offsets = [o + BLOCK_SIZE*self._skip_frames*32
                               for o in self._start_offsets]

    def _get_fileset(self, with_sync=True):
        if not with_sync:
            return K2FileSet(self._files())
        if self._start_offsets is None:
            fs = K2FileSet(self._files())
            fs.sync()
            self._cache_first_block_offsets(fs)
        else:
            fs = K2FileSet(self._files(), start_offsets=self._start_offsets)
        return fs

    def _get_partitions_per_file(self):
        sector0_fname = self._files()[0]
        stat = os.stat(sector0_fname)
        size = stat.st_size
        # let's try to aim for 512MB per partition
        return size // (512*1024*1024)

    def get_partitions(self):
        fs = self._get_fileset()
        rows, cols = self._scan_size
        num_frames = rows * cols
        approx_f_per_part = num_frames // self._get_partitions_per_file()
        f_per_part = (approx_f_per_part // cols) * cols
        try:
            for s in fs.sectors:
                c0 = itertools.count(start=0, step=f_per_part)
                c1 = itertools.count(start=f_per_part, step=f_per_part)
                for (start, stop) in zip(c0, c1):
                    if start >= num_frames:
                        break
                    stop = min(stop, num_frames)
                    yield K2ISPartition(
                        dataset=self,
                        dtype=self.dtype,
                        partition_slice=s.get_slice(
                            scan_size=self._scan_size, start=start, stop=stop
                        ),
                        path=s.fname,
                        index=s.idx,
                        offset=s.first_block_offset,
                        scan_size=self._scan_size,
                        start_frame=start,
                        num_frames=stop - start,
                    )
        finally:
            fs.close()

    def __repr__(self):
        return "<K2ISDataSet for pattern=%s scan_size=%s>" % (
            self._pattern(), self._scan_size
        )


class K2ISPartition(Partition):
    def __init__(self, path, index, offset, scan_size, start_frame, num_frames,
                 strategy='READ_STACKED', *args, **kwargs):
        self._path = path
        self._index = index
        self._offset = offset
        self._scan_size = scan_size
        self._strategy = strategy
        self._start_frame = start_frame
        self._num_frames = num_frames
        super().__init__(*args, **kwargs)

    def _get_sector(self):
        return Sector(
            fname=self._path,
            idx=self._index,
            initial_offset=self._offset
        )

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
        s = self._get_sector()

        try:
            # TODO: stackheight parameter?
            scan_size = self._scan_size
            yield from s.read_stacked(
                scan_width=scan_size[1],
                total_frames=scan_size[0] * scan_size[1],
                num_frames=self._num_frames, start_at_frame=self._start_frame,
                crop_to=crop_to
            )
        finally:
            s.close()

    def _get_tiles_tile_per_datablock(self):
        """
        yield one tile per underlying data block
        """
        s = self._get_sector()
        scan = self._scan_size

        try:
            all_blocks = s.get_blocks()
            blocks_to_read = (
                BLOCKS_PER_SECTOR_PER_FRAME * scan[0] * scan[1]
            )
            buf = np.zeros((1, 1) + BLOCK_SHAPE, dtype="float32")
            for block_idx, block in enumerate(itertools.islice(all_blocks, blocks_to_read)):
                frame_idx = block_idx // BLOCKS_PER_SECTOR_PER_FRAME
                scan_pos_y = frame_idx // scan[1]
                scan_pos_x = frame_idx % scan[1]
                h = block.header
                # TODO: move tile_slice stuff to datablock?
                sector_offset = SECTOR_SIZE[1] * block.sector.idx
                tile_slice = Slice(
                    origin=(scan_pos_y, scan_pos_x,
                            h['pixel_y_start'],
                            sector_offset + h['pixel_x_start']),
                    shape=(1, 1) + BLOCK_SHAPE,
                )
                block.readinto(buf)
                yield DataTile(
                    data=buf,
                    tile_slice=tile_slice
                )
        finally:
            s.close()

    def get_locations(self):
        return "127.0.1.1"  # FIXME

    def __repr__(self):
        return "<K2ISPartition: sector %d, start_frame=%d, num_frames=%d>" % (
            self._get_sector().idx, self._start_frame, self._num_frames,
        )
