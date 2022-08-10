from __future__ import annotations
import os
import pathlib
from functools import partial
import numpy as np
import numba
import itertools
import typing
import struct
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import sparse

from epoch_ordering import compute_epoch
from libertem.common.slice import Slice
from libertem.common.shape import Shape
from libertem.io.dataset.base.tiling_scheme import TilingScheme
from libertem.io.dataset.base.tiling import DataTile

if typing.TYPE_CHECKING:
    from io import BufferedReader

PACKET_SIZE = 8  # bytes (all are packed uint64)
CHIP_NUMBERS = (0, 1, 2, 3)
MODES = (0,)
PACKAGE_TYPES = {
    0x7: 'control',
    0x4: 'heartbeat',
    0x6: 'tdc_timestamp',
    0xb: 'events',
}
ROLLOVER_TIME = 26.8435456  # seconds


def get_search_boundaries(filepath, num):
    filesize = os.stat(filepath).st_size
    extra = filesize % PACKET_SIZE
    assert extra == 0
    normal_chunksize = filesize // num
    normal_chunksize = (normal_chunksize // PACKET_SIZE) * PACKET_SIZE
    normal_chunks = num - 1
    final_chunksize = normal_chunksize + (filesize // (normal_chunksize * normal_chunks))
    chunksizes = (normal_chunksize,) * normal_chunks + (final_chunksize,)
    upper_boundaries = tuple(itertools.accumulate(chunksizes))
    lower_boundaries = (0,) + upper_boundaries[:-1]
    return lower_boundaries, upper_boundaries


@dataclass
class Header:
    chip_nr: int
    mode: int
    size: int
    offset: int
    pkg_type: int = None
    spidr_time: int = None
    fine_time: int = None
    epoch: int = None

    @staticmethod
    def get_size(size_lsb, size_msb):
        return ((0xff & size_msb) << 8) | (0xff & size_lsb)


# dt_hit = np.dtype([
#     ('chipId', np.uint8),
#     ('x', np.uint16),
#     ('y', np.uint16),
#     ('ToT', np.uint16),
#     ('ToA', np.int64)
# ])


# dt_header = np.dtype([
#     ('chipId', np.uint8),
#     ('size', np.uint16),
#     ('offset', np.uint64),
#     ('spidr_time', np.uint16),
#     ('global_time', np.int64),
# ])



IDENT = tuple(ord(c) for c in 'TPX3')

def get_header(data, offset):
    unpacked = struct.unpack('<bbbbbbbb', data)
    if unpacked[:4] != IDENT:
        return None

    size = Header.get_size(*unpacked[-2:])
    header = Header(*unpacked[4:-2], size, offset)

    if header.mode not in MODES:
        return None
    if header.chip_nr not in CHIP_NUMBERS:
        return None
    if header.size < 0:
        return None
    return header


def read_header(fp, cursor):
    header_bytes = fp.read(PACKET_SIZE)
    if len(header_bytes) < PACKET_SIZE:
        raise OSError(f'End of file before full header {fp.tell()}')
    header = get_header(header_bytes, cursor)
    cursor = cursor + PACKET_SIZE
    return header, cursor


def skip_package(fp, cursor, header):
    fp.seek(header.size, 1)
    cursor = cursor + header.size
    return cursor


def read_first_package(fp: BufferedReader, header=None):
    # Read the first package of the data package to figure out its type
    pkg_data = fp.read(PACKET_SIZE)
    fp.seek(-PACKET_SIZE, 1)
    if len(pkg_data) < PACKET_SIZE:
        raise NotImplementedError('Truncated file')

    # <Q unsigned long long little-endian
    pkg = np.frombuffer(pkg_data, dtype=np.uint64, count=1)
    pkg_type = pkg >> 60
    if pkg_type.item() not in PACKAGE_TYPES.keys():
        raise NotImplementedError('Unrecognized package type {pkg_type}')

    # The SPIDR time is 16 bit (65536). It has a rollover time of ~26.84354 seconds
    spidr_time, coarse_toa, fine_toa = parse_time_data(pkg)
    spidr_time = spidr_time.item()
    coarse_toa = coarse_toa.item()
    fine_toa = fine_toa.item()
    # Combine spidr with fine and coarse toa give a 34-bit timestamp
    # with the same rollover time, which gives a resolution of 1.56 ns
    # this is subject to some jitter as the first hit in
    # the packet is not necessarily the first hit in time
    _, fine_time = convert_time_data(spidr_time, coarse_toa, fine_toa, 0)
    # the 30 bit timestamp has a near-perfect 25 ns resolution
    # 2.5e-9 * (2**30) = 26.8435456, which can be accessed by fine_time >> 4

    if header is not None:
        header.pkg_type = pkg_type
        header.spidr_time = spidr_time
        header.fine_time = fine_time
        return header

    return pkg_type, spidr_time, fine_time


def parse_headers(filepath, start, end):
    cursor = start
    headers = []
    header = None

    with filepath.open("rb") as fp:
        fp.seek(cursor)

        while cursor < end:
            if header:
                header = read_first_package(fp, header=header)
                cursor = skip_package(fp, cursor, header)
                try:
                    next_header, cursor = read_header(fp, cursor)
                except OSError:
                    # end of file
                    break
                if next_header:
                    headers.append(header)
                    header = next_header
                else:
                    raise RuntimeError(f'Header gave bad package size {header.size}')
            else:
                header, cursor = read_header(fp, cursor)
        return headers, cursor


def parse_file_headers(filepath, num_part=16) -> tuple[Header]:
    boundaries = get_search_boundaries(filepath, num_part)

    with ThreadPoolExecutor(num_part) as pool:
        results = pool.map(partial(parse_headers, filepath), *boundaries)
        results = tuple(results)
    # results = tuple(parse_headers(filepath, lb, ub) for lb, ub in zip(*boundaries))

    counters = tuple(r[1] for r in results)
    headers = tuple(itertools.chain.from_iterable(r[0] for r in results))
    # assert all(c0 >= c1 for c0, c1 in zip(counters[:-1], counters[1:]))
    return headers


@numba.njit
def parse_time_data(hit_data: np.ndarray):
    # spidr_time == uint16
    # coarse_toa == uint14
    # fine_toi == uint4    
    spidr_time = (hit_data & np.uint64(0xffff))
    coarse_toa = (hit_data >> np.uint64(16 + 14) & np.uint64(0x3fff))
    fine_toa = (hit_data >> np.uint64(16)) & np.uint64(0xf)
    return (spidr_time.astype(np.uint16),
            coarse_toa.astype(np.uint16),
            fine_toa.astype(np.uint16))


@numba.njit
def parse_time_data_inplace(hit_data: np.ndarray, out_buffer: np.ndarray):
    # spidr
    out_buffer[0] = np.bitwise_and(hit_data, np.uint64(0xffff))
    # coarse_toa
    out_buffer[1] = np.bitwise_and(hit_data >> np.uint64(16 + 14), np.uint64(0x3fff))
    # fine_toa
    out_buffer[2] = np.bitwise_and(hit_data >> np.uint64(16), np.uint64(0xf))


@numba.njit
def parse_hit_data(hit_data: np.ndarray, chip_nr: int, out_buffer: np.ndarray, cross_offset: int = 0):
    col = (hit_data & np.uint64(0x0FE0000000000000)) >> np.uint64(52)
    super_pix = (hit_data & np.uint64(0x001F800000000000)) >> np.uint64(45)
    pix = (hit_data & np.uint64(0x0000700000000000)) >> np.uint64(44)
    parse_time_data_inplace(hit_data, out_buffer[3:6])
    # tot
    out_buffer[2] = ((hit_data >> np.uint64(16 + 4)) & np.uint64(0x3ff)).astype(np.uint16)
    convert_position_data(col.astype(np.uint16),
                          super_pix.astype(np.uint16),
                          pix.astype(np.uint16),
                          chip_nr,
                          out_buffer[0],
                          out_buffer[1],
                          cross_offset=cross_offset)


@numba.njit
def convert_position_data(col: np.ndarray,
                          super_pix: np.ndarray,
                          pix: np.ndarray,
                          chip_nr: int,
                          out_x: np.ndarray,
                          out_y: np.ndarray,
                          cross_offset: int = 0):
    # the x/y position values are for one chip so all np.uint8 (256x256 chip)
    # potential overflow issues here ?? probably impossible given the chip design
    out_x[:] = col + (pix >> np.uint16(2))
    out_y[:] = super_pix + (pix & np.uint16(0x3))
    combine_chips(out_x, out_y, chip_nr, cross_offset=cross_offset)


@numba.njit
def combine_chips(x: np.ndarray, y: np.ndarray, chip_nr: int, cross_offset: int = 0):
    # correction is applied inplace on x and y

    # Chip are orientated like this (unspecified x/y orientation)
    # 2 1
    # 3 0
    
    # Calculate extra offset required for the cross pixels
    offset = np.uint16(256 + 2 * cross_offset)
    size = np.uint16(255)

    if chip_nr == 0:
        x[:] += offset
        y[:] = size - y + offset
    elif chip_nr == 1:
        x[:] = size - x + offset
    elif chip_nr == 2:
        x[:] = size - x
    elif chip_nr == 3:
        y[:] = size - y + offset


@numba.njit
def convert_time_data(spidr_time: np.ndarray, coarse_toa: np.ndarray, fine_toa: np.ndarray, epoch: int):
    # I'm assuming that the epoch number cannot change within a data packet
    # need to check with the documentation if that's reasonable
    # it's a solvable problem but it would require extra operations
    # to verify there is no rollover and correct the global timestamp
    # the example/upstream code *definitely* does not handle this case!
    #
    # spidr_time (u16 bit)
    # coarse_toa (u14 bit)
    # fine_toa (u4 bit)
    toa = (coarse_toa << np.uint64(4)) - fine_toa
    # toa_phase_correction would go here
    # FIXME
    # The ToA can be negative, so this is added using arithmetic operation
    global_time = (spidr_time << np.uint64(18)) | (np.uint64(epoch) << np.uint64(34)) + toa
    return toa, global_time


def add_epoch_value(headers, epochs):
    for epoch_number, slices in epochs:
        for sl in slices:
            for header in headers[sl]:
                header.epoch = epoch_number
    return headers


def time_to_timestamp(time, offset=0.):
    return int((time - offset) * 2**34 / 26.84354)


@numba.njit
def decode_headers(data: np.ndarray):
    components = np.asarray(data).reshape(-1).view(np.uint8).reshape(-1, 8)
    chip_nr = components[:, 4]
    size_lsb = components[:, -2].astype(np.uint16)
    size_msb = components[:, -1].astype(np.uint16)
    size = ((np.uint16(0xff) & size_msb) << np.uint16(8)) | (np.uint16(0xff) & size_lsb)
    # size is number of uint64
    return chip_nr, size // np.uint16(PACKET_SIZE)


@numba.njit
def decode_hits(data: np.ndarray, out_buffer: np.ndarray, cross_offset: int = 0):
    offset = np.uint64(0)  # number of uint64s
    header_size = np.uint64(1)

    total_hits = 0
    header_times = []
    while offset < data.size:
        chip_nr, package_size = decode_headers(data[offset:offset + header_size])
        chip_nr = chip_nr[0]
        package_size = package_size[0]

        first_pkg = data[offset + header_size:offset + header_size + header_size]
        pkg_type = first_pkg >> np.uint64(60)
        if pkg_type != np.uint64(0xb):
            offset += (header_size + package_size)
            continue
        spidr, _, _ = parse_time_data(first_pkg)
        header_times.append(spidr.item())

        hit_data = data[offset + header_size:offset + header_size + package_size]
        nhits = hit_data.size
        
        # from this point can read directly into output buffers
        parse_hit_data(hit_data,
                       chip_nr,
                       out_buffer[:, total_hits: total_hits + nhits],
                       cross_offset=cross_offset)
        
        out_buffer[6, total_hits: total_hits + nhits] = np.uint16(len(header_times) - 1)
        total_hits += nhits
        offset += (header_size + package_size)
    return total_hits, np.asarray(header_times, dtype=np.uint16)


@numba.njit
def compute_global_timestamps(hits: np.ndarray, start_time: np.uint64, out_buffer: np.ndarray):
    spidr_time = hits[3].astype(np.uint64)
    coarse_toa = hits[4].astype(np.uint64)
    fine_toa = hits[5].astype(np.uint64)
    toa = (coarse_toa << np.uint64(4)) - fine_toa
    out_buffer[:] = (spidr_time << np.uint64(18)) + toa
    out_buffer += start_time - out_buffer[0]


@numba.njit
def filter_and_compress(flat_idcs: np.ndarray, tot: np.ndarray,
                        global_timestamps: np.ndarray, start: np.uint64, end: np.uint64):
    timestamp_mask = np.logical_and(global_timestamps >= np.uint64(start),
                                    global_timestamps < np.uint64(end))
    return compress_hits(flat_idcs[timestamp_mask], tot[timestamp_mask])


def compress_hits_np(flat_idx: np.ndarray, tot: np.ndarray):
    unique_idcs, mapping = np.unique(flat_idx, return_inverse=True)
    accumulators = np.zeros((unique_idcs.size,), dtype=np.uint64)
    accumulators[mapping] += tot
    return unique_idcs, accumulators


@numba.njit
def compress_hits(flat_idx: np.ndarray, tot: np.ndarray):
    unique_idcs = np.unique(flat_idx)
    mapping = np.searchsorted(unique_idcs, flat_idx)
    accumulators = np.zeros((unique_idcs.size,), dtype=np.uint64)
    accumulators[mapping] += tot
    return unique_idcs, accumulators


def to_sparse_frame(data, start_time, start_span, end_span, sig_shape):
    maxhits = data.size
    hits = np.empty((7, maxhits), dtype=np.uint16)
    nhits, header_times = decode_hits(data, hits, cross_offset=2)
    hits = hits[:, :nhits]
    block_epochs = compute_epoch(header_times) # , start=data_headers_ar[read_start][EPOCH_IDX]
    if len(block_epochs) > 1:
        raise NotImplementedError('epoch_change in data block, must correct')
    global_timestamps = np.empty((nhits,), np.uint64)
    compute_global_timestamps(hits, start_time, global_timestamps)

    flat_idcs = np.ravel_multi_index((hits[1, :], hits[0, :]), sig_shape)
    unique_flat, values = filter_and_compress(flat_idcs, hits[2, :], global_timestamps,
                                              start_span, end_span)

    coords = np.stack(np.unravel_index(unique_flat, sig_shape), axis=0)
    return sparse.COO(coords, values, sorted=True, shape=sig_shape)


# @profile
def gen_sparse_tiles(data, frame_boundaries: np.ndarray, start_time: int,
                     tiling_scheme: TilingScheme, first_frame_idx: int):
    maxhits = data.size
    hits = np.empty((7, maxhits), dtype=np.uint16)
    nhits, header_times = decode_hits(data, hits, cross_offset=2)
    hits = hits[:, :nhits]
    block_epochs = compute_epoch(header_times)
    if len(block_epochs) > 1:
        raise NotImplementedError('epoch_change in data block, must correct')
    global_timestamps = np.empty((nhits,), np.uint64)
    compute_global_timestamps(hits, start_time, global_timestamps)

    ts_sorter = np.argsort(global_timestamps)
    ts_bounds_idcs = np.searchsorted(global_timestamps, frame_boundaries,
                                     side='left', sorter=ts_sorter)

    sig_shape = tuple(tiling_scheme.dataset_shape.sig)
    flat_idcs = np.ravel_multi_index((hits[1, :], hits[0, :]), sig_shape)

    frames = []
    for start, end in zip(ts_bounds_idcs[:-1], ts_bounds_idcs[1:]):
        unique_flat, values = compress_hits(flat_idcs[ts_sorter[start: end]],
                                            hits[2, ts_sorter[start: end]])
        coords = np.stack(np.unravel_index(unique_flat, sig_shape), axis=0)
        frames.append(sparse.COO(coords, values, sorted=True, shape=sig_shape))

    sparse_block = sparse.stack(frames)
    for scheme_idx, sl_arr in enumerate(tiling_scheme.slices_array):
        origin = (first_frame_idx,) + tuple(sl_arr[0])
        shape = (len(frames),) + tuple(sl_arr[1])

        tile_slice = Slice(
            origin=origin,
            shape=Shape(shape, sig_dims=tiling_scheme.dataset_shape.sig.dims)
        )

        tile_data = sparse_block[:,
                                 origin[1]: origin[1] + shape[1],
                                 origin[2]: origin[2] + shape[2]]

        yield DataTile(tile_data, tile_slice, scheme_idx)



if __name__ == '__main__':
    data_path = pathlib.Path('/home/mat/Workspace/libertem_dev/data/timepix/experimental_200kv/edge/edge1_000001.tpx3')
    assert data_path.is_file()

    num_part = 16
    headers = parse_file_headers(data_path)
    data_headers = tuple(h for h in headers if h.pkg_type == 0xb)
    data_headers_ar = np.empty((len(data_headers), 6), dtype=np.uint64)

    CHIP_NR_IDX = 0
    SIZE_IDX = 1
    OFFSET_IDX = 2
    SPIDR_IDX = 3
    GLOBAL_IDX = 4
    EPOCH_IDX = 5

    # Should consider rows of values as usually we're indexing the whole row
    print('headers to array')
    for i, h in enumerate(data_headers):
        data_headers_ar[i, :] = np.asarray((h.chip_nr,
                                            h.size,
                                            h.offset,
                                            h.spidr_time,
                                            h.fine_time,
                                            0)).astype(np.uint64)
    print('done')

    # The global spidr timer has a rollover time of ~26.8435 seconds
    # and is a 16bit unsigned integer. This gives a coarse resolution
    # of 4.1 μs for mapping data packets to scan points
    # Each hit also comes with coarse (14-bit) and fine (4-bit) counters
    # which can be used to give more precision, for example 16 + 14 bit
    # gives us 25 ns resolution
    spidr_time_values = data_headers_ar[:, SPIDR_IDX]
    # spidr_time_values = np.asarray(list(h.spidr_time for h in data_headers))

    # Compute epoch number (rollover) of the spidr timer
    # This function can handle out-of-order data packets even if
    # the upstream/example library does not allow for this
    epochs = compute_epoch(spidr_time_values)
    # data_headers = add_epoch_value(data_headers, epochs)

    # Build an ordering of headers by header.fine_time + epoch to allow
    # searching for headers belonging to a particular time span
    # fine_time = np.asarray(list(h.fine_time for h in data_headers)).astype(np.float64)
    # fine_time *= (ROLLOVER_TIME / 2**30)
    # Work with an accumulated offset to account for early rollovers (???)
    accumulated_offset = np.uint64(0)
    for epoch_number, slices in epochs:
        for sl in slices:
            data_headers_ar[sl, GLOBAL_IDX] += accumulated_offset
            data_headers_ar[sl, EPOCH_IDX] = epoch_number
            # fine_time[sl] += accumulated_offset
        # accumulated_offset += fine_time[sl][-1]
        accumulated_offset += data_headers_ar[sl, GLOBAL_IDX].max()
    # Should be mostly ordered already so use stable sort
    sorter = np.argsort(data_headers_ar[:, GLOBAL_IDX], kind='stable')

    # Search for a given span
    # Need to think about how to handle the time
    # between start of a scan and the first data packet
    # Would depend on how the global clock is triggered
    GLOBAL_TIMESTEP = (26.84354 / 2**34)

    start_span = time_to_timestamp(1.84)  # seconds
    end_span = time_to_timestamp(1.94)  # seconds

    start_idx = np.searchsorted(data_headers_ar[:, GLOBAL_IDX], start_span, side='left', sorter=sorter)
    end_idx = np.searchsorted(data_headers_ar[:, GLOBAL_IDX], end_span, side='right', sorter=sorter)
    span_header_idxs = sorter[start_idx: end_idx]
    read_start = span_header_idxs.min()
    read_end = span_header_idxs.max()

    with data_path.open('rb') as fp:
        offset = data_headers_ar[read_start][OFFSET_IDX]
        final_byte = (np.uint64(PACKET_SIZE)
                      + data_headers_ar[read_end][OFFSET_IDX]
                      + data_headers_ar[read_end][SIZE_IDX])
        data_size = (final_byte - offset) // np.uint64(PACKET_SIZE)
        data = np.fromfile(fp,
                           dtype=np.uint64,
                           offset=offset,
                           count=data_size)

    start_time = data_headers_ar[read_start][GLOBAL_IDX]
    end_time = data_headers_ar[read_end][GLOBAL_IDX]

    sig_shape = (516, 516)
    sparse_frame = to_sparse_frame(data, start_time, start_span, end_span, sig_shape)

    n_frames = 16
    frame_boundaries = np.linspace(start_time, end_time, num=n_frames + 1,
                                   endpoint=True, dtype=np.uint64)
    dataset_shape = Shape((100,) + sig_shape, sig_dims=len(sig_shape))
    first_frame_idx = 10
    tileshape = Shape((16, 16, 516), sig_dims=2)
    tiling_scheme = TilingScheme.make_for_shape(tileshape, dataset_shape, "tile")

    for tile in gen_sparse_tiles(data, frame_boundaries, start_time,
                                 tiling_scheme, first_frame_idx):
        print(tile)


    # import matplotlib.pyplot as plt
    # plt.imshow(sparse_frame.todense())
    # plt.show()
