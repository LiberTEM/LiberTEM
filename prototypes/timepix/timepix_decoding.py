from __future__ import annotations
import os
import numpy as np
import numba
from numba.typed import List
import sparse
import typing

if typing.TYPE_CHECKING:
    from libertem.executor.base import BaseJobExecutor

ROLLOVER_TIME = 26.8435456
READ_BLOCKSIZE = 8192  # 1K events max
SEEK_AMOUNT = READ_BLOCKSIZE * 8  # 64K or 8K max-events


@numba.njit
def parse_hit_data(hit_data: np.ndarray, chip_nr: int,
                   out_buffer: np.ndarray, cross_offset: int = 2):
    """
    Parse a 1D array of np.uint64 event packets into the (6, hit_data.size) np.uint16 out_buffer
    """
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
def parse_time_data_inplace(hit_data: np.ndarray, out_buffer: np.ndarray):
    # spidr
    out_buffer[0] = np.bitwise_and(hit_data, np.uint64(0xffff))
    # coarse_toa
    out_buffer[1] = np.bitwise_and(hit_data >> np.uint64(16 + 14), np.uint64(0x3fff))
    # fine_toa
    out_buffer[2] = np.bitwise_and(hit_data >> np.uint64(16), np.uint64(0xf))


@numba.njit
def convert_position_data(col: np.ndarray,
                          super_pix: np.ndarray,
                          pix: np.ndarray,
                          chip_nr: int,
                          out_x: np.ndarray,
                          out_y: np.ndarray,
                          cross_offset: int = 2):
    # the x/y position values are for one chip so all np.uint8 (256x256 chip)
    # potential overflow issues here ?? probably impossible given the chip design
    out_x[:] = col + (pix >> np.uint16(2))
    out_y[:] = super_pix + (pix & np.uint16(0x3))
    combine_chips(out_x, out_y, chip_nr, cross_offset=cross_offset)


@numba.njit
def combine_chips(x: np.ndarray, y: np.ndarray, chip_nr: int, cross_offset: int = 2):
    """
    correction is applied inplace on x and y

    Chip are orientated like this (unspecified x/y orientation)
    2 1
    3 0
    """
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
def get_global_time_no_roll(spidr_time, coarse_toa, fine_toa):
    toa = (coarse_toa.astype(np.uint64) << np.uint64(4)) - fine_toa
    global_time = (spidr_time.astype(np.uint64) << np.uint64(18)) + toa
    return global_time


@numba.njit
def is_header(value: np.uint64) -> bool:
    """
    title_ints = tuple(np.uint64(ord(x)) << np.uint64(i * 8) for i, x in enumerate('TPX3'))
    title_ints[3] | title_ints[2] | title_ints[1] | title_ints[0] == np.uint64(0x33585054)
    """
    return (value & np.uint64(0xffffffff)) == np.uint64(0x33585054)


@numba.njit
def is_hit(value: np.uint64) -> bool:
    """Bitmask with 0xb on bits 64:60"""
    return (value & np.uint64(0xf000000000000000)) == np.uint64(0xb000000000000000)


@numba.njit
def find_hits_header(data: np.ndarray):
    """Return index in data of first hits header, else -1 if no hits header found"""
    for idx in range(len(data)):
        value = data[idx]
        if is_header(value):
            next_value = data[idx + 1]
            if is_hit(next_value):
                return idx
    return -1


@numba.njit
def decode_headers(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode one or more header values, return arrays of np.uint16 for chip number and payload size

    Does not verify that the values in data actually represent headers

    return arrays are both np.uint16
    """
    components = np.asarray(data).reshape(-1).view(np.uint8).reshape(-1, 8)
    chip_nr = components[:, 4].astype(np.uint16)
    size_lsb = components[:, -2].astype(np.uint16)
    size_msb = components[:, -1].astype(np.uint16)
    size = ((np.uint16(0xff) & size_msb) << np.uint16(8)) | (np.uint16(0xff) & size_lsb)
    return chip_nr, size // np.uint16(8)


@numba.njit
def decode_block(data: np.ndarray, out_buffer: np.ndarray) -> \
                                        tuple[int, tuple[int, int], list[int]]:
    """
    Decode a block of data into events from the first header until the end of the last full payload
    Ignore all non-hit headers/payloads
    Also return tuple[int, int] which is the slice in data from which events were decoded
    Also return indices of all hit events headers in data

    out_buffer is np.ndarray with shape (6, data.size) and
    dtype np.uint16 (as defined by parse_hit_data)

    If no headers are in data, consider all of data to be 'leading', return (0, (data.size, 0), [])
        i.e. slicing data[:data.size] or slicing data[0:] will give the full block
        to append/preprend onto subsequent calls to decode_block
    """
    out_pointer = 0
    hit_headers = List()
    first_header_idx = 0
    data_size = data.size
    while first_header_idx < data_size and not is_header(data[first_header_idx]):
        first_header_idx += 1
    # No headers in data, return empty events array
    if first_header_idx == data_size:
        return out_pointer, (data_size, 0), hit_headers
    # Decode as many payloads as we can
    header_idx = first_header_idx
    while header_idx < data_size - 1:
        # loop until size-1 as a header on the last index cannot be decoded
        chip_nr, payload_size = decode_headers(data[header_idx])
        chip_nr = chip_nr.item()
        payload_size = payload_size.item()
        if (header_idx + payload_size) >= data_size:
            # end of full payloads
            break
        if not is_hit(data[header_idx + 1]):
            # not a hits header, skip over the payload
            header_idx = header_idx + payload_size + 1
            continue
        hit_headers.append(header_idx)
        hit_headers.append(out_pointer)
        # Decoding
        payload = data[header_idx + 1: header_idx + 1 + payload_size]
        parse_hit_data(payload, chip_nr, out_buffer[:, out_pointer: out_pointer + payload_size])
        header_idx += (1 + payload_size)
        out_pointer += payload_size

    return out_pointer, (first_header_idx, header_idx), hit_headers


@numba.njit
def densify_valid(flat_idcs, values, valid, out_buffer):
    for idx in range(len(flat_idcs)):
        if not valid[idx]:
            continue
        flat_idx = flat_idcs[idx]
        out_buffer[flat_idx] += values[idx]


def make_dense_valid(flat, values, valid, shape) -> np.ndarray:
    dense_frame = np.zeros(np.prod(shape), dtype=np.uint64)
    densify_valid(flat, values, valid, dense_frame)
    return dense_frame.reshape(shape)


def full_timestamp(value_s: float | np.ndarray):
    cycles_per_second = 640_000_000  # This is (2**34 / 26.8435456) which is the rollover time
    value_i = value_s * cycles_per_second
    if isinstance(value_s, np.ndarray):
        return value_i.astype(int)
    else:
        return int(value_i)


def find_timestamp_near(filepath: str, offset: int, max_offset: int) -> tuple[np.uint64, int]:
    """
    Return (first_hit_timestamp, file_offset) after a header in the file,
    search starting from offset
    Sequentially read READ_BLOCKSIZE bytes until a header + timestamp is found,
    else return (None, None) if we reach max_offset or end of file
    """
    offset = int(offset)
    assert offset % 8 == 0
    read_count = READ_BLOCKSIZE // 8
    out_buffer = np.empty((read_count,), dtype=np.uint64)
    header_idx = -1
    while header_idx < 0:
        try:
            out_buffer[:] = np.fromfile(filepath, count=read_count, dtype=np.uint64, offset=offset)
        except ValueError:
            # Unable to read full chunk of data, must be near end of file, i.e. break
            return None, None
        offset += (read_count * 8)
        header_idx = find_hits_header(out_buffer)
        if offset >= max_offset:
            return None, None
    time_buffer = np.zeros((3, 1), dtype=np.uint16)
    try:
        parse_time_data_inplace(out_buffer[header_idx + 1], time_buffer)
    except IndexError:
        return None, None
    global_time = get_global_time_no_roll(time_buffer[0], time_buffer[1], time_buffer[2])
    return global_time.item(), offset - ((out_buffer.size - header_idx) * 8)


def read_file_structure(filepath, n_samples=512, start_offset=0,
                        max_offset=None, executor: BaseJobExecutor = None) -> np.ndarray:
    """
    Sample the file at filepath n_samples times, and return
    an array (timestamp, file_offset), sorted by timestamp increasing
    where a hits header could be found in the sample

    If executor is supplied use parallel map for reading

    length of return array is <= n_samples
    """
    if max_offset is None:
        max_offset = os.stat(filepath).st_size
    offsets = np.linspace(start_offset, max_offset, num=n_samples, endpoint=False).astype(int)
    offsets -= (offsets % 8)
    if executor is None:
        structure = []
        for offset, next_offset in zip(offsets[:-1], offsets[1:]):
            global_time, header_offset = find_timestamp_near(filepath, offset, next_offset)
            structure.append((global_time, header_offset))
    else:
        def _find_timestamp_near(_offsets):
            offset, next_offset = _offsets
            return find_timestamp_near(filepath, offset, next_offset)
        structure = executor.map(_find_timestamp_near, [*zip(offsets[:-1], offsets[1:])])

    structure = [s for s in structure if s[0] is not None]
    structure = np.asarray(structure).reshape(-1, 2)
    # FIXME correct for epoch rollover here!!!
    sorter = np.argsort(structure[:, 0])
    return structure[sorter, :]


def estimate_true_offset(timestamps: np.ndarray, offsets: np.ndarray,
                         target_ts: np.uint64, ts_idx: int) -> int:
    if ts_idx == 0:
        # Before first header we saw, start at beginning of file
        return int(0)
    elif ts_idx == timestamps.size:
        # After last header we saw, start from last header
        return offsets[-1]
    else:
        offset_span_start = offsets[ts_idx - 1]
        offset_span = offsets[ts_idx] - offset_span_start
        ts_span = timestamps[ts_idx] - timestamps[ts_idx - 1]
        fraction_ts = ((target_ts - timestamps[ts_idx - 1]) / ts_span) * offset_span
        return int(offset_span_start + fraction_ts)


def offsets_for_timestamps(structure, start_timestamp, end_timestamp, max_ooo):
    _start_timestamp = start_timestamp - max_ooo
    _end_timestamp = end_timestamp + max_ooo

    start_idx, end_idx = np.searchsorted(structure[:, 0], (_start_timestamp,
                                                           _end_timestamp), side='left')
    start_offset = estimate_true_offset(structure[:, 0], structure[:, 1],
                                        _start_timestamp, start_idx)
    end_offset = estimate_true_offset(structure[:, 0], structure[:, 1],
                                      _end_timestamp, end_idx)

    # round indwards to nearest uint64
    start_offset += (8 - (start_offset % 8))
    end_offset -= end_offset % 8
    return start_offset, end_offset


def extract_between_offsets(fp, start_offset, end_offset, prepend_data=None, append_data=None):
    """
    Extract the data between (start_offset, end_offset) from the file
    and concatenate it with any prepend or append data from a previous,
    incomplete decoding.
    Then decode the data and return events, timestamps, and any un-decoded
    data from before /  after the returned events.
    """
    fp.seek(start_offset)

    read_count = (end_offset - start_offset) // 8
    data = np.fromfile(fp, count=read_count, dtype=np.uint64, offset=0)

    if (prepend_data is not None) and (append_data is not None):
        data = np.concatenate((prepend_data, data, append_data), axis=0)
    elif prepend_data is not None:
        data = np.concatenate((prepend_data, data), axis=0)
    elif append_data is not None:
        data = np.concatenate((data, append_data), axis=0)

    events_buffer = np.empty((6, data.size), dtype=np.uint16)
    out_pointer, events_slice, hit_headers = decode_block(data, events_buffer)
    events_buffer = events_buffer[:, :out_pointer]
    global_times = get_global_time_no_roll(events_buffer[3], events_buffer[4], events_buffer[5])

    head = data[:events_slice[0]]
    if head.size == 0:
        head = None
    tail = data[events_slice[1]:]
    if tail.size == 0:
        tail = None

    return events_buffer, global_times, (head, tail)


def extract_between_timestamps(filepath, structure, start_timestamp, end_timestamp, max_ooo=6400):
    """
    structure is (timestamp, file_offset), epoch-corrected
    start/end_timestamp are same base as the timestamp in structure, epoch-corrected

    timestamps in structure are ordered increasing, ties broken by offset increasing

    max_ooo is the max timestamp jitter we will allow, i.e. if the upper target timestamp
    is x, as soon as we see x + max_ooo we will stop searching for OOO values <= x
    """
    filesize = os.stat(filepath).st_size
    start_offset, end_offset = offsets_for_timestamps(structure,
                                                      start_timestamp,
                                                      end_timestamp,
                                                      max_ooo)

    # Main events block
    with open(filepath, 'rb') as fp:
        events, global_times, (main_head, main_tail) = extract_between_offsets(fp,
                                                                               start_offset,
                                                                               end_offset)
        events_collector = []
        times_collector = []

        if events.size:
            events_collector.append(events)
            times_collector.append(global_times)

        # Seek backwards if necessary
        append = main_head
        while not events_collector or times_collector[0].min() > start_timestamp - max_ooo:
            start_offset = max(0, start_offset - SEEK_AMOUNT)
            if start_offset == 0:
                # FIXME
                break
            (_events,
            _global_times,
            (_head, _tail)) = extract_between_offsets(fp,
                                                      start_offset,
                                                      start_offset + SEEK_AMOUNT,
                                                      append_data=append)
            assert _tail is None
            append = _head
            if _events.size:
                events_collector.insert(0, _events)
                times_collector.insert(0, _global_times)

        # Seek forwards if necessary
        prepend = main_tail
        while times_collector[-1].max() < end_timestamp + max_ooo:
            end_offset = min(filesize, end_offset + SEEK_AMOUNT)
            if end_offset == filesize:
                # FIXME
                break
            (_events,
            _global_times,
            (_head, _tail)) = extract_between_offsets(fp,
                                                      end_offset - SEEK_AMOUNT,
                                                      end_offset,
                                                      prepend_data=prepend)
            assert _head is None
            prepend = _tail
            if _events.size:
                events_collector.append(_events)
                times_collector.append(_global_times)

    # Concatenate all the events we read from the file
    return np.concatenate(events_collector, axis=1), np.concatenate(times_collector)


def are_spans_valid(spans: np.ndarray) -> bool:
    """
    Checks that spans are sorted, all have a width and don't overlap
    """
    nonzero = (spans[:, 1] > spans[:, 0]).all()
    disjoint = ((spans[1:, 0] - spans[:-1, 1]) >= 0).all()
    return nonzero and disjoint


def span_idx_for_ts(spans: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Assigns a span index to each timestamp
    timestamp values which are not in a span
    are valid == 0, else valid == 1

    Assumes spans sorted by start time
    """
    idcs = np.searchsorted(spans.ravel(), timestamps)
    extended_shape = (spans.shape[0] + 1, spans.shape[1])
    # Must extend shape to allow idcs == (spans.shape[0] + 1)
    # to be unravelled, but always with a validity of 0
    span_id, valid = np.unravel_index(idcs, extended_shape)
    return span_id, valid


def split_contig_spans(spans: np.ndarray, structure: np.ndarray,
                       max_ooo: int, split_threshold: int = 2**23) -> list[np.ndarray]:
    """
    Splits the timespans in spans into separate blocks
    if the estimated distance between the blocks exceeds
    split_threshold bytes in the file, by default 8 MB,
    to avoid decoding events which will later be discarded
    """
    if spans.shape[0] == 1:
        return [spans]

    span_offsets = []
    for start_timestamp, end_timestamp in spans:
        span_offsets.append(offsets_for_timestamps(structure, start_timestamp,
                                                   end_timestamp, max_ooo))
    span_offsets = np.asarray(span_offsets).reshape(-1, 2)

    inter_span_offset = np.abs(span_offsets[:-1, 1] - span_offsets[1:, 0])
    splits = np.argwhere(inter_span_offset > split_threshold) + 1
    boundaries = [0] + splits.squeeze(axis=1).tolist() + [spans.shape[0]]
    return [spans[start: end] for start, end in zip(boundaries[:-1], boundaries[1:])]


def spans_as_frames(filepath, structure: np.ndarray, spans: np.ndarray,
                    sig_shape: tuple[int, int], max_ooo=6400, as_dense=False) -> sparse.COO:
    assert are_spans_valid(spans)
    subspans = split_contig_spans(spans, structure, max_ooo)
    events, times = [], []
    for subspan in subspans:
        start_timestamp = subspan.min()
        end_timestamp = subspan.max()
        _events, _times = extract_between_timestamps(filepath, structure,
                                                   start_timestamp, end_timestamp,
                                                   max_ooo=max_ooo)
        events.append(_events)
        times.append(_times)
    events = np.concatenate(events, axis=1)
    times = np.concatenate(times)

    out_shape = (spans.shape[0],) + sig_shape
    span_id, ts_valid = span_idx_for_ts(spans, times)
    extended_shape = (spans.shape[0] + 1,) + sig_shape
    flat_idcs = np.ravel_multi_index((span_id, events[1], events[0]), extended_shape)
    dense = make_dense_valid(flat_idcs, events[2], ts_valid, out_shape)
    if as_dense:
        return dense
    coords = np.argwhere(dense)
    values = dense[coords[:, 0], coords[:, 1], coords[:, 2]]
    return sparse.COO(coords.T, values, sorted=True,
                      has_duplicates=False, shape=out_shape)


def spans_as_tiles(filepath, structure, spans, tiling_scheme, max_ooo=6400) -> list[sparse.COO]:
    # Not yet implemented but would be possible with mostly the same code
    ...
