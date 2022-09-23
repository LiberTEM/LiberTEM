import os
import operator
import typing
import concurrent.futures
import itertools
import functools
import numpy as np
from math import ceil, floor

if typing.TYPE_CHECKING:
    from libertem.common.shape import Shape
    from libertem.io.dataset.base.tiling_scheme import TilingScheme


class FortranReader:
    """
    Reads a single-file raw Fortran-ordered array from disk
    in a best-case way for tiled or multi-frame processing
    """
    def __init__(self,
                 path: os.PathLike,
                 shape: 'Shape',
                 dtype: np.dtype,
                 tiling_scheme: 'TilingScheme',
                 file_header: int = 0,
                 min_num_chunks: int = 32):
        self._path = path
        self._dtype = dtype
        self._file_header = file_header
        self._sig_shape = tuple(shape.sig)
        self._num_frames = shape.nav.size
        self._tiling_scheme = tiling_scheme

        self._memmaps = []
        self._chunks = []

        slices = tiling_scheme.slices_array
        assert (slices[:, 1, :-1] == self._sig_shape[:-1]).all(), ('slices not split '
                                                                   'in last dim only')
        tile_widths: list = slices[:, 1, -1].tolist()
        scheme_indices = list(range(len(tile_widths)))

        # Merge or split tiles to get good sized chunks
        # Try to maintain tile boundaries (consider this for merges)
        target_memmap_width = self._sig_shape[-1] // min_num_chunks
        # Split big chunks
        while max(tile_widths) > 1.25 * target_memmap_width:
            max_idx = np.argmax(tile_widths)
            max_val = tile_widths[max_idx]
            new_widths = [floor(max_val / 2), ceil(max_val / 2)]
            new_schemes = [scheme_indices[max_idx], scheme_indices[max_idx]]
            tile_widths = tile_widths[:max_idx] + new_widths + tile_widths[max_idx + 1:]
            scheme_indices = scheme_indices[:max_idx] + new_schemes + scheme_indices[max_idx + 1:]
        # Combine small chunks
        # at this point we are guaranteed to have at least min_num_chunks
        scheme_indices = [{i} for i in scheme_indices]
        # and all elements in scheme_indices are set(integer)
        while min(tile_widths) < target_memmap_width * 0.5:
            min_idx = np.argmin(tile_widths)
            min_val = tile_widths[min_idx]
            min_scheme = scheme_indices[min_idx]
            if min_idx == (len(tile_widths) - 1):
                # Merging final tile
                merge_to = min_idx - 1
            elif min_idx == 0:
                # merging first
                merge_to = 1
            else:
                previous_idx = min_idx - 1
                next_idx = min_idx + 1
                before_si = scheme_indices[previous_idx]
                after_si = scheme_indices[next_idx]
                intersects_before = min_scheme.intersection(before_si)
                intersects_after = min_scheme.intersection(after_si)
                if intersects_before and intersects_after:
                    # ties will break to previous_idx
                    merge_to = min((previous_idx, next_idx), key=lambda x: tile_widths[x])
                elif intersects_before:
                    merge_to = previous_idx
                elif intersects_after:
                    merge_to = next_idx
                else:
                    # intersects neither, merge to the index with the
                    # fewest associated scheme indices
                    # will break ties by merging into previous_idx
                    merge_to = min((previous_idx, next_idx), key=lambda x: len(scheme_indices[x]))
            # Perform the merge and pop current min
            tile_widths[merge_to] += min_val
            scheme_indices[merge_to] = scheme_indices[merge_to].union(scheme_indices[min_idx])
            tile_widths.pop(min_idx)
            scheme_indices.pop(min_idx)

        chunks = tuple((self._num_frames,) + self._sig_shape[:-1] + (width,)
                       for width in tile_widths)
        chunksizes = tuple(self._byte_size(chunkshape)
                           for chunkshape in chunks)
        offsets = tuple(self._file_header + sum(chunksizes[:idx])
                        for idx in range(len(chunksizes)))
        self._chunks = tuple((o, c) for o, c in zip(offsets, chunks))
        boundaries = [0] + np.cumsum(tile_widths).tolist()
        self._sig_last_slices = tuple(slice(a, b) for a, b in zip(boundaries[:-1], boundaries[1:]))
        # self._chunks is ordered and this order sets the order of self._memmaps
        # then the indices in self._scheme_mapping match the memmap indices
        # this could be improved with a different data structure
        self._scheme_mapping = {scheme_index: []
                                for scheme_index in set().union(*scheme_indices)}
        for idx, s_idxs in enumerate(scheme_indices):
            for scheme_index in s_idxs:
                self._scheme_mapping[scheme_index].append(idx)
        self._scheme_slices = tuple(s.get(sig_only=True) for _, s in self._tiling_scheme.slices)

    def create_memmaps(self):
        assert len(self._memmaps) == 0
        for offset, chunkshape in self._chunks:
            self._memmaps.append(np.memmap(self._path,
                                           order='F',
                                           mode='r',
                                           dtype=self._dtype,
                                           offset=offset,
                                           shape=chunkshape))

    def _byte_size(self, shape):
        return np.dtype(self._dtype).itemsize * np.prod(shape, dtype=np.int64)

    def reset_memmaps(self):
        # This is necessary to call occasionally to prevent memory issues
        self._memmaps.clear()
        self.create_memmaps()

    @staticmethod
    def _load_data(memmap, slices, out_buf_f, idx):
        slice_start = 0
        for sl in slices:
            # sl can be a slice or index array
            try:
                length = sl.stop - sl.start
            except AttributeError:
                length = len(sl)
            out_buf_f[slice_start:slice_start + length] = memmap[sl, ...]
            slice_start += length
        return idx

    def generate_tiles(self, *index_or_slice)\
            -> typing.Generator[tuple[tuple[int], int, np.ndarray], None, None]:
        """
        Partition informs the backend of all the frame indices it needs
        Backend optimises the reads to override the depth value where possible
        When we read a block of frames we can yield them one at a time with
        the correct slice information, can yield tiles as they complete
        If the partition only needs a few frames (or single frame for pick)
        then only these few frames are read (heuristically as a block or not)
        Memmaps should be aligned with tile boundaries where these exist
        If one memmap contains multiple tile stacks then the return value from
        the future should indicate this (scheme_idcs?). If one tile stack needs
        multiple memmaps then this could be handled by having a combine future taking
        multiple memmap futures as input, this could then be passed to as_completed
        (this would be the case for yielding stacks of frames)
        If just a few frames in the partition are skipped, then probably better
        to read whole stacks ignoring the small holes then slice/mask the output buffer

        Need to impose a minimum number of tiles per frame
        Tileshape is ideally sig-column-major for data on disk (last dim in numpy)
        Need to impose a minimum depth to read as a block even for process_frame
        """
        index_or_slice = self._splat_iterables(*index_or_slice)
        buffer_length, reads = self._plan_reads(self._tiling_scheme.depth, *index_or_slice)
        out_buffer_f = np.empty((buffer_length,) + self._sig_shape, dtype=self._dtype, order='F')

        with concurrent.futures.ThreadPoolExecutor() as p:
            for mmap_nav_slices, buffer_nav_slice, buffer_unpacks in reads:
                combined_slices = self._slice_combine_array(*mmap_nav_slices)
                raw_futures = []
                for raw_idx, (memmap, sig_slice) in enumerate(zip(self._memmaps,
                                                                  self._sig_last_slices)):
                    raw_futures.append(
                            p.submit(
                                self._load_data,
                                memmap,
                                combined_slices,
                                out_buffer_f[buffer_nav_slice, ..., sig_slice],
                                raw_idx
                            )
                        )
                combined_futures = self._combine_raw_futures(raw_futures, p)
                for complete in concurrent.futures.as_completed(combined_futures.keys()):
                    scheme_index = combined_futures[complete]
                    for (slice_in_buffer, idcs_in_flat_nav) in buffer_unpacks:
                        tile_slice = ((slice_in_buffer,)
                                      + (Ellipsis,)
                                      + self._scheme_slices[scheme_index])
                        tile = out_buffer_f[tile_slice]
                        yield idcs_in_flat_nav, scheme_index, tile

    def _combine_raw_futures(self, futures, pool):
        combined_futures = {}
        for scheme_index, raw_idxs in self._scheme_mapping.items():
            if len(raw_idxs) == 1:
                combined_futures[futures[raw_idxs[0]]] = scheme_index
                continue
            combined = pool.submit(self._combine, *tuple(futures[i] for i in raw_idxs))
            combined_futures[combined] = scheme_index
        return combined_futures

    @staticmethod
    def _combine(*futures):
        return tuple(f.result() for f in futures)

    @classmethod
    def _plan_reads(cls, ts_depth, *index_or_slice, min_read_depth: int = 64):
        """
        Performs a similar function to get_read_ranges but works on slices instead
        """
        # Must yield tiles/frames with ts_depth at the end to be consistent
        # Combine any sequential indices or slices into contiguous slices
        # will cast any int in index_or_slice to slice(int, int + 1) (or combine it)
        index_or_slice = cls._combine_sequential(*index_or_slice)
        # Compute number of frames to read
        slice_lengths = tuple(cls._length_slice(s) for s in index_or_slice)
        to_read = sum(slice_lengths)

        if to_read == ts_depth:
            # Best to do in a single pass
            # This should cover pick_frame, i.e. no wasted buffer/reading
            # Should probably also cover process_partition
            # NOTE check how ROI + process_partition + depth works
            buffer_length = to_read
        else:
            # With process_frame we will have ts_depth == 1,
            # we need to read multiple frames together for performance,
            # hence the multiply/max()
            # buffer_length will also be a multiple of ts_depth so we can
            # hold 1 or more complete tile/frame stacks in the buffer
            buffer_length = (min_read_depth // ts_depth) * ts_depth
            buffer_length = max(ts_depth, buffer_length)

        # split splices down to len(sl) == buffer_length at maximum
        # must functools.reduce because self._split_slice returns tuples of slices
        index_or_slice = functools.reduce(operator.add,
                                          tuple(cls._split_slice(sl, buffer_length)
                                                for sl in index_or_slice))

        # Collect slice info in one pass so we can do recombination
        slice_info = []
        for element in index_or_slice:
            read_length = (element.stop - element.start)
            first_idx = element.start
            last_idx = element.stop - 1
            try:
                # distance should always be > 1 as adjacent slices are combined
                distance = slice_info[-1][-1] - last_idx
            except IndexError:
                distance = 0
            slice_info.append((read_length, distance, first_idx, last_idx))

        # During slice combination we're constrained that even in the ROI case
        # we need to emit frames in the right order, as the tile_slice attribute
        # of DataTile only contains a flat nav origin, not the actual frame
        # indices in the tile, and the frame indices present in the tile are
        # implicit from the tiling scheme and ROI. An optimisation here would be
        # to collect all 'isolated' frames in the ROI in one pass, possibly combining
        # with a short slice at the end of the partition, then emit tiles from this data.
        # Instead we must emit tiles in an ordered way by reading blocks of slices
        # and indices matching the roi into the appropriately sized buffer
        # A further optimisation is to read contiguous blocks even if one or two
        # frames in the block are roi == false, and just discard this data after
        idx = 0
        combinations = []
        _combining = []
        num_slices = len(slice_info)
        while idx < num_slices:
            read_length, distance, first_idx, last_idx = slice_info[idx]
            # start of loop or new potential combination
            # optimistically add it and move on
            if not _combining:
                _combining.append(idx)
                idx += 1
                continue
            # Found a full-length read, push current working
            # combination onto stack and then push this one
            # read_length should never be > buffer_length
            if read_length == buffer_length:
                combinations.append(tuple(_combining))
                _combining = []
                combinations.append((idx,))
                idx += 1
                continue
            # current length of the data from _combining
            combined_length = sum(slice_info[_idx][0] for _idx in _combining)
            if combined_length + read_length > buffer_length:
                # new combination will be longer than buffer, end current and start new
                combinations.append(tuple(_combining))
                _combining = [idx]
                idx += 1
                continue
            # can combine idx into _combining without exceeding buffer
            _combining.append(idx)
            idx += 1

        if _combining:
            combinations.append(tuple(_combining))
        _combining.clear()

        reads = []
        # returns reads:
        #     list of (tuple(slice,...)_for_memmap), slice_for_buffer_alloc, list(unpacks))
        #     unpacks == (slice_in_buffer, tuple(int, ...)_flat_frame_idcs)
        #     union(slice_in_buffer) does not necessarily fully cover
        #     all read data, to allow ROI gaps with contig reads
        for combination in combinations:
            mmap_slices = tuple(index_or_slice[i] for i in combination)
            # If we are just handling a single contig combination can optimise here by not
            # returning a frame index array but rather just a single nav slice object
            frame_idcs = itertools.chain(*tuple(range(slice_info[i][2], slice_info[i][3] + 1)
                                                for i in combination))
            frame_idcs = tuple(frame_idcs)
            read_length = sum(slice_info[i][0] for i in combination)
            buffer_allocation = slice(0, read_length)
            # Even though we ignore that mmap_slices may contain multiple read chunks
            # this should be OK because the DataTile system is blind to the ROI
            unpacks = [(slice(lb, ub), frame_idcs[lb:ub])
                       for lb, ub
                       in cls._gen_slices_for_depth(read_length, ts_depth)]
            reads.append((mmap_slices, buffer_allocation, unpacks))

        return buffer_length, reads

    @staticmethod
    def _gen_slices_for_depth(length, depth):
        assert length
        boundaries = [*range(0, length, depth)]
        if boundaries[-1] != length:
            boundaries.append(length)
        yield from zip(boundaries[:-1], boundaries[1:])

    @staticmethod
    def _length_slice(sl_or_int):
        try:
            return sl_or_int.stop - sl_or_int.start
        except AttributeError:
            return 1

    @classmethod
    def _split_slice(cls, sl, target_length):
        sl_length = cls._length_slice(sl)
        if sl_length <= target_length:
            return (sl,)
        start = sl.start
        return tuple(slice(start + lb, start + ub)
                     for lb, ub
                     in cls._gen_slices_for_depth(sl_length, target_length))

    @staticmethod
    def _splat_iterables(*index_or_slice):
        slices = []
        for sl in index_or_slice:
            try:
                slices.extend(tuple(sl))
            except TypeError:
                slices.append(sl)
        return slices

    @staticmethod
    def _combine_sequential(*index_or_slice):
        slices = []
        for sl in index_or_slice:
            if isinstance(sl, (int, np.integer)):
                sl = slice(sl, sl + 1)
            try:
                if slices[-1].stop == sl.start:
                    slices[-1] = slice(slices[-1].start, sl.stop)
                else:
                    slices.append(sl)
            except IndexError:
                slices.append(sl)
        return tuple(slices)

    @staticmethod
    def _slice_combine_array(*slices, threshold_combine: int = 8):
        _slices = []
        for sl in slices:
            if (sl.stop - sl.start) <= threshold_combine:
                # convert to index array
                slice_gen = range(sl.start, sl.stop)
                if not _slices or (not isinstance(_slices[-1], list)):
                    # beginning of loop or starting new index array
                    _slices.append([*slice_gen])
                    continue
                else:
                    # extend current index array
                    _slices[-1].extend(slice_gen)
            else:
                # slice object is too long for index array, maintain as slice
                _slices.append(sl)
        assert _slices, 'No slices to read'
        return _slices
