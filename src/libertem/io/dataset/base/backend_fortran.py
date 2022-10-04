import os
import operator
import typing
from typing import Tuple, Union, Generator, Iterable, List, Set, Dict, Any, Literal
import concurrent.futures
import itertools
import functools
import numpy as np
from math import ceil, floor

if typing.TYPE_CHECKING:
    from libertem.common.shape import Shape
    from libertem.io.dataset.base.tiling_scheme import TilingScheme
    from numpy.typing import DTypeLike
    ChunkMapT = Dict[Union[int, Tuple[int]], Set[int]]


class FortranReader:
    """
    Reads a single-file raw array structured on disk as

            (flat_sig, flat_nav)

    which corresponds to a Fortran-like ordering
    Splits the file into chunks spanning flat_sig
    in a way which corresponds to the desired tiling
    scheme and parallellises the reads to improve performance

    The parameter sig_order determines if the data read should
    be reshaped from flat_sig using 'C' or 'F' ordering. In the case
    of DM4 files the data is (flat_sig, flat_nav) but the signal and nav
    dimensions were individually unrolled using C-order
    """
    MIN_MEMMAP_SIZE = 512 * 2 ** 20  # 512 MB
    MAX_NUM_MEMMAP: int = 16
    BUFFER_SIZE: int = 128 * 2 ** 20
    THRESHOLD_COMBINE: int = 8  # small slices to convert to index arrays

    def __init__(self,
                 path: os.PathLike,
                 shape: 'Shape',
                 dtype: np.dtype,
                 tiling_scheme: 'TilingScheme',
                 sig_order='F',
                 file_header: int = 0):
        self._path = path
        self._dtype = dtype
        self._file_header = file_header
        self._shape = shape
        self._sig_size = shape.sig.size
        self._sig_order = sig_order
        self._tiling_scheme = tiling_scheme

        self._memmaps = []

        self.verify_tiling(tiling_scheme, shape, sig_order)
        chunks, self._chunk_slices, chunk_scheme_indices = self.choose_chunks(tiling_scheme,
                                                                              shape,
                                                                              dtype)
        chunksizes = tuple(self._byte_size(chunkshape)
                           for chunkshape in chunks)
        offsets = tuple(self._file_header + sum(chunksizes[:idx])
                        for idx in range(len(chunksizes)))
        self._chunks = tuple((o, c) for o, c in zip(offsets, chunks))
        independent_chunks, self._chunk_combinations = self.build_chunk_map(chunk_scheme_indices)
        self._chunk_map = {**independent_chunks, **self._chunk_combinations}

    @staticmethod
    def verify_tiling(tiling_scheme: 'TilingScheme', shape: 'Shape', sig_order: Literal['F', 'C']):
        slices = tiling_scheme.slices_array
        # sig slices are flattened according to sig_order
        # if a sig slice does not span all dimensions except
        # the first/last (for C-/F-) then the slices can't be ravelled
        if sig_order == 'C':
            assert (slices[:, 1, 1:] == shape.sig[1:]).all(), ('slices not split '
                                                               'in first dim only')
        elif sig_order == 'F':
            assert (slices[:, 1, :-1] == shape.sig[:-1]).all(), ('slices not split '
                                                                 'in last dim only')
        else:
            raise ValueError(f'sig_order {sig_order} not recognized')

    @staticmethod
    def unpack_scheme(tiling_scheme: 'TilingScheme'):
        slices = tiling_scheme.slices_array
        # Get sizes of each tile in number of elements
        tile_widths: list = np.prod(slices[:, 1, :], axis=-1, dtype=np.int64).tolist()
        boundaries = [0] + np.cumsum(tile_widths).tolist()
        # slice into flat_sig for each tile in tiling scheme
        tile_slices = tuple(slice(a, b) for a, b
                            in zip(boundaries[:-1], boundaries[1:]))
        scheme_indices = list(range(len(tile_widths)))
        tile_shapes = tuple(tiling_scheme[scheme_index].shape.to_tuple()
                            for scheme_index in scheme_indices)
        return tile_slices, tile_shapes

    @classmethod
    def choose_chunks(cls, tiling_scheme: 'TilingScheme', shape: 'Shape', dtype: 'DTypeLike'):
        # NOTE could refactor and use tiling_scheme.dataset_shape

        # Merge or split tiles to get good sized chunks
        # Try to maintain tile boundaries (consider this for merges)
        nav_size = shape.nav.size
        sig_size = shape.sig.size
        slices = tiling_scheme.slices_array
        # Get sizes of each tile in number of elements
        tile_widths: list = np.prod(slices[:, 1, :], axis=-1, dtype=np.int64).tolist()
        scheme_indices = list(range(len(tile_widths)))

        nav_bytesize = np.dtype(dtype).itemsize * nav_size
        min_width_bytesize = cls.MIN_MEMMAP_SIZE // nav_bytesize
        min_width_number = sig_size // cls.MAX_NUM_MEMMAP
        data_size = sig_size * nav_size
        min_width = min(data_size, max(min_width_bytesize, min_width_number))
        max_width = min(data_size, 2 * min_width)
        # Split big chunks
        while max(tile_widths) > max_width:
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
        while len(tile_widths) > 1 and min(tile_widths) < min_width:
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

        tile_widths: Tuple[int]
        chunks = tuple((width,) + (nav_size,)
                       for width in tile_widths)
        boundaries = [0] + np.cumsum(tile_widths).tolist()
        # self._chunk_slices are the flat-sig slice objects for each mmap chunk
        # used to assign each loaded chunk into the output buffer
        chunk_slices = tuple(slice(a, b) for a, b in zip(boundaries[:-1], boundaries[1:]))
        return chunks, chunk_slices, scheme_indices

    @staticmethod
    def build_chunk_map(chunk_scheme_indices: List[Set[int]]) -> Tuple['ChunkMapT', 'ChunkMapT']:
        """
        Build the mapping between a chunk of the file and
        the scheme indices it can provide. Needed as depending on
        layout / tile size, a chunk can provide any of:

            - only part of a single tile stack
            - a mixture of complete and incomplete tile stacks

        and therefore might have to be combined with other chunks
        before a tile can be marked as ready

        Returns both the dictonary mapping {chunk_idx: {scheme_indices}}
        for individual chunks, and {tuple(chunk_idx): {scheme_indices}}
        for necessary combinations of chunks and the scheme_indices they provide
        """
        # self._chunks are the offsets and shapes of each memmap (chunk_of_sig, whole_nav)
        # scheme_indices is list(set(int)) of which tile idxs are generated
        # by each memmap chunk,
        # Here pre-compute which chunks need combining and which tiles
        # come from a single chunk. The generator will yield tiles
        # as soon as they are ready even if a combination future is still running
        max_scheme_index = max(max(c) for c in chunk_scheme_indices)
        chunk_tile_map = np.zeros((len(chunk_scheme_indices), max_scheme_index + 1), dtype=bool)
        for chunk_idx, tile_idcs in enumerate(chunk_scheme_indices):
            chunk_tile_map[chunk_idx, list(tile_idcs)] = True
        chunks_per_tile = chunk_tile_map.sum(axis=0)
        # Create lists of tiles provided uniquely by an individual chunk
        independent_tiles = set(np.flatnonzero(chunks_per_tile == 1))
        scheme_map_independent = {ci: s.intersection(independent_tiles)
                                  for ci, s in enumerate(chunk_scheme_indices)}
        # Create map of chunks which must be combined to yield a tile
        scheme_map_combination = {}
        combination_tiles = np.flatnonzero(chunks_per_tile > 1)
        for tile_idx in combination_tiles:
            chunks_for_tile: Tuple[int] = tuple(np.flatnonzero(chunk_tile_map[:, tile_idx]))
            try:
                scheme_map_combination[chunks_for_tile].append(tile_idx)
            except KeyError:
                scheme_map_combination[chunks_for_tile] = [tile_idx]
        scheme_map_combination = {k: set(v) for k, v
                                  in scheme_map_combination.items()}
        return scheme_map_independent, scheme_map_combination

    def create_memmaps(self):
        # Always memmap in C-order (slice_of_flat_sig), whole_flat_nav)
        # Access is fast to a full set of values for a single sig pixel
        # which reflects how the data is laid out
        # The memmap will be used for slicing into whole_flat_nav,
        # however this is unavoidable!
        assert len(self._memmaps) == 0
        for offset, chunkshape in self._chunks:
            self._memmaps.append(np.memmap(self._path,
                                           mode='r',
                                           dtype=self._dtype,
                                           offset=offset,
                                           shape=chunkshape))

    def _byte_size(self, shape: Iterable) -> int:
        return np.dtype(self._dtype).itemsize * np.prod(shape, dtype=np.int64)

    def reset_memmaps(self):
        # This may be necessary to call occasionally to prevent memory issues
        self._memmaps.clear()
        self.create_memmaps()

    @staticmethod
    def _load_data(memmap: np.ndarray,
                   slices: Tuple[Iterable, slice],
                   out_buf: np.ndarray,
                   idx: int) -> int:
        """
        Read slices/values from memmap into out_buf sequentially in the last dimension

        Return idx to allow tracking which memmap chunk(s) just completed
        """
        slice_start = 0
        for sl in slices:
            # sl can be a slice or index array
            try:
                length = sl.stop - sl.start
            except AttributeError:
                length = len(sl)
            out_buf[..., slice_start:slice_start + length] = memmap[..., sl]
            slice_start += length
        return idx

    def generate_tiles(self, *index_or_slice: Union[int, slice, Iterable])\
            -> Generator[tuple[tuple[int], int, np.ndarray], None, None]:
        """
        Yields tiles from dataset according to tiling_scheme (potentially unordered)
        *index_or_slice can be any/mix of [int, slice, iterable] (strictly ordered)
        of flat_nav, full (no-ROI) frame indices to yield to the caller (the partition)

        The backend optimises the reads to override the tiling_scheme depth
        but always yields tile stacks matching the tiling_scheme expectation
        If the partition only needs a few frames then only these few frames are read

        The caller should be aware of the ordering of the data on disk
        The flat indices provided are assumed to correspond to the order
        on disk, i.e. if you want the frames to be read in Fortran order
        then provide *index_or_slice for a Fortran unrolling of the nav dims
        """
        tile_slices, tile_shapes = self.unpack_scheme(self._tiling_scheme)
        index_or_slice = tuple(self._splat_iterables(*index_or_slice))
        ideal_depth = 1 + self.BUFFER_SIZE // (self._sig_size * np.dtype(self._dtype).itemsize)
        buffer_length, reads = self._plan_reads(ideal_depth,
                                                self._tiling_scheme.depth,
                                                *index_or_slice)
        out_buffer = np.empty((self._sig_size, buffer_length), dtype=self._dtype)

        with concurrent.futures.ThreadPoolExecutor() as p:
            for mmap_nav_slices, buffer_nav_slice, buffer_unpacks in reads:
                combined_slices = self._slice_combine_array(*mmap_nav_slices)
                raw_futures = []
                for raw_idx, (memmap, buffer_sig_slice) in enumerate(zip(self._memmaps,
                                                                         self._chunk_slices)):
                    raw_futures.append(
                            p.submit(
                                self._load_data,
                                memmap,
                                combined_slices,
                                out_buffer[buffer_sig_slice, buffer_nav_slice],
                                raw_idx
                            )
                        )
                combined_futures = self._combine_raw_futures(raw_futures, p)
                tiles_completed = set()
                for complete in concurrent.futures.as_completed(raw_futures + combined_futures):
                    chunks_completed = complete.result()
                    tiles_ready = self._chunk_map[chunks_completed]
                    tiles_to_yield = tiles_ready.difference(tiles_completed)
                    if not tiles_to_yield:
                        continue
                    tiles_completed = tiles_completed.union(tiles_to_yield)
                    for (buffer_nav_slice, idcs_in_flat_nav) in buffer_unpacks:
                        for scheme_index in tiles_to_yield:
                            # flat_tile has dims (flat_sig, flat_nav)
                            flat_tile = out_buffer[tile_slices[scheme_index],
                                                   buffer_nav_slice]
                            tile_shape = tile_shapes[scheme_index] + flat_tile.shape[-1:]
                            tile = flat_tile.reshape(tile_shape, order=self._sig_order)
                            # must roll final axis to provide shape == (nav, *sig)
                            tile = np.moveaxis(tile, -1, 0)
                            yield idcs_in_flat_nav, scheme_index, tile

    def _combine_raw_futures(self,
                             futures: Iterable[concurrent.futures.Future],
                             pool: concurrent.futures.Executor) -> List[concurrent.futures.Future]:
        """
        Combines futures according to self._chunk_combinations,
        which was created during __init__, to allow us to wait for multiple
        chunks to complete before yielding tiles which depend on them
        """
        return [pool.submit(self._combine, *tuple(futures[i] for i in combination))
                for combination in self._chunk_combinations.keys()]

    @staticmethod
    def _combine(*futures: concurrent.futures.Future) -> tuple[int]:
        """
        Wait on all provided futures to complete before returning
        """
        return tuple(f.result() for f in futures)

    @classmethod
    def _plan_reads(cls,
                    ideal_depth: int,
                    ts_depth: int,
                    *index_or_slice: Union[int, slice]) -> Tuple[int, List]:
        """
        For a given scheme depth and frame indices to read, try to find a good
        sequence of reads to perform which minimises the number of passes through the file

        Constraint: must yield tiles/frames with ts_depth at the end to be consistent

        Steps are:
         - Compute a good buffer_length which is multiple of ts_depth
         - Split big slices into chunks <= buffer_length
         - Combine slices sequentially up to buffer_length

        Each time we fill the buffer should ideally represent one pass through the
        the file, though in practice it depends on how the low-level file access is
        coded, especially when using an ROI and the reads for a single buffer fill
        are sparse/separated

        This function performs a similar role to get_read_ranges as it also specifies
        how to unpack the filled buffers into the data to emit as tiles
        """
        # First combine any sequential indices or slices into contiguous slices
        # will cast any int in index_or_slice to slice(int, int + 1) (or combine it)
        index_or_slice = cls._combine_sequential(*index_or_slice)
        # Compute number of frames to read
        slice_lengths = tuple(cls._length_slice(s) for s in index_or_slice)
        to_read = sum(slice_lengths)

        # Compute the size of the buffer (num_frames) we will read into
        if to_read <= ts_depth:
            # In this case we can complete in a single pass
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
            buffer_length = ideal_depth - (ideal_depth % ts_depth)
            buffer_length = max(ts_depth, buffer_length)

        # split splices down to len(sl) == buffer_length at maximum
        # must functools.reduce because self._split_slice returns tuples of slices
        index_or_slice = functools.reduce(operator.add,
                                          tuple(cls._split_slice(sl, buffer_length)
                                                for sl in index_or_slice))

        # Pre-compute the slice lengths
        slice_info = []
        for element in index_or_slice:
            read_length = (element.stop - element.start)
            first_idx = element.start
            last_idx = element.stop - 1
            slice_info.append((read_length, first_idx, last_idx))

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

        # Build combinations of sequential slices until we exceed buffer_length
        # Working set is in _combining, completed sets are in combinations
        idx = 0
        combinations = []
        _combining = []
        num_slices = len(slice_info)
        while idx < num_slices:
            read_length, first_idx, last_idx = slice_info[idx]
            # start of loop or new potential combination
            # optimistically add it and move on
            if not _combining:
                _combining.append(idx)
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

        # Handle any hanging combination
        if _combining:
            combinations.append(tuple(_combining))
        _combining.clear()

        # Now apply the combinations and convert them into groups of slice objects
        # which specify how to pack into and upack from the read buffer
        reads = []
        # The return value reads has the following structure
        #     list of (tuple(slice,...) for_memmap), slice_for_buffer_alloc, list(unpacks))
        # where:
        #     unpacks == (slice_in_buffer, tuple(int, ...) of flat_frame_idcs)
        for combination in combinations:
            mmap_slices = tuple(index_or_slice[i] for i in combination)
            # If we are just handling a single contig combination can optimise here by not
            # returning a frame index array but rather just a single nav slice object
            frame_idcs = itertools.chain(*tuple(range(slice_info[i][1], slice_info[i][2] + 1)
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
    def _gen_slices_for_depth(length: int, depth: int) -> Generator[tuple[int, int], None, None]:
        """
        Generate (lower, upper) index integers which split length
        into chunks of size depth, including a final chunk <= depth
        """
        assert length
        boundaries = [*range(0, length, depth)]
        if boundaries[-1] != length:
            boundaries.append(length)
        yield from zip(boundaries[:-1], boundaries[1:])

    @staticmethod
    def _length_slice(sl_or_int: Union[slice, int]) -> int:
        """
        Get the length of an increasing slice object. Returns 1 for integers
        on the assumption that they represent slice(i, i + 1)

        This function is only used in the context of partitions
        and so it assumes non-negative / non-zero slice objects
        slices are very flexible so it is impossible to define a
        fully coherent 'length' except in restricted circumstances
        """
        if isinstance(sl_or_int, slice):
            assert sl_or_int.step in (1, None)
            if sl_or_int.start is not None:
                assert sl_or_int.stop > sl_or_int.start
                return sl_or_int.stop - sl_or_int.start
            else:
                return sl_or_int.stop
        else:
            assert isinstance(sl_or_int, int)
            return 1

    @classmethod
    def _split_slice(cls, sl: slice, target_length: int) -> Tuple[slice]:
        """
        Split a slice into tuple[subslices] of maximum length target_length
        The final slice is <= target_length and the whole slice is covered
        """
        sl_length = cls._length_slice(sl)
        if sl_length <= target_length:
            if isinstance(sl, int):
                sl = slice(sl, sl + 1)
            return (sl,)
        start = sl.start
        if start is None:
            start = 0
        return tuple(slice(start + lb, start + ub)
                     for lb, ub
                     in cls._gen_slices_for_depth(sl_length, target_length))

    @staticmethod
    def _splat_iterables(*index_or_slice: Union[Iterable, Any]) -> Generator[Any, None, None]:
        """
        Yield from any elements of index_or_slice which are
        iterable, otherwise yield the element itself.
        Similar to itertools.chain but yields the non-iterables
        which are given as arguments, too.
        e.g. (0, (1, 2, 3), 4) => (0, 1, 2, 3, 4)
        """
        for sl in index_or_slice:
            try:
                yield from sl
            except TypeError:
                yield sl

    @staticmethod
    def _combine_sequential(*index_or_slice: Union[slice, int]) -> Tuple[slice]:
        """
        Combines an iterable of slice or integer indices
        into an iterable of slices. Any items in index_or_slice
        which are sequential (slice0.stop == slice1.start) or (slice0.stop == int1)
        are combined into a new slice slice(slice0.start, slice1.stop)
        """
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

    @classmethod
    def _slice_combine_array(cls, *slices: slice) -> List[Union[slice, List[int]]]:
        """
        Iterate over slices to find any which are shorter than threshold_combine
        and if so convert them into an integer array rather than a slice.
        If any slices are adjacent and both shorter than threshold_combine
        then the integer arrays are subsequently combined

        Used to combine read operations for sparsely filled ROIs where we'd
        like to read the data for many ROI points in one pass
        """
        if len(slices) <= 1:
            # Skip for empty or single slice objects
            # Single slices are common for Pick frame or (roi is None)
            return slices
        _slices = []
        for sl in slices:
            if sl.start is None:
                sl = slice(0, sl.stop)
            if (sl.stop - sl.start) <= cls.THRESHOLD_COMBINE:
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
        # could convert any fully sequential index arrays back to slices here
        return _slices
