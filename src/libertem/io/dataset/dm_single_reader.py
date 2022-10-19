import os
import mmap
import operator
import contextlib
import typing
from typing import Tuple, Union, Generator, Iterable, List, Set, Dict, Any, NamedTuple
from typing_extensions import Literal
import itertools
import functools
import numpy as np
from math import ceil, floor

from libertem.common.math import prod, accumulate

if typing.TYPE_CHECKING:
    from libertem.common.shape import Shape
    from libertem.io.dataset.base.tiling_scheme import TilingScheme
    from numpy.typing import DTypeLike


class MemmapContainer(NamedTuple):
    idx: int
    memmap: Union[mmap.mmap, None]
    array: Union[np.ndarray, None]


empty_memmapc = MemmapContainer(-1, None, None)


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
    MAX_MEMMAP_SIZE = 1024 * 2 ** 20  # 1024 MB
    BUFFER_SIZE: int = 128 * 2 ** 20
    THRESHOLD_COMBINE: int = 8  # small slices to convert to index arrays

    def __init__(self,
                 path: os.PathLike,
                 shape: 'Shape',
                 dtype: np.dtype,
                 tiling_scheme: 'TilingScheme',
                 sig_order: Literal['F', 'C'] = 'F',
                 file_header: int = 0):
        self._path = path
        self._dtype = dtype
        self._file_header = file_header
        self._shape = shape
        self._sig_size = shape.sig.size
        self._sig_order = sig_order
        self._tiling_scheme = tiling_scheme

        if sig_order not in ('F', 'C'):
            raise ValueError('Unrecognized sig_order')

        chunks, chunk_scheme_indices = self.choose_chunks(tiling_scheme,
                                                          shape,
                                                          dtype)
        chunksizes = tuple(self._byte_size(chunkshape)
                           for chunkshape in chunks)
        offsets = tuple(self._file_header + sum(chunksizes[:idx])
                        for idx in range(len(chunksizes)))
        self._chunks = tuple((o, c) for o, c in zip(offsets, chunks))
        self._chunk_map = self.build_chunk_map(chunk_scheme_indices)

        self._handle = None
        self._memmap = empty_memmapc

    @classmethod
    def choose_chunks(cls, tiling_scheme: 'TilingScheme', shape: 'Shape', dtype: 'DTypeLike'):
        """
        Choose sequential chunks of the file which generally align
        with the tiling_scheme and are as close to MAX_MEMMAP_SIZE
        as possible. Each chunk will be accessed separately to manage
        memory usage, hence the adherence to the size limit.

        A chunk may contain multiple tiles, or multiple adjacent chunks
        may need to combine to provide a single tile.
        """
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
        max_width_bytesize = cls.MAX_MEMMAP_SIZE // nav_bytesize
        data_size = sig_size * nav_size
        max_width = min(data_size, max_width_bytesize)

        # Split big chunks
        while max(tile_widths) > max_width:
            max_idx = np.argmax(tile_widths)
            max_val = tile_widths[max_idx]
            new_widths = [floor(max_val / 2), ceil(max_val / 2)]
            new_schemes = [scheme_indices[max_idx], scheme_indices[max_idx]]
            tile_widths = tile_widths[:max_idx] + new_widths + tile_widths[max_idx + 1:]
            scheme_indices = scheme_indices[:max_idx] + new_schemes + scheme_indices[max_idx + 1:]
        scheme_indices = [{i} for i in scheme_indices]

        # Combine small chunks
        while True:
            merge_to = None
            for idx, width in enumerate(tile_widths[:-1]):
                if tile_widths[idx + 1] + width <= max_width:
                    merge_to = idx + 1
                    break
            if merge_to is not None:
                tile_widths[merge_to] += width
                scheme_indices[merge_to] = scheme_indices[merge_to].union(scheme_indices[idx])
                tile_widths.pop(idx)
                scheme_indices.pop(idx)
            else:
                break

        # Prepare outputs
        tile_widths: Tuple[int]
        chunks = tuple((width,) + (nav_size,)
                       for width in tile_widths)
        return chunks, scheme_indices

    @staticmethod
    def build_chunk_map(chunk_scheme_indices: List[Set[int]]) -> Dict[Tuple[int, ...], Set[int]]:
        """
        Build the mapping between a chunk of the file and
        the scheme indices it can provide. Needed as depending on
        layout / tile size, a chunk can provide any of:

            - only part of a single tile stack
            - a mixture of complete and incomplete tile stacks

        and therefore might have to be combined with other chunks
        before a tile can be ready for processing

        Returns a single dictonary mapping {chunk_idx: {scheme_indices}}
        for individual chunks, and {tuple(chunk_idx): {scheme_indices}}
        for necessary combinations of chunks and the scheme_indices provided
        Any chunk providing no tiles on its own is removed from the mapping
        """
        if not all(c for c in chunk_scheme_indices):
            raise ValueError('Cannot map empty chunk')
        if not all(max(s0) <= min(s1) for s0, s1 in zip(chunk_scheme_indices[:-1],
                                                        chunk_scheme_indices[1:])):
            raise ValueError('Chunks provide out-of-order scheme indices?')
        # scheme_indices is list(set(int)) of which tile idxs
        # are generated by each memmap chunk
        # Adjacent chunks can provide the parts of a tile which spans
        # between them, but the overall sequence of scheme indexes must
        # be increasing along the chunks
        working_key = None
        scheme_map = {}
        # Build combinations of chunks based on the tiling idxs they provide
        # For very disadvantageous layouts this will lead to over-combination
        # and therefore even more read-amplication overhead, but this can be
        # avoided by good choice of tileshape
        for chunk_idx, scheme_idxs in enumerate(chunk_scheme_indices):
            _key = (chunk_idx,)
            if not scheme_map or not scheme_map[working_key].intersection(scheme_idxs):
                working_key = _key
                scheme_map[working_key] = scheme_idxs
            else:
                current_set = scheme_map.pop(working_key)
                working_key = working_key + _key
                scheme_map[working_key] = current_set.union(scheme_idxs)
        # Sort to ensure chunks are read always read in order
        return {tuple(sorted(k)): v for k, v in scheme_map.items()}

    @contextlib.contextmanager
    def open_file(self):
        self._handle = open(self._path, 'rb')
        yield
        try:
            self.close_memmap()
        finally:
            self._handle.close()

    def create_memmap(self, idx: int):
        offset, chunkshape = self._chunks[idx]
        chunksize_bytes = prod(chunkshape) * np.dtype(self._dtype).itemsize
        # Must memmap with an offset aligned mmap.ALLOCATIONGRANULARITY
        nearest_alignment = offset % mmap.ALLOCATIONGRANULARITY
        memmap = mmap.mmap(
            fileno=self._handle.fileno(),
            length=chunksize_bytes + nearest_alignment,
            offset=offset - nearest_alignment,
            access=mmap.ACCESS_READ,
        )
        array: np.ndarray = np.frombuffer(memoryview(memmap)[nearest_alignment:],
                                          dtype=self._dtype)
        return MemmapContainer(idx, memmap, array.reshape(chunkshape))

    def get_memmap(self, idx: int):
        if self._memmap.idx != idx:
            if self._memmap.memmap is not None:
                self.close_memmap()
            self._memmap = self.create_memmap(idx)
        return self._memmap.array

    def close_memmap(self):
        memmap = self._memmap.memmap
        array = self._memmap.array
        self._memmap = empty_memmapc
        # memmap.madvise(mmap.MADV_DONTNEED)
        # memmap.close()
        del array
        del memmap

    def _byte_size(self, shape: Iterable) -> int:
        return np.dtype(self._dtype).itemsize * np.prod(shape, dtype=np.int64)

    def _load_data(self,
                   memmap: np.ndarray,
                   slices: Tuple[Iterable, slice],
                   out_buf: np.ndarray) -> int:
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

    @classmethod
    def _ideal_buffer_size(cls,
                           chunksizes: Tuple[int, ...],
                           combinations: Iterable[int],
                           dtype: 'DTypeLike') -> Tuple[int, int]:
        """
        Find the largest chunk or necessary combination of chunks
        and determine the depth which creates a buffer best matching
        the BUFFER_SIZE attribute
        """
        combination_sizes = tuple(sum(chunksizes[kk] for kk in k)
                                  for k in combinations
                                  if isinstance(k, tuple))
        max_sig_block = max(combination_sizes + chunksizes)
        # implicit max with the 1 + even if // rounds down to 0
        ideal_depth = 1 + cls.BUFFER_SIZE // (max_sig_block * np.dtype(dtype).itemsize)
        return (max_sig_block, ideal_depth)

    def generate_tiles(self, *index_or_slice: Union[int, slice, Iterable])\
            -> Generator[Tuple[Tuple[int], int, np.ndarray], None, None]:
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
        chunksizes = tuple(c[0] for _, c in self._chunks)
        chunk_offsets = tuple(accumulate(chunksizes, initial=0))
        (max_sig_block, ideal_depth) = self._ideal_buffer_size(chunksizes,
                                                               self._chunk_map.keys(),
                                                               self._dtype)

        tile_shapes = tuple(s.shape.to_tuple() for s in self._tiling_scheme)
        tile_slices = self._flat_tile_slices(tile_shapes)
        index_or_slice = tuple(self._splat_iterables(*index_or_slice))
        buffer_length, reads = self._plan_reads(ideal_depth,
                                                self._tiling_scheme.depth,
                                                *index_or_slice)
        out_buffer = np.empty((max_sig_block, buffer_length), dtype=self._dtype)

        with self.open_file():
            # Outer chunk loop traverses all tiles in the tiling scheme
            # chunks can be int | tuple[int, ...] mapping the chunks which need
            # to be opened to provided the given scheme_indices
            for chunks, scheme_indices in self._chunk_map.items():
                # The tiling scheme slices are stored in whole-dataset coordinates
                # but we read into a truncated buffer, this offset shifts the sig
                # slice for this group of chunks into this truncated system
                buffer_sig_offset = chunk_offsets[chunks[0]]
                # For a given set of chunks (or single chunk), perform all necessary reads
                for mmap_nav_slices, buffer_nav_slice, buffer_unpacks in reads:
                    # Phase 1: perform reads into buffer
                    sig_read = 0
                    # Inner chunk loop accounts for needing to combine data
                    # from more than one chunk to build a tile/frame stack
                    for raw_idx in chunks:
                        # If just reading from a single chunk to make a tile then
                        # memmap is only created once per item in _chunk_map
                        # If we are building tiles/frames from chunk combinations we have
                        # to switch the memmap multiple times for each fill of the buffer
                        # which is the worst-case for this layout (massive overheads),
                        # this is necessary because Dask limits us to ~1GiB memmaps
                        memmap = self.get_memmap(raw_idx)
                        sig_read_length = chunksizes[raw_idx]
                        # The calls to _load_data could be threaded to parallise I/O and processing
                        self._load_data(
                            memmap,
                            mmap_nav_slices,
                            out_buffer[sig_read: sig_read + sig_read_length,
                                    buffer_nav_slice],
                        )
                        sig_read += sig_read_length

                    # Phase 2: unpack buffer and yield tiles
                    for (unpack_nav_slice, idcs_in_flat_nav) in buffer_unpacks:
                        for scheme_index in scheme_indices:
                            flat_tile_slice = tile_slices[scheme_index]
                            # Shift tile slice from whole sig buffer to truncated sig buffer
                            shifted_tile_slice = slice(flat_tile_slice.start - buffer_sig_offset,
                                                    flat_tile_slice.stop - buffer_sig_offset)
                            # flat_tile has dims (flat_sig, flat_nav)
                            flat_tile = out_buffer[shifted_tile_slice,
                                                unpack_nav_slice]
                            # reshape from flat tile to shape expected by UDF
                            tile_shape = tile_shapes[scheme_index] + flat_tile.shape[-1:]
                            tile = flat_tile.reshape(tile_shape, order=self._sig_order)
                            # must roll final axis to provide shape == (nav, *sig)
                            # TODO benchmark how costly this is
                            tile = np.moveaxis(tile, -1, 0)
                            yield idcs_in_flat_nav, scheme_index, tile

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

        # Don't allocate buffer longer than frames in the partition
        buffer_length = min(buffer_length, to_read)

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
            combined_slices = cls._slice_combine_array(*mmap_slices)
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
            reads.append((combined_slices, buffer_allocation, unpacks))

        return buffer_length, reads

    @staticmethod
    def _gen_slices_for_depth(length: int, depth: int) -> Generator[Tuple[int, int], None, None]:
        """
        Generate (lower, upper) index integers which split length
        into chunks of size depth, including a final chunk <= depth
        """
        if not (length > 0 and depth > 0):
            raise ValueError('Cannot generate slices for non-positive length/depth')
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

    @staticmethod
    def _flat_tile_slices(tile_shapes: Tuple[Tuple[int, ...]]) -> Tuple[slice]:
        """
        Create slice objects for iterable of shapes assuming they
        represent a single contiguous flat dimension and are sequential
        """
        tile_size = tuple(map(prod, tile_shapes))
        boundaries = tuple(accumulate(tile_size, initial=0))
        return tuple(slice(a, b) for a, b
                     in zip(boundaries[:-1], boundaries[1:]))

    @staticmethod
    def _sort_dict_int(kv):
        """
        Return first element of k is tuple if tuple else k
        """
        key, _ = kv
        return key[0] if isinstance(key, tuple) else key
