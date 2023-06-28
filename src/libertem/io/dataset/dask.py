import logging
import itertools
import numpy as np
import dask.array as da

from libertem.common import Shape, Slice
from libertem.io.dataset.base import (
    DataSet, DataSetMeta, BasePartition, File, FileSet, DataSetException
)
from libertem.io.dataset.base.backend_mmap import MMapFile, MMapBackend, MMapBackendImpl
from libertem.common.messageconverter import MessageConverter

log = logging.getLogger(__name__)


class DaskDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/DaskDatasetParams.schema.json",
        "title": "DaskDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "DASK"},
            "sig_dims": {"type": "number", "minimum": 1},
            "preserve_dimensions": {"type": "boolean"},
            "min_size": {"type": "number", "minimum": 1},
        },
        "required": ["type"],
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["sig_dims", "preserve_dimensions", "min_size"]
            if k in raw_data
        }
        return data


class FakeDaskMMapFile(MMapFile):
    """
    Implementing the same interface as MMapFile, without filesystem backing
    """
    def open(self):
        # scheduler='threads' ensures that upstream computation for this array
        # chunk happens completely on this worker and not elsewhere
        self._arr = self.desc._array.compute(scheduler='threads')
        # need to be aware that Dask can create Fortran-ordered arrays
        # when .compute is called, which can lead to downstream issues when
        # np.frombuffer is called on self._mmap in the backend. Currently it seems
        # like np.frombuffer cannot handle Fortran ordering and throws a ValueError
        self._mmap = self._arr
        return self

    def close(self):
        del self._arr
        del self._mmap


class DaskBackend(MMapBackend):
    def get_impl(self):
        return DaskBackendImpl()


class DaskBackendImpl(MMapBackendImpl):
    FILE_CLS = FakeDaskMMapFile


class DaskDataSet(DataSet):
    """
    .. versionadded:: 0.9.0

    Wraps a Dask.array.array such that it can be processed by LiberTEM.
    Partitions are created to be aligned with the array chunking. When
    the array chunking is not compatible with LiberTEM the wrapper
    merges chunks until compatibility is achieved.

    The best-case scenario is for the original array to be chunked in
    the leftmost navigation dimension. If instead another navigation
    dimension is chunked then the user can set `preserve_dimension=False`
    to re-order the navigation shape to achieve better chunking for LiberTEM.
    If more than one navigation dimension is chunked, the class will do
    its best to merge chunks without creating partitions which are too large.

    LiberTEM requires that a partition contains only whole signal frames,
    so any signal dimension chunking is immediately merged by this class.

    This wrapper is most useful when the Dask array was created using
    lazy I/O via `dask.delayed`, or via `dask.array` operations.
    The major assumption is that the chunks in the array can each be
    individually evaluated without having to read or compute more data
    than the chunk itself contains. If this is not the case then this class
    could perform very poorly due to read amplification, or even crash the Dask
    workers.

    As the class performs rechunking using a merge-only strategy it will never
    split chunks which were present in the original array. If the array
    is originally very lightly chunked, then the corresponding LiberTEM partitions
    will be very large. In addition, overly-chunked arrays (for example one chunk per
    frame) can incurr excessive Dask task graph overheads and should be avoided
    where possible.

    Parameters
    ----------

    dask_array: dask.array.array
        A Dask array

    sig_dims: int
        Number of dimensions in dask_array.shape counting from the right
        to treat as signal dimensions

    preserve_dimensions: bool, optional
        If False, allow optimization of the dask_arry chunking by
        re-ordering the nav_shape to put the most chunked dimensions first.
        This can help when more than one nav dimension is chunked.

    min_size: float, optional
        The minimum partition size in bytes if the array chunking allows
        an order-preserving merge strategy. The default min_size is 128 MiB.

    io_backend: bool, optional
        For compatibility, accept an unused io_backend argument.

    Example
    --------

    >>> import dask.array as da
    >>>
    >>> d_arr = da.ones((4, 4, 64, 64), chunks=(2, -1, -1, -1))
    >>> ds = ctx.load('dask', dask_array=d_arr, sig_dims=2)

    Will create a dataset with 5 partitions split along the zeroth dimension.
    """
    # TODO add mechanism to re-order the dimensions of results automatically
    # if preserve_dimensions is set to False
    def __init__(self, dask_array, *, sig_dims, preserve_dimensions=True,
                 min_size=None, io_backend=None):
        super().__init__(io_backend=io_backend)
        if io_backend is not None:
            raise DataSetException("DaskDataSet currently doesn't support alternative I/O backends")

        self._check_array(dask_array, sig_dims)
        self._array = dask_array
        self._sig_dims = sig_dims
        self._sig_shape = self._array.shape[-self._sig_dims:]
        self._dtype = self._array.dtype
        self._preserve_dimension = preserve_dimensions
        self._min_size = min_size
        if self._min_size is None:
            # TODO add a method to determine a sensible partition byte-size
            self._min_size = self._default_min_size

    @property
    def array(self):
        return self._array

    def get_io_backend(self):
        return DaskBackend()

    def initialize(self, executor):
        self._array = self._adapt_chunking(self._array, self._sig_dims)
        self._nav_shape = self._array.shape[:-self._sig_dims]

        self._nav_shape_product = int(np.prod(self._nav_shape))
        self._image_count = self._nav_shape_product
        shape = Shape(self._nav_shape + self._sig_shape, sig_dims=self._sig_dims)
        self._meta = DataSetMeta(
            shape=shape,
            raw_dtype=np.dtype(self._dtype),
            sync_offset=0,
            image_count=self._nav_shape_product,
        )
        return self

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    @classmethod
    def get_msg_converter(cls):
        return DaskDatasetParams

    @property
    def _default_min_size(self):
        """
        Default minimum chunk size if not supplied at init
        """
        return 128 * (2**20)  # MB

    def _chunk_slices(self, array):
        chunks = array.chunks
        boundaries = tuple(tuple(self.chunks_to_slices(chunk_lengths)) for chunk_lengths in chunks)
        return tuple(itertools.product(*boundaries))

    def _adapt_chunking(self, array, sig_dims):
        n_dimension = array.ndim
        # Handle chunked signal dimensions by merging just in case
        sig_dim_idxs = [*range(n_dimension)[-sig_dims:]]
        if any([len(array.chunks[c]) > 1 for c in sig_dim_idxs]):
            original_n_chunks = [len(c) for c in array.chunks]
            array = array.rechunk({idx: -1 for idx in sig_dim_idxs})
            log.warning('Merging sig dim chunks as LiberTEM does not '
                        'support paritioning along the sig axes. '
                        f'Original n_blocks: {original_n_chunks}. '
                        f'New n_blocks: {[len(c) for c in array.chunks]}.')
        # Warn if there is no nav_dim chunking
        n_nav_chunks = [len(dim_chunking) for dim_chunking in array.chunks[:-sig_dims]]
        if set(n_nav_chunks) == {1}:
            log.warning('Dask array is not chunked in navigation dimensions, '
                        'cannot split into nav-partitions without loading the '
                        'whole dataset on each worker. '
                        f'Array shape: {array.shape}. '
                        f'Chunking: {array.chunks}. '
                        f'array size {array.nbytes / 1e6} MiB.')
            # If we are here there is nothing else to do.
            return array
        # Orient the nav dimensions so that the zeroth dimension is
        # the most chunked, this obviously changes the dataset nav_shape !
        if not self._preserve_dimension:
            n_nav_chunks = [len(dim_chunking) for dim_chunking in array.chunks[:-sig_dims]]
            nav_sort_order = np.argsort(n_nav_chunks)[::-1].tolist()
            sort_order = nav_sort_order + sig_dim_idxs
            if not np.equal(sort_order, np.arange(n_dimension)).all():
                original_shape = array.shape
                original_n_chunks = [len(c) for c in array.chunks]
                array = da.transpose(array, axes=sort_order)
                log.warning('Re-ordered nav_dimensions to improve partitioning, '
                            'create the dataset with preserve_dimensions=True '
                            'to suppress this behaviour. '
                            f'Original shape: {original_shape} with '
                            f'n_blocks: {original_n_chunks}. '
                            f'New shape: {array.shape} with '
                            f'n_blocks: {[len(c) for c in array.chunks]}.')
        # Handle chunked nav_dimensions
        # We can allow nav_dimensions to be fully chunked (one chunk per element)
        # up-to-but-not-including the first non-fully chunked dimension. After this point
        # we must merge/rechunk all subsequent nav dimensions to ensure continuity
        # of frame indexes in a flattened nav dimension. This should be removed
        # when if we allow non-contiguous flat_idx Partitions
        nav_rechunk_dict = {}
        for dim_idx, dim_chunking in enumerate(array.chunks[:-sig_dims]):
            if set(dim_chunking) == {1}:
                continue
            else:
                merge_dimensions = [*range(dim_idx + 1, n_dimension - sig_dims)]
                for merge_i in merge_dimensions:
                    if len(array.chunks[merge_i]) > 1:
                        nav_rechunk_dict[merge_i] = -1
        if nav_rechunk_dict:
            original_n_chunks = [len(c) for c in array.chunks]
            array = array.rechunk(nav_rechunk_dict)
            log.warning('Merging nav dimension chunks according to scheme '
                        f'{nav_rechunk_dict} as we cannot maintain continuity '
                        'of frame indexing in the flattened navigation dimension. '
                        f'Original n_blocks: {original_n_chunks}. '
                        f'New n_blocks: {[len(c) for c in array.chunks]}.')
        # Merge remaining chunks maintaining C-ordering until we reach a target chunk sizes
        # or a minmum number of partitions corresponding to the number of workers
        new_chunking, min_size, max_size = merge_until_target(array, self._min_size)
        if new_chunking != array.chunks:
            original_n_chunks = [len(c) for c in array.chunks]
            chunksizes = get_chunksizes(array)
            orig_min, orig_max = chunksizes.min(), chunksizes.max()
            array = array.rechunk(new_chunking)
            log.warning('Applying re-chunking to increase minimum partition size. '
                        f'n_blocks: {original_n_chunks} => {[len(c) for c in array.chunks]}. '
                        f'Min chunk size {orig_min / 1e6:.1f} => {min_size / 1e6:.1f} MiB , '
                        f'Max chunk size {orig_max / 1e6:.1f} => {max_size / 1e6:.1f} MiB.')
        return array

    def _check_array(self, array, sig_dims):
        if not isinstance(array, da.Array):
            raise DataSetException('Expected a Dask array as input, recieved '
                                   f'{type(array)}.')
        if not isinstance(sig_dims, int) and sig_dims >= 0:
            raise DataSetException('Expected non-negative integer sig_dims,'
                                   f'recieved {sig_dims}.')
        if any([np.isnan(c).any() for c in array.shape])\
           or any([np.isnan(c).any() for c in array.chunks]):
            raise DataSetException('Dask array has an unknown shape or chunk sizes '
                                   'so cannot be interpreted as a LiberTEM partitions. '
                                   'Run array.compute_compute_chunk_sizes() '
                                   'before passing to DaskDataSet, though this '
                                   'may be performance-intensive. Chunking: '
                                   f'{array.chunks}, Shape {array.shape}')
        if sig_dims >= array.ndim:
            raise DataSetException(f'Number of sig_dims {sig_dims} not compatible '
                                   f'with number of array dims {array.ndim}, '
                                   'must be able to create partitions along nav '
                                   'dimensions.')
        return True

    def check_valid(self):
        return self._check_array(self._array, self._sig_dims)

    def get_num_partitions(self):
        return len([*itertools.product(*self._array.chunks)])

    @staticmethod
    def chunks_to_slices(chunk_lengths):
        prior = 0
        for c in chunk_lengths:
            newc = c + prior
            yield slice(prior, newc)
            prior = newc

    @staticmethod
    def slices_to_shape(slices):
        return tuple(s.stop - s.start for s in slices)

    @staticmethod
    def slices_to_origin(slices):
        return tuple(s.start for s in slices)

    @staticmethod
    def flatten_nav(slices, nav_shape, sig_dims):
        """
        Because LiberTEM partitions are set up with a flat nav dimension
        we must flatten the Dask array slices. This is ensured to be possible
        by earlier calls to _adapt_chunking but should be removed if ever
        partitions are able to have >1D navigation axes.
        """
        nav_slices = slices[:-sig_dims]
        sig_slices = slices[-sig_dims:]
        start_frame = np.ravel_multi_index([s.start for s in nav_slices], nav_shape)
        end_frame = 1 + np.ravel_multi_index([s.stop - 1 for s in nav_slices], nav_shape)
        nav_slice = slice(start_frame, end_frame)
        return (nav_slice,) + sig_slices, start_frame, end_frame

    def get_slices(self):
        """
        Generates the LiberTEM slices which correspond to the chunks
        in the Dask array backing the dataset

        Generates both the flat_nav slice for creating the LiberTEM partition
        and also the full_slices used to index into the dask array
        """
        chunk_slices = self._chunk_slices(self._array)

        for full_slices in chunk_slices:
            flat_slices, start_frame, end_frame = self.flatten_nav(full_slices, self._nav_shape,
                                                                   self._sig_dims)
            flat_slice = Slice(origin=self.slices_to_origin(flat_slices),
                               shape=Shape(self.slices_to_shape(flat_slices),
                                           sig_dims=self._sig_dims))
            yield full_slices, flat_slice, start_frame, end_frame

    def _get_fileset(self):
        """
        The fileset is set up to have one 'file' per partition
        which corresponds to one 'file' per Dask chunk
        """
        partitions = []
        for full_slices, _, start, stop in self.get_slices():
            partitions.append(DaskFile(
                array_chunk=self._array[full_slices],
                path=None,
                start_idx=start,
                end_idx=stop,
                native_dtype=self._dtype,
                sig_shape=self.shape.sig
            ))
        return DaskFileSet(partitions)

    def get_partitions(self):
        """
        Partitions contain a reference to the whole array and the whole
        fileset, but the part_slice and start_frame/num_frames provided mean
        that the subsequent call to get_read_ranges() means only one 'file'
        is read/.compute(), and this corresponds to the partition *exactly*
        """
        fileset = self._get_fileset()
        for _, part_slice, start, stop in self.get_slices():
            yield DaskPartition(
                self._array,
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
                io_backend=self.get_io_backend(),
                decoder=self.get_decoder()
            )

    def __repr__(self):
        return (f"<DaskDataSet of {self.dtype} shape={self.shape}, "
                f"n_blocks={[len(c) for c in self._array.chunks]}>")


class DaskFile(File):
    def __init__(self, *args, array_chunk=None, **kwargs):
        """
        Upon creation, the dask array has been sliced to give
        only one chunk corresponding to a LiberTEM partition
        """
        self._array = array_chunk
        super().__init__(*args, **kwargs)


class DaskFileSet(FileSet):
    pass


class DaskPartition(BasePartition):
    def __init__(self, dask_array, *args, **kwargs):
        self._array = dask_array
        super().__init__(*args, **kwargs)


def array_mult(*arrays, dtype=np.float64):
    num_arrays = len(arrays)
    if num_arrays == 1:
        return np.asarray(arrays[0]).astype(dtype)
    elif num_arrays == 2:
        return np.multiply.outer(*arrays).astype(dtype)
    elif num_arrays > 2:
        return np.multiply.outer(arrays[0], array_mult(*arrays[1:]))
    else:
        raise RuntimeError('Unexpected number of arrays')


def get_last_chunked_dim(chunking):
    n_chunks = [len(c) for c in chunking]
    chunked_dims = [idx for idx, el in enumerate(n_chunks) if el > 1]
    try:
        return chunked_dims[-1]
    except IndexError:
        return -1


def get_chunksizes(array, chunking=None):
    if chunking is None:
        chunking = array.chunks
    shape = array.shape
    el_bytes = array.dtype.itemsize
    last_chunked = get_last_chunked_dim(chunking)
    if last_chunked < 0:
        return np.asarray(array.nbytes)
    static_size = np.prod(shape[last_chunked + 1:], dtype=np.float64) * el_bytes
    chunksizes = array_mult(*chunking[:last_chunked + 1]) * static_size
    return chunksizes


def modify_chunking(chunking, dim, merge_idxs):
    chunk_dim = chunking[dim]
    merge_idxs = tuple(sorted(merge_idxs))
    before = chunk_dim[:merge_idxs[0]]
    after = chunk_dim[merge_idxs[1] + 1:]
    merged_dim = (sum(chunk_dim[merge_idxs[0]:merge_idxs[1] + 1]),)
    new_chunk_dim = tuple(before) + merged_dim + tuple(after)
    chunking = chunking[:dim] + (new_chunk_dim,) + chunking[dim + 1:]
    return chunking


def findall(sequence, val):
    return [idx for idx, e in enumerate(sequence) if e == val]


def neighbour_idxs(sequence, idx):
    max_idx = len(sequence) - 1
    if idx > 0 and idx < max_idx:
        return (idx - 1, idx + 1)
    elif idx == 0:
        return (None, idx + 1)
    elif idx == max_idx:
        return (idx - 1, None)
    else:
        raise


def min_neighbour(sequence, idx):
    left, right = neighbour_idxs(sequence, idx)
    if left is None:
        return right
    elif right is None:
        return left
    else:
        return min([left, right], key=lambda x: sequence[x])


def min_with_min_neighbor(sequence):
    min_val = min(sequence)
    occurences = findall(sequence, min_val)
    min_idx_pairs = [(idx, min_neighbour(sequence, idx)) for idx in occurences]
    pair = [sum(get_values(sequence, idxs)) for idxs in min_idx_pairs]
    min_pair = min(pair)
    min_pair_occurences = findall(pair, min_pair)
    return min_idx_pairs[min_pair_occurences[-1]]  # breaking ties from right


def get_values(sequence, idxs):
    return [sequence[idx] for idx in idxs]


def merge_until_target(array, target, min_chunks=0):
    chunking = array.chunks
    if array.nbytes < target:
        # A really small dataset, better to treat as one partition
        chunking = tuple((s,) for s in array.shape)
    chunksizes = get_chunksizes(array)
    while chunksizes.size > min_chunks and chunksizes.min() < target:
        if (chunksizes < 0).any():
            log.warn('Overflow in chunksize calculation, will be clipped!')
        chunksizes = np.clip(chunksizes, 0., np.inf)
        last_chunked_dim = get_last_chunked_dim(chunking)
        if last_chunked_dim < 0:
            # No chunking, by definition complete
            break
        last_chunking = chunking[last_chunked_dim]
        to_merge = min_with_min_neighbor(last_chunking)
        chunking = modify_chunking(chunking, last_chunked_dim, to_merge)
        chunksizes = get_chunksizes(array, chunking=chunking)
    return chunking, chunksizes.min(), chunksizes.max()
