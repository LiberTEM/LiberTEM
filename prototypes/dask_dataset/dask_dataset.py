import warnings
import logging
import itertools
import numpy as np
import dask.array as da

from libertem.common import Shape, Slice
from libertem.io.dataset.base import (
    DataSet, DataSetMeta, BasePartition, File, FileSet, DataSetException
)
from libertem.io.dataset.base.backend_mmap import MMapFile, MMapBackend
from libertem.io.dataset.memory import MemBackendImpl

from merge_util import merge_until_target, get_chunksizes

log = logging.getLogger(__name__)

class DaskRechunkWarning(RuntimeWarning):
    pass
warnings.simplefilter('always', DaskRechunkWarning)


class FakeDaskMMapFile(MMapFile):
    """
    Implementing the same interface as MMapFile, without filesystem backing
    """
    def open(self):
        # scheduler='threads' ensures that upstream computation for this array
        # chunk happens completely on this worker and not elsewhere
        self._arr = self.desc._array.compute(scheduler='threads')
        self._mmap = self._arr
        return self

    def close(self):
        self._arr = None
        self._mmap = None


class DaskBackend(MMapBackend):
    def get_impl(self):
        return DaskBackendImpl()


class DaskBackendImpl(MemBackendImpl):
    FILE_CLS = FakeDaskMMapFile


class DaskDataSet(DataSet):
    """
    DaskDataSet(DataSet)

    This dataset wraps a Dask.array.array and makes it compatible with the
    UDF interface. Partitions are created to be aligned with the array chunking
    where the restrictions of LiberTEM and Dask allow. When these restrictions are
    broken, tries to perform rechunking/merging and dimension re-ordering
    to achieve compatible and optimal behaviour. Clearly there are no guarantees.

    This is only useful if the underlying Dask array was created using
    lazy I/O with something like dask.delayed. The major assumption of this
    class is that the chunks in the provided dask array can each be individually
    .compute()'d without causing excessive read amplification. If this is not the case
    then this class could perform very poorly. This could occur either if the
    array was loaded without lazy, chunked I/O, or if upstream dask computations
    requried rechunking of the array before it was passed to this class.

    The class performs rechunking using a merge-only strategy, it will never
    split chunks which were present in the original array. Naturally, if the array
    is originally very lightly chunked, then the corresponding LiberTEM partitions
    will be very large. There is also a soft assumption that the underlying file
    is C-ordered, as we assume the signal dimensions are the rightmost and we use
    a merge strategy from right-to-left.

    Parameters
    ----------

    dask_array: dask.array.array
        A Dask array

    sig_dims: int
        Number of dimensions in dask_array.shape counting from the right
        to treat as signal dimensions

    preserve_dimensions: bool, optional
        Whether the prevent optimization of the dask_arry chunking by
        re-ordering the nav_shape to put the most chunked dimensions first.
        When False this can result in a change of nav_shape relative to the
        original array
        # TODO add mechanism to re-order the dimensions of results automatically

    min_size: float, optional
        The minimum partition size in bytes iff the array chunking allows
        an order-preserving merge strategy

    io_backend: bool, optional
        For compatibility, accept an unused io_backend argument

    Examples
    --------

    >>> from libertem.io.dataset.dask import DaskDataSet
    >>> import dask.array as da
    >>>
    >>> d_arr = da.ones((10, 100, 256, 256), chunks=(2, -1, -1, -1))
    >>> ds = DaskDataSet(d_arr, sig_dims=2)

    Will create a dataset with 5 partitions split along the zeroth dimension.
    """
    def __init__(self, dask_array, *, sig_dims, preserve_dimensions=False,
                 min_size=128e6, io_backend=None):
        super().__init__(io_backend=io_backend)
        if io_backend is not None:
            raise ValueError("DaskDataSet currently doesn't support alternative I/O backends")

        self._check_array(dask_array, sig_dims)
        self._array = dask_array
        self._sig_dims = sig_dims
        self._sig_shape = self._array.shape[-self._sig_dims:]
        self._dtype = self._array.dtype
        self._preserve_dimension = preserve_dimensions
        self._min_size = min_size

    @property
    def array(self):
        return self._array

    def _get_decoder(self):
        return None

    def get_io_backend(self):
        return DaskBackend()

    def initialize(self, executor):
        self._min_npart = len(executor.get_available_workers())
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

    def _chunk_slices(self, array):
        chunks = array.chunks
        boundaries = tuple(tuple(self.chunks_to_slices(chunk_lengths)) for chunk_lengths in chunks)
        return tuple(itertools.product(*boundaries))

    def _get_chunk(self, array, chunk_flat_idx):
        slices = self._chunk_slices(array)
        return array[slices[chunk_flat_idx]]

    def _adapt_chunking(self, array, sig_dims):
        # Warn if there is no nav_dim chunking
        n_nav_chunks = [len(dim_chunking) for dim_chunking in array.chunks[:-sig_dims]]
        if set(n_nav_chunks) == {1}:
            log.info('Dask array is not chunked in navigation dimensions, '
                      'cannot split into nav-partitions without loading the '
                      'whole dataset on each worker. '
                      f'Array shape: {array.shape}. '
                      f'Chunking: {array.chunks}. '
                      f'Array size {array.nbytes / 1e6} MiB.')
        n_dimension = array.ndim
        # Handle chunked signal dimensions by merging just in case
        sig_dim_idxs = [*range(n_dimension)[-sig_dims:]]
        if any([len(array.chunks[c]) > 1 for c in sig_dim_idxs]):
            original_n_chunks = [len(c) for c in array.chunks]
            array = array.rechunk({idx: -1 for idx in sig_dim_idxs})
            log.info('Merging sig dim chunks as LiberTEM does not '
                      'support paritioning along the sig axes. '
                      f'Original n_blocks: {original_n_chunks}. '
                      f'New n_blocks: {[len(c) for c in array.chunks]}.')
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
                log.info('Re-ordered nav_dimensions to improve partitioning, '
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
            log.info('Merging nav dimension chunks according to scheme '
                      f'{nav_rechunk_dict} as we cannot maintain continuity '
                      'of frame indexing in the flattened navigation dimension. '
                      f'Original n_blocks: {original_n_chunks}. '
                      f'New n_blocks: {[len(c) for c in array.chunks]}.')
        # Merge remaining chunks maintaining C-ordering until we reach a target chunk sizes
        # or a minmum number of partitions corresponding to the number of workers
        new_chunking, min_size, max_size = merge_until_target(array, self._min_size,
                                                              self._min_npart)
        if new_chunking != array.chunks:
            original_n_chunks = [len(c) for c in array.chunks]
            chunksizes = get_chunksizes(array)
            orig_min, orig_max = chunksizes.min(), chunksizes.max()
            array = array.rechunk(new_chunking)
            log.info('Applying re-chunking to increase minimum partition size. '
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
        if any([np.isnan(c).any() for c in array.shape]):
            raise DataSetException('Dask array has undetermined shape: '
                                   f'{array.shape}.')
        if any([np.isnan(c).any() for c in array.chunks]):
            raise DataSetException('Dask array has unknown chunk sizes so cannot '
                                   'be interpreted as a LiberTEM partitions. '
                                   'Run array.compute_compute_chunk_sizes() '
                                   'before passing to DaskDataSet, though this '
                                   'may be performance-intensive. Chunking: '
                                   f'{array.chunks}.')
        if sig_dims >= array.ndim:
            raise DataSetException(f'Number of sig_dims {sig_dims} not compatible '
                                   f'with number of array dims {array.ndim}, '
                                   'must be able to create partitions along nav '
                                   'dimensions.')
        return True

    def check_valid(self):
        return self._check_array(self._array, self._sig_dims)

    def get_num_partitions(self):
        return len(itertools.product(*self._array.chunks))

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
    def slices_to_end(slices):
        return tuple(s.stop for s in slices)

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

    def _get_decoder(self):
        return None

    def get_io_backend(self):
        return DaskBackend()
