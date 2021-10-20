import warnings
import logging
import itertools
import numpy as np
import dask.array as da

from libertem.common import Shape, Slice
from .base import (
    DataSet, DataSetMeta, BasePartition, File, FileSet
)
from libertem.io.dataset.base.backend_mmap import MMapFile, MMapBackend
from .memory import MemBackendImpl


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
    This dataset wraps a Dask.array.array and makes it compatible with the
    UDF interface. Partitions are created to be aligned with the array chunking
    where the restrictions of LiberTEM and Dask allow. When these restrictions are
    broken, tries to perform rechunking/merging and dimension re-ordering
    to achieve compatible and optimal behaviour. Clearly there are no guarantees.

    This is only useful if the underlying Dask array was created using
    lazy I/O with something like dask.delayed. If the root node for the
    Dask task graph backing the array loads the whole dataset into memory
    then you are better off using MemoryDataSet. Similarly if the dask_array
    has been re-chunked without preserving the original lazy I/O structure
    then this dataset will either underperform (read amplification) and at worst
    will cause memory trouble as each worker may load large chunks of data
    simultaneously.

    Parameters
    ----------

    dask_array: dask.array.array
        A Dask array

    sig_dims: int
        Number of dimensions in dask_array.shape counting from the right
        to treat as signal dimensions

    preserve_dimensions: bool, optional
        Whether the prevent optimization of the dask_arry chunking to
        avoid over-reading in a single partition. When False this can
        result in a change of nav_shape relative to the original array
        # TODO add mechanism to re-order the dimensions of results automatically

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
    def __init__(self, dask_array, *, sig_dims, preserve_dimensions=False, io_backend=None):
        super().__init__(io_backend=io_backend)
        if io_backend is not None:
            raise ValueError("DaskDataSet currently doesn't support alternative I/O backends")

        self._array = dask_array
        self._sig_dims = sig_dims
        self._sig_shape = self._array.shape[-self._sig_dims:]
        self._dtype = self._array.dtype
        self._preserve_dimension = preserve_dimensions

    @property
    def array(self):
        return self._array

    def _get_decoder(self):
        return None

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

    def _chunk_slices(self, array):
        chunks = array.chunks
        boundaries = tuple(tuple(self.chunks_to_slices(chunk_lengths)) for chunk_lengths in chunks)
        return tuple(itertools.product(*boundaries))

    def _get_chunk(self, array, chunk_flat_idx):
        slices = self._chunk_slices(array)
        return array[slices[chunk_flat_idx]]

    def _adapt_chunking(self, array, sig_dims):
        n_dimension = array.ndim
        # Handle chunked signal dimensions by merging just in case
        sig_dim_idxs = [*range(n_dimension)[-sig_dims:]]
        if any([len(array.chunks[c]) > 1 for c in sig_dim_idxs]):
            original_n_chunks = [len(c) for c in array.chunks]
            array = array.rechunk({idx: -1 for idx in sig_dim_idxs})
            warnings.warn(('Merging sig dim chunks as LiberTEM does not '
                           'support paritioning along the sig axes. '
                           f'Original n_blocks: {original_n_chunks}. '
                           f'New n_blocks: {[len(c) for c in array.chunks]}.'),
                          DaskRechunkWarning)
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
                warnings.warn(('Re-ordered nav_dimensions to improve partitioning, '
                               'create the dataset with preserve_dimensions=True '
                               'to suppress this behaviour. '
                               f'Original shape: {original_shape} with '
                               f'n_blocks: {original_n_chunks}. '
                               f'New shape: {array.shape} with '
                               f'n_blocks: {[len(c) for c in array.chunks]}.'),
                              DaskRechunkWarning)
        # Handle chunked nav_dimensions other than the first
        nav_rechunk_dict = {}
        for dim_idx, dim_chunking in enumerate(array.chunks[:-sig_dims]):
            if dim_idx == 0:
                continue
            # The only chunksize we can accept in the >zeroth nav_dims is 1
            # This is due to a limitation of how LiberTEM constructs partitions
            # using a flattened nav index (frame number/index)
            # If we lift this limitation then we don't need to rechunk nav dims
            # except to enforce logical chunk sizes (by merging only)
            if len(dim_chunking) > 1:
                unique_chunksizes = set(dim_chunking)
                if unique_chunksizes != {1}:
                    nav_rechunk_dict[dim_idx] = -1
        array = array.rechunk(nav_rechunk_dict)
        # Warn about poor dataset chunking for zeroth dimension
        if len(array.chunks[0]) == 1:
            warnings.warn(('Zeroth dimension of the array is not chunked, '
                           'this will likely lead to excessive memory usage '
                           'by loading the whole dataset in one operation. '
                           f'First chunk size is {self._get_chunk(array, 0).nbytes / 1e6:.1f} MiB'),
                          DaskRechunkWarning)
        return array

    def check_valid(self):
        assert isinstance(self._array, da.Array)

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
    def flatten_nav(slices, sig_dims):
        """
        Because LiberTEM partitions are set up with a flat nav dimension
        we must flatten the Dask array slices. This is ensured to be possible
        by earlier calls to _adapt_chunking but should be removed if ever
        partitions are able to have >1D navigation axes.
        """
        assert all([s.start == 0 for s in slices[1:]]),\
            'Only support chunking in first dimension for compatibility'
        nav_slices = slices[:-sig_dims]
        sig_slices = slices[-sig_dims:]
        start_frame = nav_slices[0].start * np.prod([s.stop - s.start for s in nav_slices[1:]])
        end_frame = start_frame + np.prod([s.stop - s.start for s in nav_slices])
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
            flat_slices, start_frame, end_frame = self.flatten_nav(full_slices, self._sig_dims)
            flat_slice = Slice(origin=self.slices_to_origin(flat_slices),
                               shape=Shape(self.slices_to_shape(flat_slices),
                                           sig_dims=self._sig_dims))
            # This only works if the Dask chunking is contiguous in
            # the first dimension, will not work for true blocks
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
