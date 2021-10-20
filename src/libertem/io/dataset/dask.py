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
    Parameters
    ----------

    path: str
        Path to the file

    nav_shape: tuple of int
        A n-tuple that specifies the size of the navigation region ((y, x), but
        can also be of length 1 for example for a line scan, or length 3 for
        a data cube, for example)

    sig_shape: tuple of int
        Common case: (height, width); but can be any dimensionality

    sync_offset: int, optional
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start

    dtype: numpy dtype
        The dtype of the data as it is on disk. Can contain endian indicator, for
        example >u2 for big-endian 16bit data.
    """
    def __init__(self, dask_array, nav_shape, sig_shape, io_backend=None):
        super().__init__(io_backend=io_backend)
        if io_backend is not None:
            raise ValueError("DaskDataSet currently doesn't support alternative I/O backends")

        self._nav_shape = tuple(nav_shape)
        self._sig_shape = tuple(sig_shape)
        if self._nav_shape is None:
            raise TypeError("missing 1 required argument: 'nav_shape'")
        if self._sig_shape is None:
            raise TypeError("missing 1 required argument: 'sig_shape'")
        self._array = dask_array
        self._sig_dims = len(self._sig_shape)
        self._dtype = self._array.dtype

    def _get_decoder(self):
        return None

    def get_io_backend(self):
        return DaskBackend()

    def initialize(self, executor):
        self._nav_shape_product = int(np.prod(self._nav_shape))
        self._image_count = self._nav_shape_product
        shape = Shape(self._nav_shape + self._sig_shape, sig_dims=self._sig_dims)
        self._meta = DataSetMeta(
            shape=shape,
            raw_dtype=np.dtype(self._dtype),
            sync_offset=0,
            image_count=self._nav_shape_product,
        )
        self._array = self._adapt_chunking(self._array)
        return self

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    @staticmethod
    def _adapt_chunking(array):
        chunk_sizes = array.chunks
        num_chunks_dim = [len(c) for c in chunk_sizes]
        if num_chunks_dim[0] == 1:
            print(f'chunks_per_dim: {num_chunks_dim}')
            warnings.warn('Array badly chunked, 0th dimension chunking is length 1')
        if any([n > 1 for n in num_chunks_dim[1:]]):
            array = array.rechunk({idx: -1 for idx, _ in enumerate(num_chunks_dim) if idx > 0})
            warnings.warn(f'Additional dimensions are chunked, this is currently not well handled, trying to merge\n\
Array rechunked from to {num_chunks_dim} blocks to {[len(c) for c in array.chunks]}')
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
        assert all([s.start == 0 for s in slices[1:]]),\
            'Only support chunking in first dimension for compatibility'
        nav_slices = slices[:-sig_dims]
        sig_slices = slices[-sig_dims:]
        start_frame = nav_slices[0].start * np.prod([s.stop - s.start for s in nav_slices[1:]])
        end_frame = start_frame + np.prod([s.stop - s.start for s in nav_slices])
        nav_slice = slice(start_frame, end_frame)
        return (nav_slice,) + sig_slices, start_frame, end_frame

    def get_slices(self):
        chunks = self._array.chunks
        boundaries = tuple(tuple(self.chunks_to_slices(chunk_lengths)) for chunk_lengths in chunks)
        chunk_slices = tuple(itertools.product(*boundaries))

        for full_slices in chunk_slices:
            flat_slices, start_frame, end_frame = self.flatten_nav(full_slices, self._sig_dims)
            part_slices = Slice(origin=self.slices_to_origin(flat_slices),
                                shape=Shape(self.slices_to_shape(flat_slices),
                                            sig_dims=self._sig_dims))
            # This only works if the Dask chunking is contiguous in
            # the first dimension, will not work for true blocks
            yield full_slices, part_slices, start_frame, end_frame

    def _get_fileset(self):
        partitions = []
        for full_slices, part_slice, start, stop in self.get_slices():
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
        fileset = self._get_fileset()
        for full_slices, part_slice, start, stop in self.get_slices():
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
        return f"<DaskDataSet of {self.dtype} shape={self.shape}, \
n_blocks={[len(c) for c in self._array.chunks]}>"


class DaskFile(File):
    def __init__(self, *args, array_chunk=None, **kwargs):
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
