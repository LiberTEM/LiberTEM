import itertools

import numpy as np

from libertem.common import Slice, Shape
from libertem.corrections import CorrectionSet
from .tiling import DataTile, TilingScheme
from .meta import DataSetMeta
from .fileset import FileSet
from . import MMapBackend, IOBackend
from .decode import Decoder


class WritablePartition:
    def get_write_handle(self):
        raise NotImplementedError()

    def delete(self):
        raise NotImplementedError()


class Partition(object):
    """
    Parameters
    ----------
    meta
        The `DataSet`'s `DataSetMeta` instance

    partition_slice
        The partition slice in non-flattened form

    fileset
        The files that are part of this partition (the FileSet may also contain files
        from the dataset which are not part of this partition, but that may harm performance)

    io_backend
        The I/O backend to use for accessing this partition
    """
    def __init__(
        self, meta: DataSetMeta, partition_slice: Slice, io_backend: IOBackend,
    ):
        self.meta = meta
        self.slice = partition_slice
        self._io_backend = io_backend
        if partition_slice.shape.nav.dims != 1:
            raise ValueError("nav dims should be flat")

    @classmethod
    def make_slices(cls, shape, num_partitions, sync_offset=0):
        """
        partition a 3D dataset ("list of frames") along the first axis,
        yielding the partition slice, and additionally start and stop frame
        indices for each partition.
        """
        num_frames = shape.nav.size
        f_per_part = max(1, num_frames // num_partitions)
        c0 = itertools.count(start=0, step=f_per_part)
        c1 = itertools.count(start=f_per_part, step=f_per_part)
        for (start, stop) in zip(c0, c1):
            if start >= num_frames:
                break
            stop = min(stop, num_frames)
            part_slice = Slice(
                origin=(start,) + tuple([0] * shape.sig.dims),
                shape=Shape(((stop - start),) + tuple(shape.sig),
                            sig_dims=shape.sig.dims)
            )
            yield part_slice, start + sync_offset, stop + sync_offset

    def need_decode(self, read_dtype, roi, corrections):
        raise NotImplementedError()

    def set_io_backend(self, backend):
        raise NotImplementedError()

    def validate_tiling_scheme(self, tiling_scheme):
        pass

    def set_corrections(self, corrections: CorrectionSet):
        raise NotImplementedError()

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None):
        raise NotImplementedError()

    def get_base_shape(self, roi):
        raise NotImplementedError()

    def __repr__(self):
        return "<%s>" % (
            self.__class__.__name__,
        )

    @property
    def dtype(self):
        return self.meta.dtype

    @property
    def shape(self):
        """
        the shape of the partition; dimensionality depends on format
        """
        return self.slice.shape.flatten_nav()

    def get_macrotile(self, dest_dtype="float32", roi=None):
        raise NotImplementedError()

    def adjust_tileshape(self, tileshape, roi):
        """
        Final veto of the Partition in the tileshape negotiation process,
        make sure that corrections are taken into account!
        """
        raise NotImplementedError()

    def get_max_io_size(self):
        """
        Override this method to implement a custom maximum I/O size
        """
        return None

    def get_min_sig_size(self):
        """
        minimum signal size, in number of elements
        """
        return 4 * 4096 // np.dtype(self.meta.raw_dtype).itemsize

    def get_locations(self):
        raise NotImplementedError()

    def get_io_backend(self):
        return None


class BasePartition(Partition):
    """
    Base class with default implementations

    Parameters
    ----------
    meta
        The `DataSet`'s `DataSetMeta` instance

    partition_slice
        The partition slice in non-flattened form

    fileset
        The files that are part of this partition (the FileSet may also contain files
        from the dataset which are not part of this partition, but that may harm performance)

    start_frame
        The index of the first frame of this partition (global coords)

    num_frames
        How many frames this partition should contain

    io_backend
        The I/O backend to use for accessing this partition
    """
    def __init__(
        self, meta: DataSetMeta, partition_slice: Slice,
        fileset: FileSet, start_frame: int, num_frames: int,
        io_backend: IOBackend,
    ):
        super().__init__(meta=meta, partition_slice=partition_slice, io_backend=io_backend)
        if start_frame < self.meta.image_count:
            self._fileset = fileset.get_for_range(
                max(0, start_frame), max(0, start_frame + num_frames - 1)
            )
        self._start_frame = start_frame
        self._num_frames = num_frames
        self._corrections = CorrectionSet()
        if num_frames <= 0:
            raise ValueError("invalid number of frames: %d" % num_frames)

    def get_locations(self):
        # Allow using any worker by default
        return None

    def adjust_tileshape(self, tileshape, roi):
        return tileshape

    def get_base_shape(self, roi):
        return (1,) + (1,) * (self.shape.sig.dims - 1) + (self.shape.sig[-1],)

    def get_max_io_size(self):
        # delegate to I/O backend by default:
        io_backend = self.get_io_backend()
        if io_backend is None:
            return None  # default value is set in Negotiator
        io_backend = io_backend.get_impl()
        return io_backend.get_max_io_size()

    def get_macrotile(self, dest_dtype="float32", roi=None):
        '''
        Return a single tile for the entire partition.

        This is useful to support process_partiton() in UDFs and to construct dask arrays
        from datasets.
        '''

        tiling_scheme = TilingScheme.make_for_shape(
            tileshape=self.shape,
            dataset_shape=self.meta.shape,
        )

        try:
            return next(self.get_tiles(
                tiling_scheme=tiling_scheme,
                dest_dtype=dest_dtype,
                roi=roi,
            ))
        except StopIteration:
            tile_slice = Slice(
                origin=(self.slice.origin[0], 0, 0),
                shape=Shape((0,) + tuple(self.slice.shape.sig), sig_dims=2),
            )
            return DataTile(
                np.zeros(tile_slice.shape, dtype=dest_dtype),
                tile_slice=tile_slice,
                scheme_idx=0,
            )

    def need_decode(self, read_dtype, roi, corrections):
        io_backend = self.get_io_backend().get_impl()
        return io_backend.need_copy(
            decoder=self._get_decoder(),
            roi=roi,
            native_dtype=self.meta.raw_dtype,
            read_dtype=read_dtype,
            sync_offset=self.meta.sync_offset,
            corrections=corrections,
        )

    def _get_decoder(self) -> Decoder:
        return None

    def _get_read_ranges(self, tiling_scheme, roi=None):
        return self._fileset.get_read_ranges(
            start_at_frame=self._start_frame,
            stop_before_frame=min(self._start_frame + self._num_frames, self.meta.image_count),
            tiling_scheme=tiling_scheme,
            dtype=self.meta.raw_dtype,
            sync_offset=self.meta.sync_offset,
            roi=roi,
        )

    def _get_default_io_backend(self):
        import platform
        if platform.system() == "Windows":
            from libertem.io.dataset.base import BufferedBackend
            return BufferedBackend()
        return MMapBackend()

    def get_io_backend(self):
        if self._io_backend is None:
            return self._get_default_io_backend()
        return self._io_backend

    def set_corrections(self, corrections: CorrectionSet):
        self._corrections = corrections

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None):
        """
        Return a generator over all DataTiles contained in this Partition.

        Note
        ----
        The DataSet may reuse the internal buffer of a tile, so you should
        directly process the tile and not accumulate a number of tiles and then work
        on them.

        Parameters
        ----------

        tiling_scheme
            According to this scheme the data will be tiled

        dest_dtype : numpy dtype
            convert data to this dtype when reading

        roi : numpy.ndarray
            Boolean array that matches the dataset navigation shape to limit the region to work on.
            With a ROI, we yield tiles from a "compressed" navigation axis, relative to
            the beginning of the dataset. Compressed means, only frames that have a 1
            in the ROI are considered, and the resulting tile slices are from a coordinate
            system that has the shape `(np.count_nonzero(roi),)`.
        """
        if self._start_frame < self.meta.image_count:
            dest_dtype = np.dtype(dest_dtype)
            self.validate_tiling_scheme(tiling_scheme)
            read_ranges = self._get_read_ranges(tiling_scheme, roi)
            io_backend = self.get_io_backend().get_impl()

            yield from io_backend.get_tiles(
                tiling_scheme=tiling_scheme, fileset=self._fileset,
                read_ranges=read_ranges, roi=roi,
                native_dtype=self.meta.raw_dtype,
                read_dtype=dest_dtype,
                sync_offset=self.meta.sync_offset,
                decoder=self._get_decoder(),
                corrections=self._corrections,
            )

    def __repr__(self):
        return "<%s [%d:%d]>" % (
            self.__class__.__name__,
            self._start_frame, self._start_frame + self._num_frames
        )
