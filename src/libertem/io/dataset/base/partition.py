import warnings
from typing import Optional, TYPE_CHECKING

import numpy as np
from sparseconverter import ArrayBackend, for_backend

from libertem.common import Slice, Shape
from libertem.common.math import count_nonzero
from libertem.io.corrections import CorrectionSet
from .tiling import DataTile
from .tiling_scheme import TilingScheme
from .meta import DataSetMeta
from .fileset import FileSet
from . import IOBackend
from .decode import Decoder
from .roi import roi_for_partition


if TYPE_CHECKING:
    from libertem.common.executor import WorkerContext


class WritablePartition:
    def get_write_handle(self):
        raise NotImplementedError()

    def delete(self):
        raise NotImplementedError()


class Partition:
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

    decoder
        The decoder that needs to be used for decoding this partition's data
    """
    def __init__(
        self,
        meta: DataSetMeta,
        partition_slice: Slice,
        io_backend: IOBackend,
        decoder: Optional[Decoder],
    ):
        self.meta = meta
        self.slice = partition_slice
        self._io_backend = io_backend
        self._decoder = decoder
        self._idx: Optional[int] = None
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
        if num_partitions > num_frames:
            warnings.warn(
                "dataset contains fewer frames than specified partitions, "
                f"setting num_partitions == num_frames == {num_frames} "
                "to avoid creating empty partitions",
                RuntimeWarning
            )
            num_partitions = num_frames
        boundaries = np.linspace(
            0,
            num_frames,
            num=max(2, num_partitions + 1),
            endpoint=True,
            dtype=int,
        )

        # Cast explicitly to tuple[int, ...] to avoid pickle/JSON errors
        boundaries = tuple(map(int, boundaries))

        for (start, stop) in zip(boundaries[:-1], boundaries[1:]):
            part_slice = Slice(
                origin=(start,) + tuple([0] * shape.sig.dims),
                shape=Shape(((stop - start),) + tuple(shape.sig),
                            sig_dims=shape.sig.dims)
            )
            yield part_slice, start + sync_offset, stop + sync_offset

    def set_io_backend(self, backend):
        raise NotImplementedError()

    def validate_tiling_scheme(self, tiling_scheme):
        pass

    def set_corrections(self, corrections: CorrectionSet):
        raise NotImplementedError()

    def set_worker_context(self, worker_context: "WorkerContext"):
        pass

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None,
            array_backend: Optional[ArrayBackend] = None):
        raise NotImplementedError()

    def __repr__(self):
        return "<{}>".format(
            self.__class__.__name__,
        )

    @property
    def dtype(self):
        return self.meta.dtype

    @property
    def shape(self) -> Shape:
        """
        the shape of the partition; dimensionality depends on format
        """
        return self.slice.shape.flatten_nav()

    def get_macrotile(self, dest_dtype="float32", roi=None,
            array_backend: Optional[ArrayBackend] = None):
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
            tiles = self.get_tiles(
                tiling_scheme=tiling_scheme,
                dest_dtype=dest_dtype,
                roi=roi,
                array_backend=array_backend,
            )
            tile = next(tiles)
            # NOTE: run the generator to completion, but there must not be any
            # more tiles than the one!
            rest = list(tiles)
            assert len(rest) == 0
            return tile
        except StopIteration:
            tile_slice = Slice(
                origin=(self.slice.origin[0], 0, 0),
                shape=Shape((0,) + tuple(self.slice.shape.sig), sig_dims=2),
            )
            empty = np.zeros(tile_slice.shape, dtype=dest_dtype)
            return DataTile(
                for_backend(empty, array_backend, strict=False),
                tile_slice=tile_slice,
                scheme_idx=0,
            )

    def get_locations(self):
        raise NotImplementedError()

    def get_io_backend(self):
        return None

    def set_idx(self, idx: int):
        self._idx = idx

    def get_ident(self) -> str:
        return f'part-{self._idx}'

    def get_frame_count(self, roi: Optional[np.ndarray] = None) -> int:
        if roi is None:
            return self.shape.nav.size
        else:
            return count_nonzero(roi_for_partition(roi, self))


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
        decoder: Optional[Decoder] = None,
    ):
        super().__init__(
            meta=meta,
            partition_slice=partition_slice,
            io_backend=io_backend,
            decoder=decoder,
        )
        if start_frame < self.meta.image_count:
            self._fileset = fileset.get_for_range(
                max(0, start_frame), max(0, start_frame + num_frames - 1)
            )
        self._start_frame = start_frame
        self._num_frames = num_frames
        self._corrections = CorrectionSet()
        self._worker_context: Optional["WorkerContext"] = None
        if num_frames <= 0:
            raise ValueError("invalid number of frames: %d" % num_frames)

    def get_locations(self):
        # Allow using any worker by default
        return None

    def get_max_io_size(self):
        # delegate to I/O backend by default:
        io_backend = self.get_io_backend()
        if io_backend is None:
            return None  # default value is set in Negotiator
        io_backend = io_backend.get_impl()
        return io_backend.get_max_io_size()

    def _get_read_ranges(self, tiling_scheme, roi=None):
        return self._fileset.get_read_ranges(
            start_at_frame=self._start_frame,
            stop_before_frame=min(self._start_frame + self._num_frames, self.meta.image_count),
            tiling_scheme=tiling_scheme,
            dtype=self.meta.raw_dtype,
            sync_offset=self.meta.sync_offset,
            roi=roi,
        )

    def get_io_backend(self):
        assert self._io_backend is not None
        return self._io_backend

    def set_corrections(self, corrections: Optional[CorrectionSet]):
        self._corrections = corrections

    def set_worker_context(self, worker_context: "WorkerContext"):
        self._worker_context = worker_context

    def get_tiles(self, tiling_scheme: TilingScheme, dest_dtype="float32",
            roi=None, array_backend: Optional[ArrayBackend] = None):
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

        array_backend : ArrayBackend
            Specify array backend to use. By default the first entry in the list of supported
            backends is used.

            .. versionadded:: 0.11.0
        """
        if self._start_frame < self.meta.image_count:
            dest_dtype = np.dtype(dest_dtype)
            tiling_scheme_adj = tiling_scheme.adjust_for_partition(self)
            self.validate_tiling_scheme(tiling_scheme_adj)
            read_ranges = self._get_read_ranges(tiling_scheme_adj, roi)
            io_backend = self.get_io_backend().get_impl()
            if array_backend is None:
                array_backend = self.meta.array_backends[0]

            yield from io_backend.get_tiles(
                tiling_scheme=tiling_scheme_adj, fileset=self._fileset,
                read_ranges=read_ranges, roi=roi,
                native_dtype=self.meta.raw_dtype,
                read_dtype=dest_dtype,
                sync_offset=self.meta.sync_offset,
                decoder=self._decoder,
                corrections=self._corrections,
                array_backend=array_backend,
            )

    def __repr__(self):
        return "<%s [%d:%d]>" % (
            self.__class__.__name__,
            self._start_frame, self._start_frame + self._num_frames
        )
