from __future__ import annotations
import os
import pathlib
import numpy as np
import typing

from libertem.common.shape import Shape
from libertem.common.slice import Slice
from libertem.io.dataset.base.tiling import DataTile
from libertem.common.math import prod
from libertem.io.dataset.base import (DataSet, DataSetException, DataSetMeta,
                                      BasePartition, File, FileSet)

from timepix_decoding import read_file_structure, full_timestamp, spans_as_frames
from epoch_ordering import compute_epoch

if typing.TYPE_CHECKING:
    from libertem.io.dataset.base.tiling_scheme import TilingScheme
    from libertem.executor.base import BaseJobExecutor


class Timepix3DataSet(DataSet):
    def __init__(self,
                 path: os.PathLike,
                 nav_shape: tuple[int, ...],
                 frame_times: np.ndarray | float | None = None,
                 cross_offset: int = 2,
                 sig_shape: tuple[int, int] = (512, 512),
                 sync_offset: int = 0,
                 io_backend=None):
        assert io_backend is None, 'Only default I/O backend supported'
        super().__init__(io_backend=io_backend)

        self._path = pathlib.Path(path)
        if not self._path.is_file():
            raise DataSetException(f'Cannot find the file specified {path}')

        self._nav_shape = tuple(nav_shape)
        if not all(isinstance(d, int) for d in self._nav_shape):
            raise DataSetException('Must supply tuple of ints for nav shape')

        if isinstance(frame_times, np.ndarray):
            # explicit array of frame times in seconds
            if not frame_times.shape[0] != prod(nav_shape):
                raise DataSetException('frame_times must be array of per-frame '
                                       'time spans matching nav_shape')
            self._frame_times = frame_times
        elif isinstance(frame_times, float):
            # dwell time in seconds
            boundaries = np.arange(prod(self._nav_shape) + 1) * frame_times
            self._frame_times = np.stack((boundaries[:-1], boundaries[1:]), axis=1)
        elif frame_times is None:
            # Will split file into prod(nav_shape) equal frames (approximately)
            self._frame_times = None
        else:
            raise DataSetException(f'Unrecognized frame_times argument {frame_times}')

        assert isinstance(cross_offset, int)
        assert cross_offset >= 0
        self._cross_offset = cross_offset

        sig_shape = tuple(sig_shape)
        if not all(isinstance(d, int) for d in sig_shape) or len(sig_shape) != 2:
            raise DataSetException('Must supply 2-tuple of ints for sig shape')
        self._sig_shape = tuple(s + 2 * self._cross_offset for s in sig_shape)
        self._sig_dims = len(self._sig_shape)

        assert sync_offset == 0, 'No support for sync_offset yet'
        self._sync_offset = sync_offset
        self._dtype = np.uint64

    def initialize(self, executor: BaseJobExecutor):
        self._filesize = executor.run_function(self._get_filesize)
        self._nav_shape_product = int(prod(self._nav_shape))
        # FIXME _image_count semantics
        self._image_count = self._nav_shape_product
        self._sync_offset_info = self.get_sync_offset_info()
        shape = Shape(self._nav_shape + self._sig_shape, sig_dims=self._sig_dims)
        _file_structure = read_file_structure(self._path, executor=executor)
        _file_structure = np.concatenate((_file_structure,
                                          np.zeros_like(_file_structure[:, :1])),
                                         axis=1)
        # Will assume that the global timer starts at 0 and
        # always rolls over at the full interval (2**16, 2**34 etc)
        # To support early rollovers will be extremely challenging
        # to handle robustly. In the test file the end of the control
        # packets does cause an early rollover, but here it's ignored
        epoch_interval = 2**34
        _rollover_epochs = compute_epoch(_file_structure[:, 0],
                                         interval=epoch_interval,
                                         look_back=10,
                                         threshold=0.2)
        # Compute epoch timestamp increments
        for epoch_number, slices in _rollover_epochs:
            for sl in slices:
                _file_structure[sl, 2] = epoch_number * epoch_interval
        # Add the timestamp increments to the raw timestamps
        _file_structure[:, 0] += _file_structure[:, 2]
        sorter = np.argsort(_file_structure[:, 0])
        self._file_structure = _file_structure[sorter, :]
        min_timestamp = self._file_structure[:, 0].min()
        max_timestamp = self._file_structure[:, 0].max()
        if self._frame_times is None:
            # This conversion is a little flawed because we don't have the
            # exact first/final timestamps and would have to do a better
            # parsing to get them, the implem of read_file_structure just tries
            # to get a timestamp very close to the beginning and end of the file
            boundaries = np.linspace(min_timestamp, max_timestamp,
                                     endpoint=True, num=self._nav_shape_product + 1)
            frame_times = np.stack((boundaries[:-1], boundaries[1:]), axis=1)
            int_frame_times = frame_times.astype(int)
        else:
            int_frame_times = full_timestamp(self._frame_times)
        self._meta = DataSetMeta(
            shape=shape,
            raw_dtype=np.dtype(self._dtype),
            sync_offset=self._sync_offset,
            image_count=self._nav_shape_product,
            metadata=dict(int_frame_times=int_frame_times,
                          file_structure=self._file_structure,
                          cross_offset=self._cross_offset),
        )
        return self

    def _get_filesize(self):
        return os.stat(self._path).st_size

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    def need_decode(self, *args, **kwargs) -> bool:
        return True

    def check_valid(self):
        try:
            fileset = self._get_fileset()
            backend = self.get_io_backend().get_impl()
            with backend.open_files(fileset):
                return True
        except (OSError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def _get_fileset(self):
        return Timepix3FileSet([
            Timepix3File(
                path=self._path,
                start_idx=0,
                end_idx=self._image_count,
                sig_shape=self.shape.sig,
                native_dtype=self._meta.raw_dtype,
            )
        ])

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in self.get_slices():
            yield Timepix3Partition(
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
                io_backend=self.get_io_backend(),
            )

    def __repr__(self):
        return f"<Timepix3DataSet of shape={self.shape}>"


class Timepix3Partition(BasePartition):
    def get_tiles(self, tiling_scheme: TilingScheme, dest_dtype="float32", roi=None):
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
        if self._start_frame >= self.meta.image_count:
            return

        dest_dtype = np.dtype(dest_dtype)
        tiling_scheme_adj = tiling_scheme.adjust_for_partition(self)
        self.validate_tiling_scheme(tiling_scheme_adj)

        filepath = self._fileset[0]._path
        frame_times = self.meta.metadata['int_frame_times']
        structure = self.meta.metadata['file_structure']
        start_frame = self._start_frame
        num_frames = self._num_frames
        cross_offset = self.meta.metadata['cross_offset']
        # Check if this is 1-beyond or actual end index ?
        end_frame = start_frame + num_frames
        sig_shape = tuple(self.meta.shape.sig)
        sig_dims = len(sig_shape)

        if roi is None:
            roi_slice = slice(None)
            # flattened nav coord
            frame_idcs = np.arange(start_frame, end_frame)
        else:
            roi_slice, = np.nonzero(roi.ravel()[start_frame: end_frame])
            if not roi_slice.size:
                return
            in_part = np.arange(roi_slice.size)
            roi_idc_offset = np.count_nonzero(roi.ravel()[:start_frame])
            frame_idcs = roi_idc_offset + in_part

        part_frame_times = frame_times[start_frame: end_frame]
        part_frame_times = part_frame_times[roi_slice]
        n_frames_roi = part_frame_times.shape[0]

        depth = tiling_scheme.depth
        for idx in range(0, n_frames_roi, depth):
            spans_for_tiles = part_frame_times[idx: idx + depth]
            tile_block = spans_as_frames(filepath, structure,
                                         spans_for_tiles,
                                         sig_shape, max_ooo=6400,
                                         as_dense=True,
                                         cross_offset=cross_offset)
            tile_block = tile_block.astype(np.dtype(dest_dtype))

            flat_nav_origin = frame_idcs[idx]
            for scheme_idx, scheme_slice in enumerate(tiling_scheme):
                origin = (flat_nav_origin,) + tuple(scheme_slice.origin)
                shape = (len(spans_for_tiles),) + tuple(scheme_slice.shape)

                tile_slice = Slice(
                    origin=origin,
                    shape=Shape(shape, sig_dims=sig_dims)
                )
                block_slices = (slice(None), *scheme_slice.get(sig_only=True))
                data = tile_block[block_slices]
                yield DataTile(
                    data,
                    tile_slice=tile_slice,
                    scheme_idx=scheme_idx,
                )


class Timepix3FileSet(FileSet):
    ...


class Timepix3File(File):
    def get_offsets_sizes(self, size: int):
        raise NotImplementedError('Undefined for sparse file')

    def get_array_from_memview(self, mem: memoryview, slicing):
        raise NotImplementedError('Undefined for sparse file')


if __name__ == '__main__':
    import libertem.api as lt
    from libertem.udf.sum import SumUDF
    from libertem.udf.sumsigudf import SumSigUDF

    data_path = pathlib.Path('~/Workspace/libertem_dev/data/timepix/'
                             'experimental_200kv/edge/edge1_000001.tpx3').expanduser()

    ctx = lt.Context.make_with('inline')
    ds = Timepix3DataSet(data_path, (10, 10))
    ds = ds.initialize(ctx.executor)
    ds.set_num_cores(4)

    roi = np.random.randint(0, 2, size=ds.meta.shape.nav, dtype=bool)
    udfs = (SumUDF(), SumSigUDF())
    res = ctx.run_udf(dataset=ds, udf=udfs, progress=True, roi=roi)

    import matplotlib.pyplot as plt
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax0.imshow(res[0]['intensity'].data)
    ax0.set_title(udfs[0].__class__.__name__)
    ax1.imshow(res[1]['intensity'].data)
    ax1.set_title(udfs[1].__class__.__name__)
    if roi is not None:
        ax2.imshow(roi)
    ax2.set_title('ROI')
    plt.show()
