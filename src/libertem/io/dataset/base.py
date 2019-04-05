import math
import collections

import numpy as np
import numba

from libertem.io.utils import get_partition_shape
from libertem.common import Slice, Shape
from libertem.common.buffers import bytes_aligned


class DataSetException(Exception):
    pass


class FileTree(object):
    def __init__(self, low, high, v, idx, l, r):
        self.low = low
        self.high = high
        self.v = v
        self.idx = idx
        self.l = l
        self.r = r

    @classmethod
    def make(cls, files):
        """
        build a balanced binary tree by bisecting the files list
        """

        def _make(files):
            if len(files) == 0:
                return None
            mid = len(files) // 2
            idx, v = files[mid]

            return FileTree(
                low=v.start_idx,
                high=v.end_idx,
                v=v,
                idx=idx,
                l=_make(files[:mid]),
                r=_make(files[mid + 1:]),
            )
        return _make(list(enumerate(files)))

    def search_start(self, value):
        """
        search a node that has start_idx <= value && end_idx > value
        """
        if self.low <= value and self.high > value:
            return self.idx, self.v
        elif self.low > value:
            return self.l.search_start(value)
        else:
            return self.r.search_start(value)


class DataSet(object):
    def initialize(self):
        """
        pre-load metadata. this will be executed on a worker node. should return self.
        """
        raise NotImplementedError()

    def get_partitions(self):
        """
        Return a generator over all Partitions in this DataSet
        """
        raise NotImplementedError()

    @property
    def dtype(self):
        """
        the destination data type
        """
        raise NotImplementedError()

    @property
    def raw_dtype(self):
        """
        the underlying data type
        """
        raise NotImplementedError()

    @property
    def shape(self):
        """
        the effective shape, for example imprinted by the scan_size parameter of some dataset impls
        """
        return self.raw_shape

    @property
    def raw_shape(self):
        """
        the "real" shape of the dataset, as it makes sense for the format
        """
        raise NotImplementedError()

    def check_valid(self):
        raise NotImplementedError()

    @classmethod
    def detect_params(cls, path):
        """
        Guess if path can be opened using this DataSet implementation and
        detect parameters.

        returns dict of detected parameters if path matches this dataset type,
        returns False if path is most likely not of a matching type.
        """
        # FIXME: return hints for the user and additional values,
        # for example number of signal elements
        raise NotImplementedError()

    @property
    def diagnostics(self):
        """
        Diagnistics common for all DataSet implementations
        """
        p = next(self.get_partitions())

        return self.get_diagnostics() + [
            {"name": "Partition shape",
             "value": str(p.shape)},

            {"name": "Number of partitions",
             "value": str(len(list(self.get_partitions())))}
        ]

    def get_diagnostics(self):
        """
        Get relevant diagnostics for this dataset, as a list of
        dicts with keys name, value, where value may be string or
        a list of dicts itself. Subclasses should override this method.
        """
        return []

    def partition_shape(self, datashape, framesize, dtype, target_size, min_num_partitions=None):
        """
        Calculate partition shape for the given ``target_size``
        Parameters
        ----------
        datashape : (int, int, int, int)
            size of the whole dataset
        framesize : int
            number of pixels per frame
        dtype : numpy.dtype or str
            data type of the dataset
        target_size : int
            target size in bytes - how large should each partition be?
        min_num_partitions : int
            minimum number of partitions desired, defaults to twice the number of CPU cores
        Returns
        -------
        (int, int, int, int)
            the shape calculated from the given parameters
        """
        return get_partition_shape(datashape, framesize, dtype, target_size,
                                   min_num_partitions)


class DataSetMeta(object):
    def __init__(self, shape, raw_shape, dtype, raw_dtype=None):
        self.shape = shape
        self.raw_shape = raw_shape
        self.dtype = np.dtype(dtype)
        if raw_dtype is None:
            raw_dtype = dtype
        self.raw_dtype = np.dtype(raw_dtype)


class Partition(object):
    def __init__(self, meta, partition_slice):
        self.meta = meta
        self.slice = partition_slice

    @property
    def dtype(self):
        return self.meta.dtype

    @property
    def shape(self):
        """
        the shape of the partition; dimensionality depends on format
        """
        return self.slice.shape

    def get_tiles(self, crop_to=None, full_frames=False):
        """
        Return a generator over all DataTiles contained in this Partition.

        Note
        ----
        The DataSet may reuse the internal buffer of a tile, so you should
        directly process the tile and not accumulate a number of tiles and then work
        on them.

        Parameters
        ----------

        crop_to : Slice or None
            crop to this slice. datasets may impose additional limits to the shape of the slice

        full_frames : boolean, default False
            always read full frames, not stacks of crops of frames
        """
        raise NotImplementedError()

    def get_locations(self):
        # Allow using any worker by default
        return None


class DataTile(object):
    __slots__ = ["data", "tile_slice"]

    def __init__(self, data, tile_slice):
        """
        A unit of data that can easily be processed at once, for example using
        one of the BLAS routines. For large frames, this may be a stack of sub-frame
        tiles.

        Parameters
        ----------
        tile_slice : Slice
            the global coordinates for this data tile

        data : numpy.ndarray
            the data corresponding to the origin/shape of tile_slice
        """
        self.data = data
        self.tile_slice = tile_slice
        assert hasattr(tile_slice.shape, "to_tuple")
        # FIXME: the world isn't ready for awesomeness yet, saving this for the future:
        # assert tile_slice.shape.nav.dims == 1, "DataTile should have flat nav"
        assert data.shape == tuple(tile_slice.shape),\
            "shape mismatch: data=%s, tile_slice=%s" % (data.shape, tile_slice.shape)

    @property
    def flat_nav(self):
        """
        Flatten the nav axis of the data.
        """
        shape = self.tile_slice.shape
        tileshape = (
            shape.nav.size,    # stackheight, number of frames we process at once
        ) + tuple(shape.sig)
        return self.data.reshape(tileshape)

    @property
    def flat_data(self):
        """
        Flatten the data.

        The result is a 2D array where each row contains pixel data
        from a single frame. It is just a reshape, so it is a view into
        the original data.
        """
        shape = self.tile_slice.shape
        tileshape = (
            shape.nav.size,    # stackheight, number of frames we process at once
            shape.sig.size,    # framesize, number of pixels per tile
        )
        return self.data.reshape(tileshape)

    def __repr__(self):
        return "<DataTile %r>" % self.tile_slice

    def __getstate__(self):
        return {
            k: getattr(self, k)
            for k in self.__slots__
        }

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)


class File3D(object):
    def __init__(self):
        self._buffers = {}

    @property
    def num_frames(self):
        raise NotImplementedError()

    @property
    def start_idx(self):
        raise NotImplementedError()

    def readinto(self, start, stop, out, crop_to=None):
        raise NotImplementedError()

    @property
    def end_idx(self):
        return self.start_idx + self.num_frames

    def open(self):
        pass

    def close(self):
        pass

    def get_buffer(self, name, size):
        k = (name, size)
        b = self._buffers.get(k, None)
        if b is None:
            b = bytes_aligned(size)
            self._buffers[k] = b
        return b

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()
        return self


class FileSet3D(object):
    def __init__(self, files):
        """
        Parameters
        ----------
        files : list of File3D
            files that are part of a partition or dataset
        """
        self._files = files
        self._tree = FileTree.make(files)
        if self._tree is None:
            raise ValueError(str(files))

    def get_for_range(self, start, stop):
        """
        return new FileSet3D filtered for files having frames in the [start, stop) range
        """
        files = []
        for f in self.files_from(start):
            if f.start_idx > stop:
                break
            files.append(f)
        assert len(files) > 0
        return self.__class__(files=files)

    def files_from(self, start):
        lower_bound, f = self._tree.search_start(start)
        for idx in range(lower_bound, len(self._files)):
            yield self._files[idx]

    def read_images_multifile(self, start, stop, out, crop_to=None):
        """
        read frames starting at index `start` up to but not including index `stop`
        from multiple files into the buffer `out`.
        """
        frames_read = 0
        for f in self.files_from(start):
            # after the range of interest
            if f.start_idx > stop:
                break
            with f:
                f_start = max(0, start - f.start_idx)
                f_stop = min(stop, f.end_idx) - f.start_idx

                # slice output buffer to the part that is contained in the current file
                buf = out[
                    frames_read:frames_read + (f_stop - f_start)
                ]
                f.readinto(start=f_start, stop=f_stop, out=buf, crop_to=crop_to)

            frames_read += f_stop - f_start
        assert frames_read == out.shape[0]


class Partition3D(Partition):
    def __init__(self, fileset, start_frame, num_frames, stackheight=None, *args, **kwargs):
        """
        Parameters
        ----------

        fileset : FileSet3D
        """
        self._fileset = fileset
        self._start_frame = start_frame
        self._num_frames = num_frames
        self._stackheight = stackheight
        super().__init__(*args, **kwargs)

    def _get_stackheight(self, sig_shape, target_dtype, target_size=1 * 1024 * 1024):
        """
        Compute target stackheight

        The cropped tile of size `sig_shape` should fit into `target_size`,
        once converted to `target_dtype`.
        """
        # FIXME: centralize this decision and make target_size tunable
        if self._stackheight is not None:
            return self._stackheight
        framesize = sig_shape.size * target_dtype.itemsize
        return max(1, math.floor(target_size / framesize))

    def get_tiles(self, crop_to=None, full_frames=False):
        start_at_frame = self._start_frame
        num_frames = self._num_frames
        dtype = self.meta.dtype
        sig_shape = self.meta.shape.sig
        sig_origin = tuple([0] * len(sig_shape))
        if crop_to is not None:
            sig_origin = tuple(crop_to.origin[-sig_shape.dims:])
            sig_shape = crop_to.shape.sig
        stackheight = self._get_stackheight(sig_shape=sig_shape, target_dtype=self.meta.dtype)
        tile_buf_full = np.zeros((stackheight,) + tuple(sig_shape), dtype=dtype)

        if (crop_to is not None
                and tuple(crop_to.shape.sig) != tuple(self.meta.shape.sig)
                and full_frames):
            raise ValueError("cannot crop and request full frames at the same time")

        tileshape = (
            stackheight,
        ) + tuple(sig_shape)

        for outer_frame in range(start_at_frame, start_at_frame + num_frames, stackheight):
            if start_at_frame + num_frames - outer_frame < stackheight:
                end_frame = start_at_frame + num_frames
                current_stackheight = end_frame - outer_frame
                current_tileshape = (
                    current_stackheight,
                ) + tuple(sig_shape)
                tile_buf = np.zeros(current_tileshape, dtype=dtype)
            else:
                current_stackheight = stackheight
                current_tileshape = tileshape
                tile_buf = tile_buf_full
            tile_slice = Slice(
                origin=(outer_frame,) + sig_origin,
                shape=Shape(current_tileshape, sig_dims=sig_shape.dims)
            )
            if crop_to is not None:
                intersection = tile_slice.intersection_with(crop_to)
                if intersection.is_null():
                    continue
            self._fileset.read_images_multifile(
                start=outer_frame,
                stop=outer_frame + current_stackheight,
                out=tile_buf,
                crop_to=crop_to,
            )
            yield DataTile(
                data=tile_buf,
                tile_slice=tile_slice
            )
