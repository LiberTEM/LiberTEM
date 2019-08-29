import itertools

import numpy as np

from libertem.io.utils import get_partition_shape
from libertem.common import Slice, Shape
from libertem.common.buffers import bytes_aligned, zeros_aligned


def _roi_to_indices(roi, start, stop):
    """
    helper function to calculate indices from roi mask

    roi : numpy.ndarray of type bool, matching the navigation shape of the dataset

    start : int
        start frame index, relative to dataset start
        can for example be the start frame index of a partition

    stop : int
        stop frame index, relative to dataset start
        can for example be the stop frame index of a partition
    """
    roi = roi.reshape((-1,))
    frames_in_roi = np.count_nonzero(roi)
    total = 0
    for flag, idx in zip(roi[start:stop], range(start, stop)):
        if flag:
            yield idx
            # early exit: we know we don't have more frames in the roi
            total += 1
            if total == frames_in_roi:
                break


class IOCaps:
    """
    I/O capabilities for a dataset (may depend on dataset parameters and concrete format)
    """
    ALL_CAPS = {
        "MMAP",             # .mmap is implemented on the file subclass
        "DIRECT",           # supports direct reading
        "FULL_FRAMES",      # can read full frames
        "SUBFRAME_TILES",   # can read tiles that slice frames into pieces
        "FRAME_CROPS",      # can efficiently crop on signal dimension without needing mmap
    }

    def __init__(self, caps):
        """
        create new capability set
        """
        caps = set(caps)
        for cap in caps:
            self._validate_cap(cap)
        self._caps = caps

    def _validate_cap(self, cap):
        if cap not in self.ALL_CAPS:
            raise ValueError("invalid I/O capability: %s" % cap)

    def __contains__(self, cap):
        return cap in self._caps

    def __getstate__(self):
        return {"caps": self._caps}

    def __setstate__(self, state):
        self._caps = state["caps"]

    def add(self, *caps):
        for cap in caps:
            self._validate_cap(cap)
        self._caps = self._caps.union(caps)

    def remove(self, *caps):
        for cap in caps:
            self._validate_cap(cap)
        self._caps = self._caps.difference(caps)


class DataSetException(Exception):
    pass


class FileTree(object):
    def __init__(self, low, high, value, idx, left, right):
        self.low = low
        self.high = high
        self.value = value
        self.idx = idx
        self.left = left
        self.right = right

    @classmethod
    def make(cls, files):
        """
        build a balanced binary tree by bisecting the files list
        """

        def _make(files):
            if len(files) == 0:
                return None
            mid = len(files) // 2
            idx, value = files[mid]

            return FileTree(
                low=value.start_idx,
                high=value.end_idx,
                value=value,
                idx=idx,
                left=_make(files[:mid]),
                right=_make(files[mid + 1:]),
            )
        return _make(list(enumerate(files)))

    def search_start(self, value):
        """
        search a node that has start_idx <= value && end_idx > value
        """
        if self.low <= value and self.high > value:
            return self.idx, self.value
        elif self.low > value:
            return self.left.search_start(value)
        else:
            return self.right.search_start(value)


class DataSet(object):
    def __init__(self):
        self._cores = 1

    def initialize(self):
        """
        pre-load metadata. this will be executed on a worker node. should return self.
        """
        raise NotImplementedError()

    def set_num_cores(self, cores):
        self._cores = cores

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
        The shape of the DataSet, as it makes sense for the application domain
        (for example, 4D for pixelated STEM)
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

    @classmethod
    def get_msg_converter(cls):
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
            minimum number of partitions desired. Defaults to the number of workers in the cluster.
        Returns
        -------
        (int, int, int, int)
            the shape calculated from the given parameters
        """
        if min_num_partitions is None:
            min_num_partitions = self._cores
        return get_partition_shape(datashape, framesize, dtype, target_size,
                                   min_num_partitions)


class DataSetMeta(object):
    def __init__(self, shape, raw_dtype=None, metadata=None, iocaps=None):
        self.shape = shape
        self.raw_dtype = np.dtype(raw_dtype)
        self.metadata = metadata
        if iocaps is None:
            iocaps = {}
        self.iocaps = IOCaps(iocaps)

    def __getitem__(self, key):
        return self.metadata[key]


class Partition(object):
    def __init__(self, meta, partition_slice):
        self.meta = meta
        self.slice = partition_slice
        assert partition_slice.shape.nav.dims == 1, "nav dims should be flat"

    @property
    def dtype(self):
        return self.meta.raw_dtype

    @property
    def shape(self):
        """
        the shape of the partition; dimensionality depends on format
        """
        return self.slice.shape.flatten_nav()

    def get_tiles(self, crop_to=None, full_frames=False, mmap=False, dest_dtype="float32",
                  roi=None, target_size=None):
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

        mmap : boolean, default False
            enable mmap if possible (not guaranteed to be supported by dataset)

        dest_dtype : numpy dtype
            convert data to this dtype when reading

        roi : numpy.ndarray
            1d mask that matches the dataset navigation shape to limit the region to work on.
            With a ROI, we yield tiles from a "compressed" navigation axis, relative to
            the beginning of the dataset. Compressed means, only frames that have a 1
            in the ROI are considered, and the resulting tile slices are from a coordinate
            system that has the shape `(np.count_nonzero(roi),)`.
        target_size : int
            Target size for each tile in bytes.
        """
        raise NotImplementedError()

    def get_macrotile(self, mmap=False, dest_dtype="float32", roi=None):
        '''
        Return a single tile for the entire partition.

        This is useful to support process_partiton() in UDFs and to construct dask arrays
        from datasets.
        '''
        try:
            return next(self.get_tiles(
                full_frames=True, mmap=mmap, dest_dtype=dest_dtype, roi=roi,
                target_size=float('inf')
            ))
        except StopIteration:
            tile_slice = Slice(
                origin=(self.slice.origin[0], 0, 0),
                shape=Shape((0,) + tuple(self.slice.shape.sig), sig_dims=2),
            )
            return DataTile(
                data=np.zeros(tile_slice.shape, dtype=dest_dtype),
                tile_slice=tile_slice
            )

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
        assert tile_slice.shape.nav.dims == 1, "DataTile should have flat nav"
        assert data.shape == tuple(tile_slice.shape),\
            "shape mismatch: data=%s, tile_slice=%s" % (data.shape, tile_slice.shape)

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
        """
        number of frames contained in this file
        """
        raise NotImplementedError()

    @property
    def start_idx(self):
        """
        global start index of frames contained in this file
        """
        raise NotImplementedError()

    def readinto(self, start, stop, out, crop_to=None):
        """
        Read a number of frames into an existing buffer

        Note: this method is not thread safe!

        Parameters
        ----------

        start : int
            file-local index of first frame to read
        stop : int
            file-local end index
        out : buffer
            output buffer that should fit `stop - start` frames
        crop_to : Slice
            crop to the signal part of this Slice
        """
        raise NotImplementedError()

    def mmap(self):
        """
        return a memory mapped array of this file
        """
        raise NotImplementedError()

    @property
    def end_idx(self):
        return self.start_idx + self.num_frames

    def open(self):
        pass

    def close(self):
        pass

    def get_buffer(self, name, size):
        """
        Get a buffer of `size` bytes. cache buffers by key (`size`, `name`) for efficient re-use.
        """
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


class FileSet3D(object):
    def __init__(self, files):
        """
        Parameters
        ----------
        files : list of File3D
            files that are part of a partition or dataset
        """
        self._files = files
        assert len(files) > 0
        self._tree = FileTree.make(files)
        if self._tree is None:
            raise ValueError(str(files))
        self._files_open = False

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
        if not self._files_open:
            raise RuntimeError(
                "only call read_images_multifile when having the "
                "fileset opened via with-statement"
            )
        frames_read = 0
        for f in self.files_from(start):
            # after the range of interest
            if f.start_idx > stop or frames_read == stop - start:
                break
            f_start = max(0, start - f.start_idx)
            f_stop = min(stop, f.end_idx) - f.start_idx

            # slice output buffer to the part that is contained in the current file
            buf = out[
                frames_read:frames_read + (f_stop - f_start)
            ]
            f.readinto(start=f_start, stop=f_stop, out=buf, crop_to=crop_to)

            frames_read += f_stop - f_start
        assert frames_read == out.shape[0]

    def __enter__(self):
        for f in self._files:
            f.open()
        self._files_open = True
        return self

    def __exit__(self, *exc):
        for f in self._files:
            f.close()
        self._files_open = False

    def __iter__(self):
        return iter(self._files)


class Partition3D(Partition):
    def __init__(self, fileset, start_frame, num_frames, stackheight=None,
                 *args, **kwargs):
        """
        Parameters
        ----------
        fileset : FileSet3D
            The files that are part of this partition (the FileSet3D may also contain files
            from the dataset which are not part of this partition, but that may harm performance)

        start_frame : int
            The index of the first frame of this partition (global coords)

        num_frames : int
            How many frames this partition should contain

        stackheight : int
            How many frames per tile? Default value, can be overridden by
            `target_size` in `get_tiles`.
        """
        self._fileset = fileset
        self._start_frame = start_frame
        self._num_frames = num_frames
        self._stackheight = stackheight
        super().__init__(*args, **kwargs)

    def _get_stackheight(self, sig_shape, dest_dtype, target_size=None):
        """
        Compute target stackheight

        The cropped tile of size `sig_shape` should fit into `target_size`,
        once converted to `dest_dtype`.
        """
        if target_size is None:
            if self._stackheight is not None:
                return self._stackheight
            target_size = 1 * 1024 * 1024
        framesize = sig_shape.size * dest_dtype.itemsize
        return int(min(max(1, np.floor(target_size / framesize)), self._num_frames))

    @classmethod
    def make_slices(cls, shape, num_partitions):
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
            yield part_slice, start, stop

    def get_tiles(self, crop_to=None, full_frames=False, mmap=False, dest_dtype="float32",
                  roi=None, target_size=None):
        """
        roi should be a 1d bitmask with the same shape as the navigation axis of the dataset
        """
        # TODO: implement reading tiles that contain parts of frames (and more depth)
        # the same notes as below in _get_tiles_normal apply; takes some work to make it efficient
        dest_dtype = np.dtype(dest_dtype)
        # disable mmap if type conversion takes place:
        if dest_dtype != self.meta.raw_dtype:
            mmap = False
        mmap = mmap and "MMAP" in self.meta.iocaps

        if (crop_to is not None and "FRAME_CROPS" not in self.meta.iocaps
                and tuple(crop_to.shape.sig) != tuple(self.meta.shape.sig)):
            raise ValueError("cannot crop in signal dimensions yet")
            # FIXME: not fully implemented; see _get_tiles_mmap
            # efficient impl:
            #  - read only as much as we need
            #    (hard, because that means strided reads → many syscalls! io_uring?)
            #  - no copying (except if dtype conversion is needed)
            # compromise:
            #  - only read full rows
            #  - only needs 1 read per frame
            #  - return view of cropped area into buffer that contains full rows
            # bad first impl possible:
            #  - read stackheight full frames
            #  - crop out the region we are interested in
            #  - can be implemented right here in this function actually!
            #  - stackheight needs to be limited, otherwise we thrash the CPU cache
            #    (we can't set stackheight to something that would fit the crop into the caches...)
        if (crop_to is not None
                and tuple(crop_to.shape.sig) != tuple(self.meta.shape.sig)
                and full_frames):
            raise ValueError("cannot crop and request full frames at the same time")
        if crop_to is not None and roi is not None:
            if crop_to.shape.nav.size != self._num_frames:
                raise ValueError("don't use crop_to with roi")

        if roi is not None:
            yield from self._get_tiles_with_roi(
                crop_to, full_frames, dest_dtype, roi, target_size=target_size)
        elif mmap:
            yield from self._get_tiles_mmap(
                crop_to, full_frames, dest_dtype)
        else:
            yield from self._get_tiles_normal(
                crop_to, full_frames, dest_dtype, target_size=target_size)

    def _get_tiles_normal(self, crop_to, full_frames, dest_dtype, target_size=None):
        start_at_frame = self._start_frame
        num_frames = self._num_frames
        sig_shape = self.meta.shape.sig
        sig_origin = tuple([0] * len(sig_shape))
        if crop_to is not None:
            sig_origin = tuple(crop_to.origin[-sig_shape.dims:])
            sig_shape = crop_to.shape.sig
        if full_frames:
            sig_shape = self.meta.shape.sig
        stackheight = self._get_stackheight(
            sig_shape=sig_shape, dest_dtype=dest_dtype, target_size=target_size)
        tile_buf_full = zeros_aligned((stackheight,) + tuple(sig_shape), dtype=dest_dtype)

        tileshape = (
            stackheight,
        ) + tuple(sig_shape)

        with self._fileset as fileset:
            for outer_frame in range(start_at_frame, start_at_frame + num_frames, stackheight):
                if start_at_frame + num_frames - outer_frame < stackheight:
                    end_frame = start_at_frame + num_frames
                    current_stackheight = end_frame - outer_frame
                    current_tileshape = (
                        current_stackheight,
                    ) + tuple(sig_shape)
                    tile_buf = zeros_aligned(current_tileshape, dtype=dest_dtype)
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
                fileset.read_images_multifile(
                    start=outer_frame,
                    stop=outer_frame + current_stackheight,
                    out=tile_buf,
                    crop_to=crop_to,
                )
                yield DataTile(
                    data=tile_buf,
                    tile_slice=tile_slice
                )

    def _get_tiles_with_roi(self, crop_to, full_frames, dest_dtype, roi, target_size=None):
        """
        With a ROI, we yield tiles from a "compressed" navigation axis, relative to
        the beginning of the partition. Compressed means, only frames that have a 1
        in the ROI are considered, and the resulting tile slices are from a coordinate
        system that has the shape `(np.count_nonzero(roi),)`.
        """
        start_at_frame = self._start_frame
        sig_shape = self.meta.shape.sig
        sig_origin = tuple([0] * len(sig_shape))
        if crop_to is not None:
            sig_origin = tuple(crop_to.origin[-sig_shape.dims:])
            sig_shape = crop_to.shape.sig
        if full_frames:
            sig_shape = self.meta.shape.sig
        stackheight = self._get_stackheight(
            sig_shape=sig_shape, dest_dtype=dest_dtype, target_size=target_size)
        tile_buf = zeros_aligned((stackheight,) + tuple(sig_shape), dtype=dest_dtype)

        frames_read = 0
        tile_idx = 0
        frame_idx = start_at_frame
        indices = _roi_to_indices(roi, start_at_frame, start_at_frame + self._num_frames)

        roi = roi.reshape((-1,))
        with self._fileset as fileset:
            outer_frame = 0
            frame_offset = np.count_nonzero(roi[:start_at_frame])
            for frame_idx in indices:
                fileset.read_images_multifile(
                    start=frame_idx,
                    stop=frame_idx + 1,
                    out=tile_buf[tile_idx].reshape((1,) + tuple(sig_shape)),
                    crop_to=crop_to,
                )

                tile_idx += 1
                frames_read += 1

                if tile_idx == stackheight:
                    tile_slice = Slice(
                        origin=(outer_frame + frame_offset,) + sig_origin,
                        shape=Shape((tile_idx,) + tuple(sig_shape), sig_dims=sig_shape.dims)
                    )
                    yield DataTile(
                        data=tile_buf[:tile_idx, ...],
                        tile_slice=tile_slice
                    )
                    tile_idx = 0
                    outer_frame = frames_read
        if tile_idx != 0:
            # last frame, size != stackheight
            tile_slice = Slice(
                origin=(outer_frame + frame_offset,) + sig_origin,
                shape=Shape((tile_idx,) + tuple(sig_shape), sig_dims=sig_shape.dims)
            )
            yield DataTile(
                data=tile_buf[:tile_idx, ...],
                tile_slice=tile_slice
            )

    def _get_tiles_mmap(self, crop_to, full_frames, dest_dtype):
        start_at_frame = self._start_frame
        num_frames = self._num_frames
        sig_shape = self.meta.shape.sig
        sig_origin = tuple([0] * len(sig_shape))
        if dest_dtype != self.meta.raw_dtype:
            raise ValueError("using mmap with dtype conversion is not efficient")
        if crop_to is not None:
            sig_origin = tuple(crop_to.origin[-sig_shape.dims:])
            sig_shape = crop_to.shape.sig
        if full_frames:
            sig_shape = self.meta.shape.sig

        fileset = self._fileset.get_for_range(
            start=start_at_frame,
            stop=start_at_frame + num_frames,
        )

        with self._fileset as fileset:
            for f in fileset:
                # global start/stop indices:
                start = max(f.start_idx, self._start_frame)
                stop = min(f.end_idx, self._start_frame + num_frames)

                tile_slice = Slice(
                    origin=(start,) + sig_origin,
                    shape=Shape((stop - start,) + tuple(sig_shape), sig_dims=sig_shape.dims)
                )
                arr = f.mmap()
                # limit to this partition (translate to file-local coords)
                arr = arr[start - f.start_idx:stop - f.start_idx]

                if crop_to is not None:
                    intersection = tile_slice.intersection_with(crop_to)
                    if intersection.is_null():
                        continue
                    # crop to, signal part:
                    arr = arr[(...,) + tile_slice.get(sig_only=True)]
                yield DataTile(
                    data=arr,
                    tile_slice=tile_slice
                )
