import itertools

import numpy as np

from libertem.common import Shape, Slice
from libertem.common.buffers import bytes_aligned
from .utils import FileTree
from .roi import _roi_to_indices
from .partition import Partition
from .datatile import DataTile


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

    @property
    def end_idx(self):
        return self.start_idx + self.num_frames

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
        files = self._get_files_for_range(start, stop)
        return self.__class__(files=files)

    def _get_files_for_range(self, start, stop):
        """
        return new list of files filtered for files having frames in the [start, stop) range
        """
        files = []
        for f in self.files_from(start):
            if f.start_idx > stop:
                break
            files.append(f)
        assert len(files) > 0
        return files

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
        self._fileset = fileset.get_for_range(start_frame, start_frame + num_frames - 1)
        self._start_frame = start_frame
        self._num_frames = num_frames
        self._stackheight = stackheight
        assert num_frames > 0, "invalid number of frames: %d" % num_frames
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
        tile_buf_full = self.empty((stackheight,) + tuple(sig_shape), dtype=dest_dtype)

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
                    tile_buf = self.empty(current_tileshape, dtype=dest_dtype)
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

        frames_in_roi = np.count_nonzero(roi.reshape(-1,)[
            start_at_frame:start_at_frame + self._num_frames
        ])

        if frames_in_roi == 0:
            return

        tile_buf = self.zeros((stackheight,) + tuple(sig_shape), dtype=dest_dtype)

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

    def zeros(self, *args, **kwargs):
        return np.zeros(*args, **kwargs)

    def empty(self, *args, **kwargs):
        return np.empty(*args, **kwargs)
