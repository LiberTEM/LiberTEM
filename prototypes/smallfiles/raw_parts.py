import itertools
import glob
import re
import os

import numpy as np

from libertem.common import Slice, Shape
from libertem.io.dataset.base import DataSet, Partition, DataTile, DataSetException


def _get_files(path):
    """
    From one example file, make a pattern and return all files that are part of the dataset.

    If path is /foo/bar00007.bin, all files matching /foo/bar*.bin will be returned.

    The files will be sorted by alphanumerical order, so make sure to include leading zeros
    if you are using numbers!
    """
    path, ext = os.path.splitext(path)
    pattern = "%s*%s" % (
        re.sub(r'[0-9]+$', '', path),
        ext
    )
    print(pattern)
    files = list(sorted(glob.glob(pattern)))
    return files


class RawFilesDataSet(DataSet):
    """
    test dataset for raw dataset split over many small files

    current assumptions
    1) one frame per file
    2) hardcoded number of partitions (16?)
    """

    def __init__(self, path, nav_shape, sig_shape, file_shape, tileshape, dtype):
        self._path = path
        self._nav_shape = tuple(nav_shape)
        self._sig_shape = tuple(sig_shape)
        self._file_shape = tuple(file_shape)
        self._dtype = dtype
        self._sig_dims = len(sig_shape)
        self._tileshape = Shape(tileshape, sig_dims=self._sig_dims)

    def initialize(self):
        self._files = _get_files(self._path)
        return self

    def open_file(self, filename):
        f = np.memmap(filename, dtype=self.dtype, mode='r',
                      shape=tuple(self.raw_shape))
        return f

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return Shape(self._nav_shape + self._sig_shape, sig_dims=self._sig_dims)

    @property
    def raw_shape(self):
        return self.shape.flatten_nav()

    def check_valid(self):
        try:
            # FIXME: maybe validate existence of all files?
            self.open_file(self._files[0])
            # TODO: check file size match
            # TODO: try to read from file(s)?
            return True
        except (IOError, OSError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def get_partitions(self):
        ds_slice = Slice(origin=(0, 0, 0), shape=self.raw_shape)
        num_partitions = 16  # FIXME: remove hardcoded
        num_frames = self.raw_shape[0]
        partition_shape = Shape((num_frames // num_partitions,) + tuple(self.raw_shape.sig),
                                sig_dims=self._sig_dims)
        for pslice in ds_slice.subslices(partition_shape):
            files_slice = pslice.get(nav_only=True)
            assert len(files_slice) == 1
            pfiles = self._files[files_slice[0]]
            yield RawFilesPartition(
                tileshape=self._tileshape,
                file_shape=self._file_shape,
                dataset=self,
                dtype=self.dtype,
                partition_slice=pslice,
                files=pfiles,
            )

    def __repr__(self):
        return "<RawFileDataSet of %s shape=%s>" % (self.dtype, self.shape)


def grouper(iterable, n, fillvalue=None):
    """
    from https://stackoverflow.com/a/434411/540644
    """
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


class RawFilesPartition(Partition):
    def __init__(self, tileshape, files, file_shape, *args, **kwargs):
        self.tileshape = tileshape
        self.files = files
        self.file_shape = file_shape
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return "<RawFilesPartition: %s>" % (
            self.slice
        )

    # @profile
    def get_tiles(self, crop_to=None):
        stackheight = self.tileshape.nav.size
        tile_buffer = np.zeros(self.tileshape.flatten_nav(), dtype=self.dataset.dtype)
        files_by_tile = list(grouper(self.files, stackheight))
        sig_dims = self.tileshape.sig.dims
        dtype = self.dataset.dtype

        partition_origin = self.slice.origin
        tile_slice_shape = Shape(tile_buffer.shape, sig_dims=sig_dims)

        files_done = 0
        for files in files_by_tile:
            if len(files) != stackheight:
                # allocate different size buffer for "rest-tile":
                tile_buffer = np.zeros((len(files),) + tuple(self.dataset.raw_shape.sig),
                                       dtype=dtype)
                tile_slice_shape = Shape(tile_buffer.shape, sig_dims=sig_dims)
            origin = tuple(sum(x) for x in zip(partition_origin, (files_done, 0, 0)))
            tile_slice = Slice(
                origin=origin,
                shape=tile_slice_shape,
            )
            files_done += len(files)
            if crop_to is not None:
                intersection = tile_slice.intersection_with(crop_to)
                if intersection.is_null():
                    continue
            if stackheight == 1:
                with open(files[0], "rb") as f:
                    f.readinto(tile_buffer)
                yield DataTile(
                    data=tile_buffer,
                    tile_slice=tile_slice
                )
                continue
            else:
                for idx, fn in enumerate(files):
                    with open(fn, "rb") as f:
                        f.readinto(tile_buffer[idx, ...])

            yield DataTile(
                data=tile_buffer,
                tile_slice=tile_slice
            )
