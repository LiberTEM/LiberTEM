import numpy as np

from libertem.common import Slice, Shape
from .datatile import DataTile


class WritablePartition:
    def get_write_handle(self):
        raise NotImplementedError()

    def delete(self):
        raise NotImplementedError()


class Partition(object):
    def __init__(self, meta, partition_slice):
        self.meta = meta
        self.slice = partition_slice
        assert partition_slice.shape.nav.dims == 1, "nav dims should be flat"

    @property
    def dtype(self):
        return self.meta.dtype

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
