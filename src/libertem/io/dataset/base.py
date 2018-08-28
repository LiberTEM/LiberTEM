class DataSetException(Exception):
    pass


class DataSet(object):
    def get_partitions(self):
        """
        Return a generator over all `Partition`s in this `DataSet`
        """
        raise NotImplementedError()

    @property
    def dtype(self):
        raise NotImplementedError()

    @property
    def shape(self):
        raise NotImplementedError()

    def check_valid(self):
        raise NotImplementedError()


class Partition(object):
    def __init__(self, dataset, dtype, partition_slice):
        self.dataset = dataset
        self.dtype = dtype
        self.slice = partition_slice

    @property
    def shape(self):
        return self.slice.shape

    def get_tiles(self, crop_to=None):
        """
        Return a generator over all `DataTile`s contained in this Partition. Note that the DataSet
        may reuse the internal buffer of a tile, so you should directly process the tile.

        right:
        >>> tile_iter = p.get_tiles()
        >>> for tile in tile_iter:
        >>>     do_stuff_with(tile)

        wrong:
        >>> tile_iter = p.get_tiles()
        >>> some_tiles = [next(tile_iter), next(tile_iter), next(tile_iter)]
        >>> do_stuff_with(some_tiles)
        # the internal buffer of all three tiles may contain the same data at this point


        Parameters
        ----------

        crop_to : Slice or None
            crop to this slice. datasets may impose additional limits to the shape of the slice
        """
        raise NotImplementedError()

    def get_locations(self):
        raise NotImplementedError()


class DataTile(object):
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
        assert data.shape == tile_slice.shape,\
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
            shape[0] * shape[1],    # stackheight, number of frames we process at once
            shape[2] * shape[3],    # framesize, number of pixels per tile
        )
        return self.data.reshape(tileshape)

    def __repr__(self):
        return "<DataTile %r>" % self.tile_slice
