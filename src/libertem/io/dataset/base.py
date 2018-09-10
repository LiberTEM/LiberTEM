class DataSetException(Exception):
    pass


class DataSet(object):
    def get_partitions(self):
        """
        Return a generator over all Partitions in this DataSet
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
