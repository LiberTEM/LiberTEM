class DataSetException(Exception):
    pass


class DataSet(object):
    def get_partitions(self):
        raise NotImplementedError()

    def get_raw_data(self, slice_):
        for partition in self.get_partitions():
            if slice_.intersection_with(partition.slice).is_null():
                continue
            for tile in partition.get_tiles():
                if tile.tile_slice.intersection_with(slice_):
                    yield tile

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

    def get_tiles(self):
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
        assert data.shape == tile_slice.shape

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
