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

    def astype(self, dtype):
        return DataTile(data=self.data.astype(dtype), tile_slice=self.tile_slice)

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
